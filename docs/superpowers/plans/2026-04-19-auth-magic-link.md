# Magic Link Auth + Fly.io Deploy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add email-whitelist Magic Link authentication to Zoom to Text and deploy to Fly.io, so only approved users can access the app and each user sees only their own tasks.

**Architecture:** DB-backed sessions (random UUID stored in SQLite, sent as HttpOnly cookie). Magic link tokens expire in 15 minutes; sessions last 30 days. Resend.com sends the login email via httpx (already in requirements). All task endpoints require a valid session via a FastAPI `Depends` dependency.

**Tech Stack:** FastAPI, aiosqlite, httpx (Resend REST API), pytest + TestClient, Fly.io

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `app/config.py` | Modify | Add `allowed_emails`, `resend_api_key`, `cors_origin` settings |
| `app/state.py` | Modify | Add `users`, `magic_tokens`, `sessions` tables + CRUD; add `user_id` to `tasks` |
| `app/api/deps.py` | Create | `get_current_user()` FastAPI dependency |
| `app/api/auth.py` | Create | Auth router: request / verify / logout |
| `app/api/routes.py` | Modify | Add `Depends(get_current_user)` + filter tasks by `user_id` |
| `app/main.py` | Modify | Register auth router, tighten CORS, protect `/` route, add `/login` route |
| `static/login.html` | Create | Standalone login page (email input + status message) |
| `static/index.html` | Modify | Add logout button |
| `fly.toml` | Create | Fly.io deployment config |
| `tests/conftest.py` | Create | Shared pytest fixtures (temp DB, mock email) |
| `tests/test_auth.py` | Create | Auth flow tests |
| `tests/test_routes_protected.py` | Create | Route protection + user isolation tests |

---

## Task 1: Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add test dependencies to requirements.txt**

Append to the end of `requirements.txt`:

```
# ── Testing ─────────────────────────────────────────────────────────────────────
pytest==8.2.2
pytest-asyncio==0.23.8
anyio==4.4.0
```

- [ ] **Step 2: Create tests/__init__.py**

Create empty file `tests/__init__.py`.

- [ ] **Step 3: Create tests/conftest.py**

```python
"""
Shared fixtures for all tests.

Key design decisions:
- Each test gets a fresh SQLite DB in a temp directory.
- TestClient triggers the app lifespan (startup/shutdown), which calls init_db()
  and close_db(), so the DB is always in a clean state between tests.
- Resend HTTP calls are patched to avoid real network calls.
"""
import pytest
import app.state as state_module
from app.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient with an isolated temp database."""
    monkeypatch.setattr(state_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(state_module, "_db", None)
    monkeypatch.setattr(settings, "allowed_emails", "allowed@example.com")
    monkeypatch.setattr(settings, "resend_api_key", "test_key")
    monkeypatch.setattr(settings, "base_url", "http://testserver")
    monkeypatch.setattr(settings, "cors_origin", "http://testserver")

    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def mock_email(monkeypatch):
    """Capture sent magic links instead of calling Resend."""
    sent = []

    async def fake_send(email: str, token: str) -> None:
        sent.append({"email": email, "token": token})

    import app.api.auth as auth_module
    monkeypatch.setattr(auth_module, "_send_magic_link_email", fake_send)
    return sent
```

- [ ] **Step 4: Verify pytest collects (no tests yet)**

```bash
cd "C:\Users\אסף\zoom to text"
python -m pytest tests/ --collect-only
```

Expected: `no tests ran` (0 errors, 0 failures)

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/conftest.py requirements.txt
git commit -m "test: add pytest infrastructure and shared fixtures"
```

---

## Task 2: Config — new auth settings

**Files:**
- Modify: `app/config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth.py`:

```python
"""Tests for Magic Link authentication flow."""
from app.config import settings


def test_config_has_auth_fields():
    """Settings must expose the three new auth fields."""
    assert hasattr(settings, "allowed_emails")
    assert hasattr(settings, "resend_api_key")
    assert hasattr(settings, "cors_origin")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_auth.py::test_config_has_auth_fields -v
```

Expected: FAIL — `AttributeError` or `AssertionError`

- [ ] **Step 3: Add the new settings to app/config.py**

Inside the `Settings` class, after the `# ── App` block, add:

```python
    # ── Auth ────────────────────────────────────────────────────────────────────
    # Comma-separated list of emails allowed to log in
    # Example: "alice@example.com,bob@example.com"
    allowed_emails: str = ""
    resend_api_key: str = ""
    # Allowed CORS origin — set to your Fly.io domain in production
    cors_origin: str = "http://localhost:8000"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_auth.py::test_config_has_auth_fields -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_auth.py
git commit -m "feat: add allowed_emails, resend_api_key, cors_origin to config"
```

---

## Task 3: State — auth tables, migration, CRUD

**Files:**
- Modify: `app/state.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_auth.py`:

```python
import pytest


def test_magic_token_full_flow(client, mock_email):
    """Request a magic link, extract token, verify it — should create session cookie."""
    # Request magic link
    resp = client.post("/api/auth/request", json={"email": "allowed@example.com"})
    assert resp.status_code == 200

    # Token was "sent"
    assert len(mock_email) == 1
    token = mock_email[0]["token"]
    assert token  # non-empty UUID

    # Verify token — should redirect and set cookie
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    assert "session_id" in resp.cookies


def test_invalid_token_returns_400(client, monkeypatch):
    """verify endpoint returns 400 when token is invalid or expired."""
    import app.state as state_module

    async def fake_consume(token: str):
        return None  # simulate expired/invalid/used token

    monkeypatch.setattr(state_module, "consume_magic_token", fake_consume)
    resp = client.get("/api/auth/verify?token=any-token", follow_redirects=False)
    assert resp.status_code == 400


def test_magic_token_used_twice_rejected(client, mock_email):
    """A used token cannot be used again."""
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]

    # First use — OK
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302

    # Second use — rejected
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 400


def test_unknown_email_returns_generic_message(client, mock_email):
    """Unknown emails get the same response as known ones (no enumeration)."""
    resp = client.post("/api/auth/request", json={"email": "hacker@evil.com"})
    assert resp.status_code == 200  # same response
    assert len(mock_email) == 0  # but no email sent
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_auth.py -v
```

Expected: FAIL — endpoints don't exist yet (404 or ImportError)

- [ ] **Step 3: Add new tables and CRUD to app/state.py**

After the existing `CREATE_TABLE_SQL` constant, add:

```python
CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    email      TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
)
"""

CREATE_MAGIC_TOKENS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS magic_tokens (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    expires_at TEXT NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0
)
"""

CREATE_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)
"""
```

Replace the `init_db()` function with:

```python
async def init_db():
    """Create all tables on startup. Migrate tasks table. Mark interrupted tasks as failed."""
    db = await _get_db()
    await db.execute(CREATE_TABLE_SQL)
    await db.execute(CREATE_USERS_TABLE_SQL)
    await db.execute(CREATE_MAGIC_TOKENS_TABLE_SQL)
    await db.execute(CREATE_SESSIONS_TABLE_SQL)
    await db.commit()

    # Migrate: add user_id column to tasks if it doesn't exist yet
    async with db.execute("PRAGMA table_info(tasks)") as cursor:
        cols = [row[1] for row in await cursor.fetchall()]
    if "user_id" not in cols:
        await db.execute("ALTER TABLE tasks ADD COLUMN user_id TEXT")
        await db.commit()
        logger.info("Migrated tasks table: added user_id column")

    await _mark_interrupted_tasks_failed()
    logger.info(f"Database ready: {DB_PATH}")
```

At the top of `app/state.py`, add `import uuid` if not already present (it is not — add it after the existing imports):

```python
import uuid
```

After the existing CRUD section, add the new auth CRUD functions:

```python
# ── Auth CRUD ─────────────────────────────────────────────────────────────────────

async def get_or_create_user(email: str) -> str:
    """Return user_id for the email, creating the user row if this is their first login."""
    db = await _get_db()
    db.row_factory = aiosqlite.Row
    async with db.execute("SELECT id FROM users WHERE email=?", [email.lower()]) as cursor:
        row = await cursor.fetchone()
    if row:
        return row["id"]
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO users (id, email, created_at) VALUES (?,?,?)",
        [user_id, email.lower(), now],
    )
    await db.commit()
    return user_id


async def create_magic_token(user_id: str) -> str:
    """Create a 15-minute single-use token. Returns the token string."""
    from datetime import timedelta
    token = str(uuid.uuid4())
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    db = await _get_db()
    await db.execute(
        "INSERT INTO magic_tokens (token, user_id, expires_at) VALUES (?,?,?)",
        [token, user_id, expires_at],
    )
    await db.commit()
    return token


async def consume_magic_token(token: str) -> Optional[str]:
    """
    Validate and consume a magic token.
    Returns user_id if valid, None if expired/used/unknown.
    Marks the token used=1 on success.
    """
    db = await _get_db()
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT user_id, expires_at, used FROM magic_tokens WHERE token=?", [token]
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or row["used"]:
        return None
    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        return None
    await db.execute("UPDATE magic_tokens SET used=1 WHERE token=?", [token])
    await db.commit()
    return row["user_id"]


async def create_session(user_id: str) -> str:
    """Create a 30-day session. Returns the session_id (stored in cookie)."""
    from datetime import timedelta
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(days=30)).isoformat()
    db = await _get_db()
    await db.execute(
        "INSERT INTO sessions (id, user_id, created_at, expires_at) VALUES (?,?,?,?)",
        [session_id, user_id, now.isoformat(), expires_at],
    )
    await db.commit()
    return session_id


async def get_session_user(session_id: str) -> Optional[str]:
    """Return user_id if session exists and has not expired. None otherwise."""
    db = await _get_db()
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT user_id, expires_at FROM sessions WHERE id=?", [session_id]
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        return None
    return row["user_id"]


async def delete_session(session_id: str):
    """Delete a session (logout)."""
    db = await _get_db()
    await db.execute("DELETE FROM sessions WHERE id=?", [session_id])
    await db.commit()
```

- [ ] **Step 4: Commit state changes (tests still fail — auth endpoints not yet created)**

```bash
git add app/state.py
git commit -m "feat: add auth tables (users, magic_tokens, sessions) and CRUD to state"
```

---

## Task 4: Auth dependency

**Files:**
- Create: `app/api/deps.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_auth.py`:

```python
def test_unauthenticated_tasks_returns_401(client):
    """All task endpoints require authentication."""
    resp = client.get("/api/tasks")
    assert resp.status_code == 401

def test_authenticated_tasks_returns_200(client, mock_email):
    """After login, task endpoints are accessible."""
    # Login
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    session_cookie = resp.cookies["session_id"]

    # Use session
    resp = client.get("/api/tasks", cookies={"session_id": session_cookie})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_auth.py::test_unauthenticated_tasks_returns_401 -v
```

Expected: FAIL — currently returns 200 (no auth)

- [ ] **Step 3: Create app/api/deps.py**

```python
"""
FastAPI dependencies shared across routers.
"""
from typing import Optional

from fastapi import Cookie, HTTPException

from app import state


async def get_current_user(
    session_id: Optional[str] = Cookie(default=None),
) -> str:
    """
    Read session_id cookie and return the authenticated user_id.
    Raises HTTP 401 if the session is missing or expired.
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = await state.get_session_user(session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Session expired — please log in again")
    return user_id
```

- [ ] **Step 4: Commit**

```bash
git add app/api/deps.py
git commit -m "feat: add get_current_user dependency"
```

---

## Task 5: Auth router

**Files:**
- Create: `app/api/auth.py`

- [ ] **Step 1: Create app/api/auth.py**

(Tests for this were written in Task 3 Step 1 — they cover request/verify/logout)

```python
"""
Authentication endpoints.

POST /api/auth/request   — request a magic link email
GET  /api/auth/verify    — verify token, create session, redirect home
POST /api/auth/logout    — delete session, clear cookie
"""
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Cookie, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from app import state
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class MagicLinkRequest(BaseModel):
    email: str


@router.post("/auth/request")
async def request_magic_link(body: MagicLinkRequest):
    """
    Send a magic link to the given email if it is on the whitelist.
    Always returns the same response to prevent email enumeration.
    """
    email = body.email.strip().lower()
    allowed = {e.strip().lower() for e in settings.allowed_emails.split(",") if e.strip()}

    if email in allowed:
        user_id = await state.get_or_create_user(email)
        token = await state.create_magic_token(user_id)
        try:
            await _send_magic_link_email(email, token)
            logger.info(f"Magic link sent to {email}")
        except Exception as exc:
            logger.error(f"Failed to send magic link to {email}: {exc}")
            # Don't reveal the failure to the client
    else:
        logger.warning(f"Magic link requested for non-whitelisted email: {email}")

    return {"message": "אם המייל רשום במערכת, תקבל קישור כניסה תוך דקה."}


@router.get("/auth/verify")
async def verify_magic_link(token: str):
    """
    Validate the magic link token.
    On success: create a 30-day session, set HttpOnly cookie, redirect to /.
    On failure: return 400.
    """
    user_id = await state.consume_magic_token(token)
    if not user_id:
        raise HTTPException(status_code=400, detail="קישור לא תקין או פג תוקף. בקש קישור חדש.")

    session_id = await state.create_session(user_id)

    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=not settings.base_url.startswith("http://localhost"),
        samesite="lax",
        max_age=30 * 24 * 60 * 60,  # 30 days in seconds
    )
    return response


@router.post("/auth/logout")
async def logout(session_id: Optional[str] = Cookie(default=None)):
    """Delete the current session and clear the cookie."""
    if session_id:
        await state.delete_session(session_id)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session_id")
    return response


async def _send_magic_link_email(email: str, token: str) -> None:
    """Call the Resend API to send the magic link email."""
    magic_url = f"{settings.base_url}/api/auth/verify?token={token}"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {settings.resend_api_key}"},
            json={
                "from": "Zoom to Text <noreply@zoom-to-text.fly.dev>",
                "to": [email],
                "subject": "כניסה ל-Zoom to Text",
                "html": (
                    "<div dir='rtl' style='font-family:sans-serif;max-width:400px;margin:auto'>"
                    "<h2>כניסה ל-Zoom to Text</h2>"
                    "<p>לחץ על הכפתור להתחברות:</p>"
                    f"<a href='{magic_url}' style='display:inline-block;padding:12px 24px;"
                    "background:#6c63ff;color:#fff;border-radius:8px;text-decoration:none;"
                    "font-size:16px'>כניסה למערכת</a>"
                    "<p><small>הקישור תקף ל-15 דקות בלבד.<br>"
                    "אם לא ביקשת להתחבר, התעלם מהודעה זו.</small></p>"
                    "</div>"
                ),
            },
            timeout=10.0,
        )
        resp.raise_for_status()
```

- [ ] **Step 2: Run the auth tests**

```bash
python -m pytest tests/test_auth.py -v
```

Expected: most tests still FAIL because auth router is not registered in main.py yet.

- [ ] **Step 3: Commit**

```bash
git add app/api/auth.py
git commit -m "feat: add magic link auth router (request/verify/logout)"
```

---

## Task 6: Update state — user_id in create_task and list_tasks

**Files:**
- Modify: `app/state.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_routes_protected.py`:

```python
"""Tests for route protection and user data isolation."""
import pytest


def _login(client, mock_email, email="allowed@example.com"):
    """Helper: full login flow, returns session cookie value."""
    client.post("/api/auth/request", json={"email": email})
    token = mock_email[-1]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    return resp.cookies["session_id"]


def test_user_sees_only_own_tasks(client, mock_email, monkeypatch):
    """Two users cannot see each other's tasks."""
    from app.config import settings
    monkeypatch.setattr(settings, "allowed_emails", "alice@example.com,bob@example.com")

    alice_session = _login(client, mock_email, "alice@example.com")
    bob_session = _login(client, mock_email, "bob@example.com")

    # Alice creates a task (we mock the background processor so it doesn't run)
    import app.services.processor as proc
    monkeypatch.setattr(proc, "run_pipeline", lambda **kw: None)

    resp = client.post(
        "/api/tasks",
        json={"url": "https://example.com/recording", "mode": "gemini_direct", "language": "he"},
        cookies={"session_id": alice_session},
    )
    assert resp.status_code == 202

    # Bob lists tasks — should see nothing
    resp = client.get("/api/tasks", cookies={"session_id": bob_session})
    assert resp.status_code == 200
    assert resp.json() == []

    # Alice lists tasks — should see her task
    resp = client.get("/api/tasks", cookies={"session_id": alice_session})
    assert resp.status_code == 200
    assert len(resp.json()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_routes_protected.py::test_user_sees_only_own_tasks -v
```

Expected: FAIL (auth not wired up, tasks not filtered)

- [ ] **Step 3: Update create_task in app/state.py**

Change the existing `create_task` signature and INSERT:

```python
async def create_task(task_id: str, url: str, user_id: Optional[str] = None) -> TaskResponse:
    now = datetime.now(timezone.utc).isoformat()
    db = await _get_db()
    await db.execute(
        "INSERT INTO tasks (id, status, progress, message, created_at, url, user_id) VALUES (?,?,?,?,?,?,?)",
        [task_id, TaskStatus.PENDING.value, 0, "Task queued", now, url, user_id],
    )
    await db.commit()
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0,
        message="Task queued",
        created_at=now,
        url=url,
    )
```

Change `list_tasks` to support user filtering:

```python
async def list_tasks(limit: int = 50, user_id: Optional[str] = None) -> list[dict]:
    db = await _get_db()
    db.row_factory = aiosqlite.Row
    if user_id:
        async with db.execute(
            "SELECT id, status, progress, message, created_at, url FROM tasks "
            "WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            [user_id, limit],
        ) as cursor:
            rows = await cursor.fetchall()
    else:
        async with db.execute(
            "SELECT id, status, progress, message, created_at, url FROM tasks "
            "ORDER BY created_at DESC LIMIT ?",
            [limit],
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]
```

- [ ] **Step 4: Commit**

```bash
git add app/state.py
git commit -m "feat: add user_id to create_task and filter list_tasks by user"
```

---

## Task 7: Protect API routes

**Files:**
- Modify: `app/api/routes.py`

- [ ] **Step 1: Update routes.py**

Replace the top imports block with:

```python
import uuid
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app import state
from app.api.deps import get_current_user
from app.config import settings
from app.models import ProcessingMode, TaskCreate, TaskResponse
from app.services import processor, summarizer
```

Update `create_task`:

```python
@router.post("/tasks", response_model=TaskResponse, status_code=202)
async def create_task(
    task_in: TaskCreate,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    task_id = str(uuid.uuid4())
    task = await state.create_task(task_id, task_in.url, user_id=user_id)
    background_tasks.add_task(
        processor.run_pipeline,
        task_id=task_id,
        url=task_in.url,
        mode=task_in.mode,
        cookies=task_in.cookies,
        language=task_in.language,
    )
    return task
```

Update `create_task_from_upload`:

```python
@router.post("/tasks/upload", response_model=TaskResponse, status_code=202)
async def create_task_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: ProcessingMode = Form(ProcessingMode.GEMINI_DIRECT),
    language: str = Form("he"),
    user_id: str = Depends(get_current_user),
):
```

Inside `create_task_from_upload`, change the `create_task` call:

```python
    task = await state.create_task(task_id, f"upload:{safe_name}", user_id=user_id)
```

Update `list_tasks`:

```python
@router.get("/tasks", response_model=list)
async def list_tasks(limit: int = 20, user_id: str = Depends(get_current_user)):
    """Return the most recent N processing jobs for the current user."""
    return await state.list_tasks(limit=limit, user_id=user_id)
```

Update `get_task`, `delete_task`, and `ask_question` to add the dependency:

```python
@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, user_id: str = Depends(get_current_user)):
    task = await state.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str, user_id: str = Depends(get_current_user)):
    task = await state.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    await state.delete_task(task_id)


@router.post("/tasks/{task_id}/ask")
async def ask_question(task_id: str, body: AskRequest, user_id: str = Depends(get_current_user)):
```

- [ ] **Step 2: Run all auth tests**

```bash
python -m pytest tests/ -v
```

Expected: most tests still FAIL — auth router not registered in main.py yet.

- [ ] **Step 3: Commit**

```bash
git add app/api/routes.py
git commit -m "feat: protect all task endpoints with get_current_user dependency"
```

---

## Task 8: Update main.py

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Update app/main.py**

Replace the full file with:

```python
"""
FastAPI application entry point.

Lifespan handles:
  1. Database initialization (creates tables, marks crashed tasks as failed)
  2. GCP credentials setup
  3. Background idle-watcher (unloads Whisper from RAM when not in use)
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import state
from app.api.routes import router
from app.api.auth import router as auth_router
from app.config import settings
from app.services import transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Background tasks ──────────────────────────────────────────────────────────────

async def _idle_watcher():
    while True:
        await asyncio.sleep(60)
        try:
            await transcriber.unload_model_if_idle()
        except Exception as e:
            logger.warning(f"Idle watcher error (non-fatal): {e}")


# ── Application lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"  {settings.app_title} — starting up")
    logger.info("=" * 60)

    await state.init_db()

    creds_path = settings.google_application_credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and Path(creds_path).exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        logger.info(f"GCP credentials loaded from: {creds_path}")
    elif settings.google_api_key:
        logger.info("Using Gemini API key (AI Studio)")
    else:
        logger.warning(
            "No Google credentials found! "
            "Set GOOGLE_API_KEY in .env or ensure key.json is present."
        )

    watcher = asyncio.create_task(_idle_watcher())
    logger.info(
        f"Idle watcher started (unloads Whisper after "
        f"{settings.auto_shutdown_idle_minutes} idle minutes)"
    )
    logger.info("✅ Server ready — listening on port 8000")
    logger.info(f"   API docs: {settings.base_url}/docs")

    yield

    watcher.cancel()
    await state.close_db()
    logger.info("Server shutting down — goodbye")


# ── App factory ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version="2.0.0",
    description="Transcribe and summarize Zoom class recordings with AI",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Routes ────────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api", tags=["tasks"])
app.include_router(auth_router, prefix="/api", tags=["auth"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    """Serve the login page."""
    login_path = Path("static/login.html")
    if login_path.exists():
        return FileResponse(login_path)
    return HTMLResponse("<h1>Login</h1><p>static/login.html not found</p>")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """Serve the frontend. Redirect to /login if not authenticated."""
    from fastapi import Cookie
    session_id = request.cookies.get("session_id")
    user_id = await state.get_session_user(session_id) if session_id else None
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Zoom Transcriber</h1><p>static/index.html not found</p>")
```

- [ ] **Step 2: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: register auth router, tighten CORS, protect / route"
```

---

## Task 9: Login page + logout button

**Files:**
- Create: `static/login.html`
- Modify: `static/index.html`

- [ ] **Step 1: Create static/login.html**

```html
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>כניסה — Zoom to Text</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
      font-family: 'Segoe UI', system-ui, sans-serif;
      color: #e0e0e0;
    }

    .card {
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 20px;
      padding: 48px 40px;
      width: 100%;
      max-width: 400px;
      text-align: center;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    h1 {
      font-size: 1.8rem;
      margin-bottom: 8px;
      background: linear-gradient(135deg, #6c63ff, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .subtitle {
      color: rgba(255,255,255,0.5);
      font-size: 0.9rem;
      margin-bottom: 32px;
    }

    input[type="email"] {
      width: 100%;
      padding: 14px 16px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 12px;
      color: #fff;
      font-size: 1rem;
      text-align: right;
      margin-bottom: 16px;
      outline: none;
      transition: border-color 0.2s;
    }

    input[type="email"]:focus {
      border-color: #6c63ff;
    }

    input[type="email"]::placeholder { color: rgba(255,255,255,0.3); }

    button {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #6c63ff, #a78bfa);
      border: none;
      border-radius: 12px;
      color: #fff;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }

    button:hover { opacity: 0.9; }
    button:active { transform: scale(0.98); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    .message {
      margin-top: 20px;
      padding: 12px 16px;
      border-radius: 10px;
      font-size: 0.9rem;
      display: none;
    }

    .message.success {
      background: rgba(74,222,128,0.15);
      border: 1px solid rgba(74,222,128,0.3);
      color: #4ade80;
      display: block;
    }

    .message.error {
      background: rgba(248,113,113,0.15);
      border: 1px solid rgba(248,113,113,0.3);
      color: #f87171;
      display: block;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎙️ Zoom to Text</h1>
    <p class="subtitle">תמלול וסיכום הרצאות בינה מלאכותית</p>

    <form id="loginForm">
      <input
        type="email"
        id="emailInput"
        placeholder="כתובת המייל שלך"
        required
        autocomplete="email"
      />
      <button type="submit" id="submitBtn">שלח קישור כניסה</button>
    </form>

    <div id="message" class="message"></div>
  </div>

  <script>
    const form = document.getElementById('loginForm');
    const btn = document.getElementById('submitBtn');
    const msg = document.getElementById('message');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const email = document.getElementById('emailInput').value.trim();
      if (!email) return;

      btn.disabled = true;
      btn.textContent = 'שולח...';
      msg.className = 'message';
      msg.textContent = '';

      try {
        const resp = await fetch('/api/auth/request', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email }),
        });
        const data = await resp.json();
        msg.className = 'message success';
        msg.textContent = data.message;
        form.reset();
      } catch {
        msg.className = 'message error';
        msg.textContent = 'אירעה שגיאה. נסה שוב.';
      } finally {
        btn.disabled = false;
        btn.textContent = 'שלח קישור כניסה';
      }
    });
  </script>
</body>
</html>
```

- [ ] **Step 2: Add logout button to static/index.html**

Find the opening `<body>` tag in `static/index.html` and add immediately after it:

```html
  <div style="position:fixed;top:16px;left:16px;z-index:1000">
    <form method="POST" action="/api/auth/logout">
      <button type="submit" style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);color:rgba(255,255,255,0.6);padding:8px 16px;border-radius:8px;cursor:pointer;font-size:0.85rem">
        יציאה
      </button>
    </form>
  </div>
```

- [ ] **Step 3: Manual verification**

```bash
docker compose up -d
```

Open http://localhost:8000 — should redirect to /login.
Fill in a non-existent email — should show the generic message (no error exposed).

- [ ] **Step 4: Commit**

```bash
git add static/login.html static/index.html
git commit -m "feat: add login page and logout button"
```

---

## Task 10: key.json cleanup + fly.toml + deployment

**Files:**
- Modify: `.gitignore` (verify)
- Create: `fly.toml`

- [ ] **Step 1: Verify key.json is not tracked by git**

```bash
cd "C:\Users\אסף\zoom to text"
git ls-files key.json
```

Expected: empty output (file is ignored, not tracked).

If output shows `key.json`, remove it from tracking:

```bash
git rm --cached key.json
git commit -m "chore: stop tracking key.json (already in .gitignore)"
```

- [ ] **Step 2: Create fly.toml**

```toml
# Fly.io deployment configuration for Zoom to Text
app = "zoom-to-text"
primary_region = "ams"

[build]
  # Uses the existing Dockerfile

[[vm]]
  memory = "512mb"
  cpu_kind = "shared"
  cpus = 1

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "off"
  auto_start_machines = true
  min_machines_running = 1

  [http_service.concurrency]
    type = "requests"
    hard_limit = 10
    soft_limit = 8

  [[http_service.checks]]
    grace_period = "30s"
    interval = "15s"
    method = "GET"
    path = "/health"
    timeout = "5s"

[[mounts]]
  source = "zoom_data"
  destination = "/data"
```

- [ ] **Step 3: Set DATA_DIR in .env for production compatibility**

The Dockerfile / Fly volume mounts at `/data`. Add to `.env` (do NOT commit this):

```env
DATA_DIR=/data
DOWNLOADS_DIR=/data/downloads
```

Both `DATA_DIR=/data` and `DOWNLOADS_DIR=/data/downloads` are set as Fly secrets — no code changes needed.

- [ ] **Step 4: Fly.io deployment steps**

Run these commands in order:

```bash
# 1. Install flyctl if not installed
# https://fly.io/docs/flyctl/install/

# 2. Login
fly auth login

# 3. Create the app (first time only)
fly launch --no-deploy --name zoom-to-text --region ams

# 4. Create persistent volume (first time only)
fly volumes create zoom_data --region ams --size 10

# 5. Set all secrets (replace values with your real ones)
fly secrets set \
  GOOGLE_API_KEY="your_gemini_api_key" \
  RESEND_API_KEY="re_your_resend_key" \
  ALLOWED_EMAILS="your@email.com,student1@email.com" \
  CORS_ORIGIN="https://zoom-to-text.fly.dev" \
  BASE_URL="https://zoom-to-text.fly.dev" \
  DATA_DIR="/data" \
  DOWNLOADS_DIR="/data/downloads"

# 6. Deploy
fly deploy

# 7. Check logs
fly logs
```

- [ ] **Step 5: Add fly.toml and resend note to repo**

Before committing, create a `docs/resend-setup.md` with setup instructions (not code — just one sentence per step):

```markdown
# Resend Setup

1. Register at https://resend.com (free tier: 3,000 emails/month)
2. Create an API key at https://resend.com/api-keys
3. For production: verify your domain at https://resend.com/domains
   - Until domain is verified, you can only send to your own verified email
   - For testing: use the Resend dashboard to send to your address
4. Set RESEND_API_KEY in Fly secrets (see fly.toml Task 10 Step 4)
```

- [ ] **Step 6: Commit**

```bash
git add fly.toml docs/resend-setup.md
git commit -m "feat: add Fly.io config and Resend setup docs"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS, no warnings about missing fixtures.

- [ ] **Run the app locally end-to-end**

```bash
docker compose up -d
```

1. Open http://localhost:8000 → redirects to /login ✅
2. Enter an email NOT in `ALLOWED_EMAILS` → generic "check your email" message, nothing sent ✅
3. Add your email to `ALLOWED_EMAILS` in `.env`, restart
4. Enter your email → check server logs for the magic link URL (in dev, logs print the token) ✅
5. Open the link → redirected to `/`, app loads ✅
6. Click "יציאה" → redirected to `/login` ✅
7. Check http://localhost:8000/api/tasks without cookie → 401 ✅

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: Phase 1 complete — magic link auth + user isolation + Fly.io deploy"
```

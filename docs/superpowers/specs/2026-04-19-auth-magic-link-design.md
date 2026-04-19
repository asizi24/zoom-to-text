# Phase 1: Internet-Ready — Magic Link Auth + Fly.io Deploy

**Date:** 2026-04-19  
**Scope:** Authentication, user isolation, deployment  
**Target:** Small closed group (owner + ~10 students)  
**Deployment:** Fly.io (persistent machine, Docker)

---

## Goals

1. Any unauthenticated request redirects to `/login`
2. Authenticated users see only their own tasks
3. Access controlled by an email whitelist in `.env`
4. Deployed to Fly.io with a public HTTPS URL
5. `key.json` removed from git, stored as Fly secret

## Non-Goals (deferred to Phase 2)

- ARQ/Celery background workers (BackgroundTasks sufficient on Fly.io persistent machine)
- Google Cloud Storage (Fly volumes sufficient for few users)
- Rate limiting
- Frontend redesign

---

## Database Changes

Three new tables added to the existing SQLite schema in `app/state.py`:

```sql
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    email      TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS magic_tokens (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    expires_at TEXT NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
```

Existing `tasks` table gets one new nullable column:

```sql
ALTER TABLE tasks ADD COLUMN user_id TEXT;
```

Nullable to preserve existing local task history.  
All new tasks will have `user_id` set at creation time.

---

## Auth Flow

### Login

1. User visits any page → middleware checks session cookie
2. No valid session → redirect to `/login`
3. User enters email → `POST /api/auth/request`
4. Server checks email is in `ALLOWED_EMAILS` (comma-separated env var)
5. If allowed: create `magic_tokens` row with `expires_at = now + 15 min`
6. Send email via Resend API with link: `https://<domain>/api/auth/verify?token=<uuid>`
7. User clicks link → `GET /api/auth/verify?token=xxx`
8. Server: validates token exists, not expired, not used
9. Mark token `used=1`, create `sessions` row (`expires_at = now + 30 days`)
10. Set `session_id` cookie: `HttpOnly; Secure; SameSite=Lax; Max-Age=2592000`
11. Redirect to `/`

### Session Check (every API request)

- `get_current_user()` FastAPI dependency reads `session_id` cookie
- Looks up session in DB, checks `expires_at > now`
- Returns `user_id` string or raises `HTTPException(401)`
- All `/api/tasks/*` endpoints get `Depends(get_current_user)`

### Logout

- `POST /api/auth/logout` → deletes session from DB → clears cookie → redirect `/login`

---

## Code Changes

| File | Change |
|------|--------|
| `app/state.py` | Add `users`, `magic_tokens`, `sessions` table creation. Add `user_id` column migration. Add CRUD functions for all three tables. |
| `app/api/auth.py` | New router: `POST /api/auth/request`, `GET /api/auth/verify`, `POST /api/auth/logout` |
| `app/api/deps.py` | New file: `get_current_user()` dependency that reads session cookie |
| `app/api/routes.py` | Add `Depends(get_current_user)` to all endpoints. Filter `list_tasks` by `user_id`. Pass `user_id` to `create_task`. |
| `app/main.py` | Register auth router. Tighten CORS `allow_origins` to `[settings.cors_origin]`. Add session middleware. |
| `app/config.py` | Add: `allowed_emails`, `resend_api_key`, `cors_origin` |
| `static/login.html` | New standalone page: email input + submit button + status message |
| `fly.toml` | Fly.io config: 1 shared CPU, 512MB RAM, persistent volume at `/data`, health check on `/health` |
| `.gitignore` | Ensure `key.json` is listed |

---

## Configuration (.env additions)

```env
# Auth
ALLOWED_EMAILS=you@example.com,student1@example.com,student2@example.com
RESEND_API_KEY=re_xxxxxxxxxxxxxxxx
CORS_ORIGIN=https://zoom-to-text.fly.dev

# Deployment
BASE_URL=https://zoom-to-text.fly.dev
```

---

## Fly.io Deployment

- Single machine: `shared-cpu-1x`, 512MB RAM (sufficient for Gemini Direct mode)
- Persistent volume: 10GB mounted at `/data` (SQLite + downloaded audio files)
- Secrets: `GOOGLE_API_KEY`, `RESEND_API_KEY`, `SESSION_SECRET`, `ALLOWED_EMAILS` set via `fly secrets set`
- `key.json` removed from repo; GCP auth switches to `GOOGLE_API_KEY` (AI Studio)
- Auto-HTTPS via Fly.io (no cert management needed)

---

## Security Decisions

- **Tokens:** `uuid4()` — 122 bits of entropy, sufficient for internal use
- **Cookie flags:** `HttpOnly` (no JS access), `Secure` (HTTPS only), `SameSite=Lax` (CSRF protection)
- **Token expiry:** 15 minutes — short enough to be safe, long enough to click
- **Session expiry:** 30 days — students don't re-login constantly
- **No session secret needed** — sessions are DB-backed (random UUID in SQLite), not signed JWTs
- **No CSRF tokens needed** — `SameSite=Lax` covers the attack surface for this use case
- **key.json:** Removed from git history via `git filter-repo` or `BFG`

---

## What Does NOT Change

- Processing pipeline (`processor.py`, `transcriber.py`, `summarizer.py`, `zoom_downloader.py`)
- Frontend (`static/index.html`) — only a redirect guard added
- Docker setup — same `Dockerfile`, Fly uses it directly
- SQLite — stays as the DB (fine for few concurrent users on a persistent machine)

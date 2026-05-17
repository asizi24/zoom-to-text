"""
Tests for GET /api/tasks search + pagination (Task 4 — History tab UX).

DB-level tests verify state.list_tasks() directly.
Route-level tests hit the HTTP endpoint through the TestClient.
"""
import pytest


# ── DB-level tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_filters_by_url(client):
    """Only tasks whose URL contains the search term are returned."""
    from app import state
    await state.create_task("hs1", "https://zoom.us/rec/abc123")
    await state.create_task("hs2", "https://youtube.com/watch?v=xyz999")

    results = await state.list_tasks(search="abc123")
    ids = {r["id"] for r in results}
    assert "hs1" in ids
    assert "hs2" not in ids


@pytest.mark.asyncio
async def test_search_is_case_insensitive(client):
    """SQLite LIKE is case-insensitive for ASCII characters."""
    from app import state
    await state.create_task("hs3", "https://zoom.us/rec/UPPERCASE_TITLE")

    results = await state.list_tasks(search="uppercase_title")
    assert any(r["id"] == "hs3" for r in results)


@pytest.mark.asyncio
async def test_search_no_match_returns_empty_list(client):
    """A search term with no matching tasks returns []."""
    from app import state
    await state.create_task("hs4", "https://zoom.us/rec/unrelated")

    results = await state.list_tasks(search="NOMATCH_UNIQUE_XYZ_7a3k")
    assert results == []


@pytest.mark.asyncio
async def test_offset_paginates_without_overlap(client):
    """Page 1 and page 2 results are disjoint (no task appears twice)."""
    from app import state
    for i in range(5):
        await state.create_task(f"pg{i}", f"https://example.com/lecture/{i}")

    page1 = await state.list_tasks(limit=3, offset=0)
    page2 = await state.list_tasks(limit=3, offset=3)

    assert len(page1) == 3
    assert len(page2) == 2   # 5 total, first 3 in page1
    assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})


@pytest.mark.asyncio
async def test_limit_is_respected_with_search(client):
    """limit= caps results even when search matches more tasks than the limit."""
    from app import state
    for i in range(5):
        await state.create_task(f"lm{i}", f"https://zoom.us/lectures/class{i}")

    results = await state.list_tasks(search="lectures", limit=2)
    assert len(results) == 2


# ── Date-filter DB tests ──────────────────────────────────────────────────────

async def _backdate(task_id: str, iso: str) -> None:
    """Force a task's created_at to a known ISO timestamp."""
    from app import state
    db = await state._get_db()
    await db.execute("UPDATE tasks SET created_at=? WHERE id=?", [iso, task_id])
    await db.commit()


@pytest.mark.asyncio
async def test_since_filter_excludes_older_tasks(client):
    """since= returns only tasks whose created_at >= the given ISO bound."""
    from app import state
    await state.create_task("dt_old", "https://example.com/old")
    await state.create_task("dt_new", "https://example.com/new")
    await _backdate("dt_old", "2025-01-01T00:00:00+00:00")
    await _backdate("dt_new", "2026-05-01T00:00:00+00:00")

    results = await state.list_tasks(since="2026-01-01T00:00:00+00:00")
    ids = {r["id"] for r in results}
    assert "dt_new" in ids
    assert "dt_old" not in ids


@pytest.mark.asyncio
async def test_until_filter_excludes_newer_tasks(client):
    """until= returns only tasks whose created_at <= the given ISO bound."""
    from app import state
    await state.create_task("dt_old2", "https://example.com/old2")
    await state.create_task("dt_new2", "https://example.com/new2")
    await _backdate("dt_old2", "2025-01-01T00:00:00+00:00")
    await _backdate("dt_new2", "2026-05-01T00:00:00+00:00")

    results = await state.list_tasks(until="2026-01-01T00:00:00+00:00")
    ids = {r["id"] for r in results}
    assert "dt_old2" in ids
    assert "dt_new2" not in ids


@pytest.mark.asyncio
async def test_since_and_until_combine(client):
    """since + until bracket a window inclusively."""
    from app import state
    await state.create_task("dt_a", "https://example.com/a")
    await state.create_task("dt_b", "https://example.com/b")
    await state.create_task("dt_c", "https://example.com/c")
    await _backdate("dt_a", "2026-01-01T00:00:00+00:00")
    await _backdate("dt_b", "2026-03-01T00:00:00+00:00")
    await _backdate("dt_c", "2026-06-01T00:00:00+00:00")

    results = await state.list_tasks(
        since="2026-02-01T00:00:00+00:00",
        until="2026-04-01T00:00:00+00:00",
    )
    ids = {r["id"] for r in results}
    assert ids == {"dt_b"}


@pytest.mark.asyncio
async def test_date_filter_combines_with_search(client):
    """search and since= apply together (AND, not OR)."""
    from app import state
    await state.create_task("dts1", "https://zoom.us/window/match")
    await state.create_task("dts2", "https://zoom.us/window/match")
    await state.create_task("dts3", "https://zoom.us/other/excluded")
    await _backdate("dts1", "2025-12-01T00:00:00+00:00")
    await _backdate("dts2", "2026-05-01T00:00:00+00:00")
    await _backdate("dts3", "2026-05-01T00:00:00+00:00")

    results = await state.list_tasks(
        search="window/match",
        since="2026-01-01T00:00:00+00:00",
    )
    ids = {r["id"] for r in results}
    assert ids == {"dts2"}


# ── Route-level tests ─────────────────────────────────────────────────────────

def _login(client, monkeypatch) -> str:
    """Authenticate via magic link and return session_id."""
    import app.api.auth as auth_module
    captured: list[str] = []

    async def fake_send(email: str, token: str) -> None:
        captured.append(token)

    monkeypatch.setattr(auth_module, "_send_magic_link_email", fake_send)
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = captured[0]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    return resp.cookies["session_id"]


def test_route_accepts_search_and_offset_params(client, monkeypatch):
    """GET /api/tasks?search=&offset= must return 200, not 422."""
    sid = _login(client, monkeypatch)
    r = client.get(
        "/api/tasks?search=hello&offset=0&limit=5",
        cookies={"session_id": sid},
    )
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_route_accepts_since_and_until_params(client, monkeypatch):
    """GET /api/tasks?since=&until= must return 200, not 422."""
    sid = _login(client, monkeypatch)
    r = client.get(
        "/api/tasks?since=2026-01-01T00:00:00&until=2026-12-31T23:59:59",
        cookies={"session_id": sid},
    )
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_route_rejects_invalid_since(client, monkeypatch):
    """Garbage ISO bounds yield 422, not a 500 from SQLite."""
    sid = _login(client, monkeypatch)
    r = client.get(
        "/api/tasks?since=not-a-date",
        cookies={"session_id": sid},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_route_search_returns_only_matching_task(client, monkeypatch):
    """End-to-end: task seeded in DB is found by search, non-matching task is excluded."""
    from app import state
    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    await state.create_task("rt1", "https://zoom.us/unique_search_term_7z9", user_id=user_id)
    await state.create_task("rt2", "https://zoom.us/other_recording",        user_id=user_id)

    r = client.get(
        "/api/tasks?search=unique_search_term_7z9",
        cookies={"session_id": sid},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["id"] == "rt1"

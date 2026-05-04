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

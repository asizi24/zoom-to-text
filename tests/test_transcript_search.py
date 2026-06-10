"""
Tests for GET /api/tasks/{id}/search — per-task content search.

Verifies that searching within a completed task's content (summary, chapters,
transcript) returns matching excerpts with context and position metadata.
"""
import pytest

from app.models import Chapter, LessonResult, QuizQuestion


def _login(client, monkeypatch) -> str:
    import app.api.auth as auth_module

    captured: list[str] = []

    async def fake_send(email: str, token: str) -> None:
        captured.append(token)

    monkeypatch.setattr(auth_module, "_send_magic_link_email", fake_send)
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = captured[0]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    return resp.cookies["session_id"]


def _make_result() -> LessonResult:
    return LessonResult(
        summary="This lecture covers binary search trees and their traversal algorithms.",
        chapters=[
            Chapter(
                title="Introduction to BST",
                content="A binary search tree is a data structure where each node has at most two children.",
                key_points=["O(log n) average case", "in-order traversal gives sorted output"],
                start_time="00:05",
            ),
            Chapter(
                title="Graph Algorithms",
                content="Dijkstra's algorithm finds shortest paths in weighted graphs.",
                key_points=["greedy approach", "priority queue"],
                start_time="00:35",
            ),
        ],
        transcript="Speaker A: Today we will study binary search trees. Speaker B: Can you explain traversal? Speaker A: In-order traversal visits left subtree first.",
        quiz=[
            QuizQuestion(
                question="What is the average time complexity of BST search?",
                options=["O(1)", "O(log n)", "O(n)", "O(n log n)"],
                correct_answer="O(log n)",
                explanation="BST search is O(log n) on average for balanced trees.",
            )
        ],
        language="en",
    )


# ── state-level tests ────────────────────────────────────────────────────────

async def test_search_finds_term_in_summary(client):
    from app import state

    result = _make_result()
    await state.create_task("ts1", "http://test/rec")
    await state.complete_task("ts1", result)

    hits = await state.search_task_content("ts1", None, "binary search trees")
    assert any(h["type"] == "summary" for h in hits)


async def test_search_finds_term_in_chapter(client):
    from app import state

    result = _make_result()
    await state.create_task("ts2", "http://test/rec")
    await state.complete_task("ts2", result)

    hits = await state.search_task_content("ts2", None, "Dijkstra")
    chapter_hits = [h for h in hits if h["type"] == "chapter"]
    assert len(chapter_hits) >= 1
    assert chapter_hits[0]["chapter_title"] == "Graph Algorithms"
    assert chapter_hits[0]["chapter_start_time"] == "00:35"


async def test_search_finds_term_in_transcript(client):
    from app import state

    result = _make_result()
    await state.create_task("ts3", "http://test/rec")
    await state.complete_task("ts3", result)

    hits = await state.search_task_content("ts3", None, "traversal")
    transcript_hits = [h for h in hits if h["type"] == "transcript"]
    assert len(transcript_hits) >= 1


async def test_search_returns_empty_for_no_match(client):
    from app import state

    result = _make_result()
    await state.create_task("ts4", "http://test/rec")
    await state.complete_task("ts4", result)

    hits = await state.search_task_content("ts4", None, "NOMATCH_ZZZ_9999")
    assert hits == []


async def test_search_returns_none_for_unknown_task(client):
    from app import state

    hits = await state.search_task_content("unknown-task-id", None, "anything")
    assert hits is None


async def test_search_returns_context_with_surrounding_text(client):
    from app import state

    result = _make_result()
    await state.create_task("ts5", "http://test/rec")
    await state.complete_task("ts5", result)

    hits = await state.search_task_content("ts5", None, "binary search tree")
    assert len(hits) > 0
    assert all("context" in h for h in hits)
    assert all("binary search tree" in h["context"].lower() for h in hits)


async def test_search_is_case_insensitive(client):
    from app import state

    result = _make_result()
    await state.create_task("ts6", "http://test/rec")
    await state.complete_task("ts6", result)

    hits_lower = await state.search_task_content("ts6", None, "dijkstra")
    hits_upper = await state.search_task_content("ts6", None, "DIJKSTRA")
    assert len(hits_lower) == len(hits_upper)


async def test_search_result_has_required_fields(client):
    from app import state

    result = _make_result()
    await state.create_task("ts7", "http://test/rec")
    await state.complete_task("ts7", result)

    hits = await state.search_task_content("ts7", None, "binary")
    for h in hits:
        assert "type" in h
        assert "context" in h
        assert h["type"] in ("summary", "chapter", "transcript")


# ── route-level tests ────────────────────────────────────────────────────────

async def test_route_returns_200_with_results(client, monkeypatch):
    from app import state

    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    result = _make_result()
    await state.create_task("tr1", "http://test/rec", user_id=user_id)
    await state.complete_task("tr1", result)

    r = client.get("/api/tasks/tr1/search?q=binary", cookies={"session_id": sid})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "query" in data
    assert data["query"] == "binary"
    assert len(data["results"]) > 0


def test_route_returns_404_for_unknown_task(client, monkeypatch):
    sid = _login(client, monkeypatch)
    r = client.get("/api/tasks/no-such-task/search?q=hello", cookies={"session_id": sid})
    assert r.status_code == 404


def test_route_returns_422_for_missing_query(client, monkeypatch):
    sid = _login(client, monkeypatch)
    r = client.get("/api/tasks/any/search", cookies={"session_id": sid})
    assert r.status_code == 422


async def test_route_returns_400_for_incomplete_task(client, monkeypatch):
    from app import state

    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    await state.create_task("tr2", "http://test/rec", user_id=user_id)
    # No complete_task call — task has no result

    r = client.get("/api/tasks/tr2/search?q=hello", cookies={"session_id": sid})
    assert r.status_code == 400


async def test_route_returns_401_without_auth(client):
    r = client.get("/api/tasks/any/search?q=hello")
    assert r.status_code in (401, 403)

"""
Tests for POST /api/study-guide — Multi-Lecture Cram Mode.

Verifies that combining 2-10 completed task results produces a synthesized
study guide and comprehensive exam via the summarizer.generate_cram_guide hook.
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


def _make_result(topic: str) -> LessonResult:
    return LessonResult(
        summary=f"This lecture covers {topic}.",
        chapters=[
            Chapter(
                title=f"{topic} — Introduction",
                content=f"Key concepts of {topic} including definitions and examples.",
                key_points=[f"{topic} basics", "applications"],
            ),
        ],
        quiz=[
            QuizQuestion(
                question=f"What is {topic}?",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                explanation=f"{topic} is explained in this lecture.",
            )
        ],
        language="en",
    )


def _fake_cram_result():
    from app.models import CramChapter, CramGuideResult

    return CramGuideResult(
        overall_summary="Combined summary of two lectures.",
        key_themes=["trees", "graphs"],
        chapters=[
            CramChapter(title="BST", summary="Binary trees", key_concepts=["O(log n)"]),
            CramChapter(title="Graphs", summary="Graph search", key_concepts=["Dijkstra"]),
        ],
        quiz=[
            QuizQuestion(
                question="What is BST search complexity?",
                options=["O(1)", "O(log n)", "O(n)", "O(n^2)"],
                correct_answer="O(log n)",
                explanation="Average case for balanced BST.",
            )
        ],
        lecture_count=2,
    )


# ── model / schema tests ──────────────────────────────────────────────────────

def test_cram_guide_request_requires_at_least_2_tasks():
    from pydantic import ValidationError

    from app.models import CramGuideRequest

    with pytest.raises(ValidationError):
        CramGuideRequest(task_ids=["only-one"])


def test_cram_guide_request_rejects_more_than_10_tasks():
    from pydantic import ValidationError

    from app.models import CramGuideRequest

    with pytest.raises(ValidationError):
        CramGuideRequest(task_ids=[f"t{i}" for i in range(11)])


def test_cram_guide_request_accepts_2_to_10_tasks():
    from app.models import CramGuideRequest

    req = CramGuideRequest(task_ids=["a", "b", "c"])
    assert len(req.task_ids) == 3


def test_cram_guide_result_has_safe_defaults():
    from app.models import CramGuideResult

    r = CramGuideResult()
    assert r.overall_summary == ""
    assert r.quiz == []
    assert r.key_themes == []
    assert r.chapters == []
    assert r.lecture_count == 0


def test_cram_chapter_model():
    from app.models import CramChapter

    ch = CramChapter(title="Test", summary="A summary.")
    assert ch.key_concepts == []


# ── route-level tests ─────────────────────────────────────────────────────────

async def test_route_requires_authentication(client):
    r = client.post("/api/study-guide", json={"task_ids": ["a", "b"]})
    assert r.status_code in (401, 403)


def test_route_rejects_single_task_id(client, monkeypatch):
    sid = _login(client, monkeypatch)
    r = client.post(
        "/api/study-guide",
        json={"task_ids": ["only-one"]},
        cookies={"session_id": sid},
    )
    assert r.status_code == 422


def test_route_rejects_eleven_task_ids(client, monkeypatch):
    sid = _login(client, monkeypatch)
    r = client.post(
        "/api/study-guide",
        json={"task_ids": [f"t{i}" for i in range(11)]},
        cookies={"session_id": sid},
    )
    assert r.status_code == 422


async def test_route_returns_404_when_task_not_owned(client, monkeypatch):
    from app import state

    sid = _login(client, monkeypatch)

    # Create tasks owned by a different user (no user_id → no owner match)
    await state.create_task("cg_other1", "http://test/rec1")
    await state.complete_task("cg_other1", _make_result("Topic A"))

    r = client.post(
        "/api/study-guide",
        json={"task_ids": ["cg_other1", "cg_other2"]},
        cookies={"session_id": sid},
    )
    assert r.status_code == 404


async def test_route_returns_400_when_task_incomplete(client, monkeypatch):
    from app import state

    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    await state.create_task("cg_pend1", "http://test/rec1", user_id=user_id)
    await state.create_task("cg_pend2", "http://test/rec2", user_id=user_id)
    # Neither task is completed — no result_json

    r = client.post(
        "/api/study-guide",
        json={"task_ids": ["cg_pend1", "cg_pend2"]},
        cookies={"session_id": sid},
    )
    assert r.status_code == 400


async def test_route_returns_guide_for_two_valid_tasks(client, monkeypatch):
    from app import state
    from app.services import summarizer

    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    await state.create_task("cg1", "http://test/rec1", user_id=user_id)
    await state.complete_task("cg1", _make_result("Binary Search Trees"))
    await state.create_task("cg2", "http://test/rec2", user_id=user_id)
    await state.complete_task("cg2", _make_result("Graph Algorithms"))

    async def fake_generate(lessons: list[dict]):
        return _fake_cram_result()

    monkeypatch.setattr(summarizer, "generate_cram_guide", fake_generate)

    r = client.post(
        "/api/study-guide",
        json={"task_ids": ["cg1", "cg2"]},
        cookies={"session_id": sid},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["overall_summary"] == "Combined summary of two lectures."
    assert data["lecture_count"] == 2
    assert len(data["quiz"]) == 1
    assert len(data["chapters"]) == 2
    assert "trees" in data["key_themes"]


async def test_generate_cram_guide_called_with_lesson_list(client, monkeypatch):
    """generate_cram_guide receives a list with one dict per task."""
    from app import state
    from app.services import summarizer

    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    await state.create_task("cg3", "http://test/rec3", user_id=user_id)
    await state.complete_task("cg3", _make_result("Sorting Algorithms"))
    await state.create_task("cg4", "http://test/rec4", user_id=user_id)
    await state.complete_task("cg4", _make_result("Dynamic Programming"))

    received: list = []

    async def capturing_generate(lessons: list[dict]):
        received.extend(lessons)
        return _fake_cram_result()

    monkeypatch.setattr(summarizer, "generate_cram_guide", capturing_generate)

    client.post(
        "/api/study-guide",
        json={"task_ids": ["cg3", "cg4"]},
        cookies={"session_id": sid},
    )

    assert len(received) == 2
    assert all("summary" in lesson for lesson in received)
    assert all("chapters" in lesson for lesson in received)

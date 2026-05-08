"""
Tests for PATCH /api/tasks/{task_id}/speakers and state.update_speaker_map.

The endpoint lets a user rename diarization labels ("Speaker A" → "Asaf") so
the transcript display + chat context use real names. The DB helper sanitizes
the input (trim, drop empty values, cap names at 80 chars).
"""
import pytest

from app.models import LessonResult


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


@pytest.mark.asyncio
async def test_update_speaker_map_sets_field(client):
    from app import state
    await state.create_task("sp1", "https://zoom.us/sp", user_id="u")
    await state.complete_task("sp1", LessonResult(summary="hi", diarized_transcript="Speaker A: hi"))

    ok = await state.update_speaker_map("sp1", "u", {"Speaker A": "Asaf"})
    assert ok is True
    task = await state.get_task_for_user("sp1", "u")
    assert task is not None
    assert task.result is not None
    assert task.result.speaker_map == {"Speaker A": "Asaf"}


@pytest.mark.asyncio
async def test_update_speaker_map_drops_empty_values(client):
    """Empty string / whitespace-only values delete the key (rename → undo)."""
    from app import state
    await state.create_task("sp2", "https://zoom.us/sp", user_id="u")
    await state.complete_task("sp2", LessonResult(summary="x"))

    ok = await state.update_speaker_map(
        "sp2", "u", {"Speaker A": "Asaf", "Speaker B": "  ", "Speaker C": ""}
    )
    assert ok
    task = await state.get_task_for_user("sp2", "u")
    assert task.result.speaker_map == {"Speaker A": "Asaf"}


@pytest.mark.asyncio
async def test_update_speaker_map_caps_long_names(client):
    """Speaker names are capped at 80 chars to avoid DB bloat / rendering issues."""
    from app import state
    await state.create_task("sp3", "https://zoom.us/sp", user_id="u")
    await state.complete_task("sp3", LessonResult(summary="x"))

    long_name = "X" * 200
    ok = await state.update_speaker_map("sp3", "u", {"Speaker A": long_name})
    assert ok
    task = await state.get_task_for_user("sp3", "u")
    assert len(task.result.speaker_map["Speaker A"]) == 80


@pytest.mark.asyncio
async def test_update_speaker_map_rejects_foreign_owner(client):
    from app import state
    await state.create_task("sp4", "https://zoom.us/sp", user_id="real-owner")
    await state.complete_task("sp4", LessonResult(summary="x"))

    ok = await state.update_speaker_map("sp4", "intruder", {"Speaker A": "x"})
    assert ok is False


@pytest.mark.asyncio
async def test_update_speaker_map_no_result_yet(client):
    """If task has no result_json (still pending), update returns False."""
    from app import state
    await state.create_task("sp5", "https://zoom.us/sp", user_id="u")
    ok = await state.update_speaker_map("sp5", "u", {"Speaker A": "x"})
    assert ok is False


# ── Route-level tests ─────────────────────────────────────────────────────────


def test_speakers_patch_requires_auth(client):
    r = client.patch(
        "/api/tasks/whatever/speakers",
        json={"speaker_map": {"Speaker A": "x"}},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_speakers_patch_updates_and_returns_task(client, monkeypatch):
    from app import state
    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    await state.create_task("sp-rt1", "https://zoom.us/sp", user_id=user_id)
    await state.complete_task(
        "sp-rt1",
        LessonResult(summary="hi", diarized_transcript="Speaker A: שלום"),
    )

    r = client.patch(
        "/api/tasks/sp-rt1/speakers",
        json={"speaker_map": {"Speaker A": "אסף"}},
        cookies={"session_id": sid},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task_id"] == "sp-rt1"
    assert body["result"]["speaker_map"] == {"Speaker A": "אסף"}


@pytest.mark.asyncio
async def test_speakers_patch_404_for_foreign_task(client, monkeypatch):
    from app import state
    sid = _login(client, monkeypatch)
    await state.create_task("sp-foreign", "https://zoom.us/sp", user_id="other")
    await state.complete_task("sp-foreign", LessonResult(summary="x"))

    r = client.patch(
        "/api/tasks/sp-foreign/speakers",
        json={"speaker_map": {"Speaker A": "x"}},
        cookies={"session_id": sid},
    )
    assert r.status_code == 404

"""
Phase C tests for Task 1.5 — error_details JSON persistence in SQLite.
"""
import pytest


@pytest.mark.asyncio
async def test_fail_task_persists_error_details_dict(client):
    """state.fail_task accepts a dict and stores it; get_task returns it."""
    from app import state
    from app.errors import ProcessingError, ProcessingStage

    await state.create_task("t-err-1", url="https://example.com")
    pe = ProcessingError(
        stage=ProcessingStage.SUMMARIZE,
        code="llm_rate_limit",
        user_message="⚠️ מכסה הוצתה",
        technical_details="HTTP 429",
    )
    await state.fail_task("t-err-1", pe.user_message, pe.to_dict())

    task = await state.get_task("t-err-1")
    assert task is not None
    assert task.error == "⚠️ מכסה הוצתה"
    assert task.error_details is not None
    assert task.error_details["stage"] == "summarize"
    assert task.error_details["code"] == "llm_rate_limit"
    assert task.error_details["user_message"] == "⚠️ מכסה הוצתה"
    assert task.error_details["technical_details"] == "HTTP 429"


@pytest.mark.asyncio
async def test_fail_task_without_error_details_works(client):
    """Backward-compat: fail_task with no error_details kwarg sets it None."""
    from app import state

    await state.create_task("t-err-2", url="https://example.com")
    await state.fail_task("t-err-2", "plain string error")

    task = await state.get_task("t-err-2")
    assert task is not None
    assert task.error == "plain string error"
    assert task.error_details is None


@pytest.mark.asyncio
async def test_old_rows_without_column_return_none(client):
    """Rows that pre-date the migration (column NULL) return error_details=None."""
    from app import state

    db = await state._get_db()
    # Simulate a row that was inserted before the migration ran
    await db.execute(
        "INSERT INTO tasks (id, status, progress, message, created_at, error) "
        "VALUES (?, 'failed', 0, '', '2026-01-01T00:00:00Z', 'old error')",
        ["t-err-old"],
    )
    await db.commit()

    task = await state.get_task("t-err-old")
    assert task is not None
    assert task.error == "old error"
    assert task.error_details is None


@pytest.mark.asyncio
async def test_task_response_serializes_error_details(client):
    """The Pydantic TaskResponse round-trips error_details through JSON."""
    from app import state
    from app.errors import ProcessingError, ProcessingStage

    await state.create_task("t-err-3", url="https://example.com")
    pe = ProcessingError(
        stage=ProcessingStage.DOWNLOAD,
        code="zoom_cookies_expired",
        user_message="🍪 cookies expired",
        technical_details="HTTP 401 from Zoom",
    )
    await state.fail_task("t-err-3", pe.user_message, pe.to_dict())

    task = await state.get_task("t-err-3")
    blob = task.model_dump(mode="json")
    assert blob["error_details"]["code"] == "zoom_cookies_expired"
    assert blob["error_details"]["stage"] == "download"

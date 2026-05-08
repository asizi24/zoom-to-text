"""
Tests for the per-user URL cache: when a user submits a Zoom URL twice, the
second task should be served from the previous run's result_json instead of
re-running the pipeline.

DB-level: state.find_cached_task() returns the most recent COMPLETED task
          owned by user_id with the matching URL.
Cloning:  state.copy_result_from_cached() clones the result_json onto a new
          task row so deleting the original does NOT cascade.
Pipeline: processor.run_pipeline short-circuits before the download stage when
          a cached task is found.
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.models import LessonResult, ProcessingMode


@pytest.mark.asyncio
async def test_find_cached_task_returns_completed(client):
    """The most recent COMPLETED task with same url+user is returned."""
    from app import state
    await state.create_task("c1", "https://zoom.us/rec/SAME", user_id="user-A")
    await state.complete_task("c1", LessonResult(summary="hello"))

    cached = await state.find_cached_task("https://zoom.us/rec/SAME", "user-A")
    assert cached is not None
    assert cached.task_id == "c1"
    assert cached.result is not None
    assert cached.result.summary == "hello"


@pytest.mark.asyncio
async def test_find_cached_task_scopes_per_user(client):
    """User B does NOT see user A's cached results."""
    from app import state
    await state.create_task("c2", "https://zoom.us/rec/PRIVATE", user_id="user-A")
    await state.complete_task("c2", LessonResult(summary="A's secret"))

    cached_for_b = await state.find_cached_task(
        "https://zoom.us/rec/PRIVATE", "user-B"
    )
    assert cached_for_b is None


@pytest.mark.asyncio
async def test_find_cached_task_ignores_pending(client):
    """A pending/failed task with same URL must NOT be served as cache."""
    from app import state
    await state.create_task("c3", "https://zoom.us/rec/PEND", user_id="u")
    cached = await state.find_cached_task("https://zoom.us/rec/PEND", "u")
    assert cached is None


@pytest.mark.asyncio
async def test_find_cached_task_skips_uploads(client):
    """upload:filename URLs are never cached (file content can vary)."""
    from app import state
    await state.create_task("up1", "upload:foo.mp4", user_id="u")
    await state.complete_task("up1", LessonResult(summary="x"))
    cached = await state.find_cached_task("upload:foo.mp4", "u")
    assert cached is None


@pytest.mark.asyncio
async def test_copy_result_from_cached_clones_independently(client):
    """
    After cloning, deleting the source must NOT remove the clone's result.
    This protects users who delete an old recording from losing replays.
    """
    from app import state
    src_result = LessonResult(summary="source content")
    await state.create_task("src1", "https://zoom.us/rec/X", user_id="u")
    await state.complete_task("src1", src_result)
    src_json = await state.get_result_json("src1")
    assert src_json is not None

    # Spawn the second task and copy
    await state.create_task("clone1", "https://zoom.us/rec/X", user_id="u")
    await state.copy_result_from_cached("clone1", src_json)

    # Sanity: clone is now completed with same content
    clone = await state.get_task_for_user("clone1", "u")
    assert clone is not None
    assert clone.status.value == "completed"
    assert clone.result is not None
    assert clone.result.summary == "source content"
    assert "מהמטמון" in (clone.message or "")

    # Delete the source — clone must survive intact
    await state.delete_task("src1")
    clone_after = await state.get_task_for_user("clone1", "u")
    assert clone_after is not None
    assert clone_after.result is not None
    assert clone_after.result.summary == "source content"


@pytest.mark.asyncio
async def test_pipeline_short_circuits_on_cache_hit(client, monkeypatch):
    """
    processor.run_pipeline() must bail out before the download stage when a
    cached task is available — confirming this is the entire point of the
    URL cache. We patch zoom_downloader.download_audio to detonate; if the
    cache hook works, it never runs.
    """
    from app import state
    from app.services import processor, zoom_downloader

    # Seed a completed source task for cache-user
    await state.create_task("seed1", "https://zoom.us/rec/CACHED", user_id="cache-user")
    await state.complete_task("seed1", LessonResult(summary="cached output"))

    # Spawn the second task with the same URL+user
    await state.create_task("new1", "https://zoom.us/rec/CACHED", user_id="cache-user")

    download_called = False

    async def boom(*args, **kwargs):
        nonlocal download_called
        download_called = True
        raise RuntimeError("cache hit should have prevented download")

    monkeypatch.setattr(zoom_downloader, "download_audio", boom)

    await processor.run_pipeline(
        task_id="new1",
        url="https://zoom.us/rec/CACHED",
        mode=ProcessingMode.GEMINI_DIRECT,
        cookies=None,
        language="he",
    )

    assert download_called is False, "download_audio must NOT be called on cache hit"
    final = await state.get_task_for_user("new1", "cache-user")
    assert final is not None
    assert final.status.value == "completed"
    assert final.result is not None
    assert final.result.summary == "cached output"


@pytest.mark.asyncio
async def test_pipeline_runs_downloader_when_cache_miss(client, monkeypatch):
    """Mirror test: when no cached task exists, the downloader IS invoked."""
    from app import state
    from app.services import processor, zoom_downloader

    await state.create_task("nocache1", "https://zoom.us/rec/UNIQUE", user_id="cache-user")

    download_called = False

    async def fake_download(*args, **kwargs):
        nonlocal download_called
        download_called = True
        # Raise to short-circuit the rest of the pipeline (we only care that
        # the downloader was called past the cache check).
        raise RuntimeError("test-only abort after download started")

    monkeypatch.setattr(zoom_downloader, "download_audio", fake_download)

    # We expect run_pipeline to swallow the error into ProcessingError → fail_task
    await processor.run_pipeline(
        task_id="nocache1",
        url="https://zoom.us/rec/UNIQUE",
        mode=ProcessingMode.GEMINI_DIRECT,
        cookies=None,
        language="he",
    )
    assert download_called is True

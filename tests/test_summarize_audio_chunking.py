"""
Tests for the long-audio chunked path in `summarize_audio`.

Background: Gemini's `generate_content` rejects requests where audio + prompt
together exceed 1,048,576 tokens with
"The input token count exceeds the maximum number of tokens allowed".
This used to fail tasks at progress=72%. The fix routes recordings longer than
``_GEMINI_DIRECT_MAX_SECONDS`` through `_summarize_audio_chunked`, which splits
the audio, transcribes each chunk via Gemini, then funnels the merged text
through `summarize_transcript`.

These tests stub out every Gemini SDK and ffmpeg interaction so they run
deterministically on any host.
"""
import asyncio

import pytest

from app.models import LessonResult
from app.services import summarizer


# ── Routing decision ──────────────────────────────────────────────────────────

def test_short_audio_uses_direct_path(monkeypatch):
    """Audio at or below the threshold goes through the original direct path."""
    threshold = summarizer._GEMINI_DIRECT_MAX_SECONDS

    monkeypatch.setattr(summarizer, "_audio_duration_seconds", lambda _p: threshold - 1)

    direct_called = {"flag": False}
    chunked_called = {"flag": False}

    def fake_upload(path, progress_cb=None):
        direct_called["flag"] = True
        return _FakeAudioFile()

    async def fake_chunked(path, progress_cb):
        chunked_called["flag"] = True
        return LessonResult(summary="from-chunked")

    monkeypatch.setattr(summarizer, "_upload_audio_to_gemini", fake_upload)
    monkeypatch.setattr(summarizer, "_summarize_audio_chunked", fake_chunked)
    monkeypatch.setattr(summarizer, "_is_gemini_provider", lambda: True)
    monkeypatch.setattr(
        summarizer, "_synthesize_audio_capture",
        lambda audio_file, progress_cb=None: (LessonResult(summary="direct"), "raw"),
    )
    monkeypatch.setattr(
        summarizer, "_extract_audio_capture",
        lambda audio_file: (None, None),
    )
    monkeypatch.setattr(summarizer, "_delete_gemini_file", lambda _f: None)

    result = asyncio.run(summarizer.summarize_audio("/fake/audio.m4a"))

    assert direct_called["flag"] is True
    assert chunked_called["flag"] is False
    assert result.summary == "direct"


def test_long_audio_routes_to_chunked_path(monkeypatch):
    """Audio above the threshold skips direct mode and uses the chunked path."""
    threshold = summarizer._GEMINI_DIRECT_MAX_SECONDS

    monkeypatch.setattr(
        summarizer, "_audio_duration_seconds", lambda _p: threshold + 60
    )

    direct_called = {"flag": False}

    def fake_upload(path, progress_cb=None):
        direct_called["flag"] = True
        return _FakeAudioFile()

    async def fake_chunked(path, progress_cb):
        return LessonResult(summary="from-chunked-path")

    monkeypatch.setattr(summarizer, "_upload_audio_to_gemini", fake_upload)
    monkeypatch.setattr(summarizer, "_summarize_audio_chunked", fake_chunked)
    monkeypatch.setattr(summarizer, "_is_gemini_provider", lambda: True)

    result = asyncio.run(summarizer.summarize_audio("/fake/long.m4a"))

    assert direct_called["flag"] is False
    assert result.summary == "from-chunked-path"


def test_unknown_duration_falls_back_to_direct_path(monkeypatch):
    """If ffprobe is missing/broken, we don't crash — direct path runs as before."""
    monkeypatch.setattr(summarizer, "_audio_duration_seconds", lambda _p: None)

    monkeypatch.setattr(summarizer, "_is_gemini_provider", lambda: True)
    monkeypatch.setattr(
        summarizer, "_upload_audio_to_gemini",
        lambda path, progress_cb=None: _FakeAudioFile(),
    )
    monkeypatch.setattr(
        summarizer, "_synthesize_audio_capture",
        lambda audio_file, progress_cb=None: (LessonResult(summary="direct-fallback"), "raw"),
    )
    monkeypatch.setattr(
        summarizer, "_extract_audio_capture",
        lambda audio_file: (None, None),
    )
    monkeypatch.setattr(summarizer, "_delete_gemini_file", lambda _f: None)

    result = asyncio.run(summarizer.summarize_audio("/fake/unknown.m4a"))
    assert result.summary == "direct-fallback"


# ── Chunked-path orchestration ────────────────────────────────────────────────

def test_chunked_path_concatenates_transcripts_and_summarizes(monkeypatch):
    """`_summarize_audio_chunked` merges chunk transcripts and routes to
    `summarize_transcript`, preserving the merged text on the result.

    Order of the merged transcript must match input chunk order even though
    transcription runs in parallel — the result shouldn't shuffle when
    chunks complete out-of-order.
    """
    fake_chunks = ["/tmp/chunk0.mp3", "/tmp/chunk1.mp3", "/tmp/chunk2.mp3"]
    cleanup_calls = {"chunks": None}

    monkeypatch.setattr(
        summarizer, "_hard_split_audio_for_gemini", lambda _p: list(fake_chunks)
    )

    def fake_cleanup(chunks):
        cleanup_calls["chunks"] = list(chunks)

    monkeypatch.setattr(summarizer, "_cleanup_chunk_files", fake_cleanup)

    transcribe_call_log = []

    def fake_transcribe(chunk_path):
        transcribe_call_log.append(chunk_path)
        return f"text-from-{chunk_path.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(
        summarizer, "_transcribe_audio_chunk_via_gemini", fake_transcribe
    )

    seen_transcript = {"value": None}

    async def fake_summarize_transcript(transcript, progress_cb=None):
        seen_transcript["value"] = transcript
        return LessonResult(summary="merged-summary")

    monkeypatch.setattr(
        summarizer, "summarize_transcript", fake_summarize_transcript
    )

    result = asyncio.run(
        summarizer._summarize_audio_chunked("/fake/long.m4a", progress_cb=None)
    )

    # Order in the merged transcript must match input order (chunk0 → chunk1 → chunk2)
    expected = "text-from-chunk0.mp3\n\ntext-from-chunk1.mp3\n\ntext-from-chunk2.mp3"
    assert seen_transcript["value"] == expected
    assert seen_transcript["value"] == result.transcript
    assert sorted(transcribe_call_log) == sorted(fake_chunks)  # all chunks transcribed
    assert result.summary == "merged-summary"
    # Chunks must be cleaned up regardless of outcome.
    assert cleanup_calls["chunks"] == fake_chunks


def test_chunked_path_cleans_up_when_transcription_fails(monkeypatch):
    """If a chunk transcription raises, _cleanup_chunk_files still runs."""
    fake_chunks = ["/tmp/chunk0.mp3", "/tmp/chunk1.mp3"]
    cleanup_calls = {"chunks": None}

    monkeypatch.setattr(
        summarizer, "_hard_split_audio_for_gemini", lambda _p: list(fake_chunks)
    )
    monkeypatch.setattr(
        summarizer, "_cleanup_chunk_files",
        lambda chunks: cleanup_calls.update(chunks=list(chunks)),
    )

    def boom(_chunk):
        raise RuntimeError("gemini error")

    monkeypatch.setattr(summarizer, "_transcribe_audio_chunk_via_gemini", boom)

    with pytest.raises(RuntimeError, match="gemini error"):
        asyncio.run(
            summarizer._summarize_audio_chunked("/fake/long.m4a", progress_cb=None)
        )

    assert cleanup_calls["chunks"] == fake_chunks


def test_chunked_path_rejects_empty_transcripts(monkeypatch):
    """All-empty transcripts must raise rather than silently produce a summary."""
    fake_chunks = ["/tmp/chunk0.mp3"]

    monkeypatch.setattr(
        summarizer, "_hard_split_audio_for_gemini", lambda _p: list(fake_chunks)
    )
    monkeypatch.setattr(summarizer, "_cleanup_chunk_files", lambda _c: None)
    monkeypatch.setattr(summarizer, "_transcribe_audio_chunk_via_gemini", lambda _c: "")

    with pytest.raises(RuntimeError, match="ריק"):
        asyncio.run(
            summarizer._summarize_audio_chunked("/fake/long.m4a", progress_cb=None)
        )


def test_hard_split_raises_when_duration_unavailable(monkeypatch):
    """The hard splitter must NOT silently fall back to the full file when
    ffprobe can't measure duration — that was the bug the previous attempt hit."""
    monkeypatch.setattr(summarizer, "_audio_duration_seconds", lambda _p: None)

    with pytest.raises(RuntimeError, match="ffprobe"):
        summarizer._hard_split_audio_for_gemini("/fake/audio.m4a")


# ── Token-exceeded fallback (defense in depth) ────────────────────────────────

@pytest.mark.parametrize(
    "msg",
    [
        "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'The input "
        "token count exceeds the maximum number of tokens allowed 1048576.'}}",
        "Input token count is too large: 1048576 > limit",
        "request payload exceeds the maximum number of tokens",
    ],
)
def test_token_exceeded_detector_recognizes_gemini_payload(msg):
    """The detector must catch the actual Gemini wire format, not just one
    fixed string — Gemini sometimes paraphrases the error message."""
    assert summarizer._is_token_exceeded_error(RuntimeError(msg))


def test_token_exceeded_detector_ignores_unrelated_errors():
    assert not summarizer._is_token_exceeded_error(RuntimeError("connection reset"))
    assert not summarizer._is_token_exceeded_error(ValueError("bad json"))


def test_summarize_audio_falls_back_to_chunked_on_token_exceeded(monkeypatch):
    """If the direct path's synthesis call raises a token-exceeded error
    despite a sub-threshold ffprobe reading, we must fall back to the chunked
    path rather than failing the task. Defends against high-token-rate audio
    where 45 min isn't actually safe."""
    threshold = summarizer._GEMINI_DIRECT_MAX_SECONDS

    monkeypatch.setattr(
        summarizer, "_audio_duration_seconds", lambda _p: threshold - 60
    )
    monkeypatch.setattr(summarizer, "_is_gemini_provider", lambda: True)
    monkeypatch.setattr(
        summarizer, "_upload_audio_to_gemini",
        lambda path, progress_cb=None: _FakeAudioFile(),
    )

    def boom(audio_file, progress_cb=None):
        raise RuntimeError(
            "400 INVALID_ARGUMENT. The input token count exceeds the "
            "maximum number of tokens allowed 1048576."
        )

    monkeypatch.setattr(summarizer, "_synthesize_audio_capture", boom)
    monkeypatch.setattr(
        summarizer, "_extract_audio_capture",
        lambda audio_file: (None, None),
    )
    monkeypatch.setattr(summarizer, "_delete_gemini_file", lambda _f: None)

    chunked_args = {"path": None}

    async def fake_chunked(path, progress_cb):
        chunked_args["path"] = path
        return LessonResult(summary="recovered-via-chunked")

    monkeypatch.setattr(summarizer, "_summarize_audio_chunked", fake_chunked)

    result = asyncio.run(summarizer.summarize_audio("/fake/audio.m4a"))

    assert result.summary == "recovered-via-chunked"
    assert chunked_args["path"] == "/fake/audio.m4a"


def test_summarize_audio_does_not_fall_back_on_other_errors(monkeypatch):
    """Non-token errors must propagate — falling back hides real bugs and
    wastes the chunking-path Gemini calls."""
    monkeypatch.setattr(
        summarizer, "_audio_duration_seconds",
        lambda _p: summarizer._GEMINI_DIRECT_MAX_SECONDS - 60,
    )
    monkeypatch.setattr(summarizer, "_is_gemini_provider", lambda: True)
    monkeypatch.setattr(
        summarizer, "_upload_audio_to_gemini",
        lambda path, progress_cb=None: _FakeAudioFile(),
    )

    def boom(audio_file, progress_cb=None):
        raise RuntimeError("some other API error")

    monkeypatch.setattr(summarizer, "_synthesize_audio_capture", boom)
    monkeypatch.setattr(
        summarizer, "_extract_audio_capture",
        lambda audio_file: (None, None),
    )
    monkeypatch.setattr(summarizer, "_delete_gemini_file", lambda _f: None)

    fallback_called = {"flag": False}

    async def fake_chunked(path, progress_cb):
        fallback_called["flag"] = True
        return LessonResult(summary="should-not-be-called")

    monkeypatch.setattr(summarizer, "_summarize_audio_chunked", fake_chunked)

    with pytest.raises(RuntimeError, match="some other API error"):
        asyncio.run(summarizer.summarize_audio("/fake/audio.m4a"))

    assert fallback_called["flag"] is False


# ── Parallelism ───────────────────────────────────────────────────────────────

def test_chunked_path_runs_chunks_in_parallel(monkeypatch):
    """At any moment, no more than `_GEMINI_CHUNK_PARALLELISM` chunks should be
    transcribing concurrently — but multiple should be in flight at once."""
    fake_chunks = [f"/tmp/chunk{i}.mp3" for i in range(8)]

    monkeypatch.setattr(
        summarizer, "_hard_split_audio_for_gemini", lambda _p: list(fake_chunks)
    )
    monkeypatch.setattr(summarizer, "_cleanup_chunk_files", lambda _c: None)

    in_flight = {"current": 0, "max": 0}
    import threading
    lock = threading.Lock()

    def fake_transcribe(chunk_path):
        with lock:
            in_flight["current"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["current"])
        # Simulate work — long enough that parallel calls overlap
        import time as _t
        _t.sleep(0.05)
        with lock:
            in_flight["current"] -= 1
        return f"text-{chunk_path.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(
        summarizer, "_transcribe_audio_chunk_via_gemini", fake_transcribe
    )

    async def fake_summarize_transcript(transcript, progress_cb=None):
        return LessonResult(summary="merged")

    monkeypatch.setattr(
        summarizer, "summarize_transcript", fake_summarize_transcript
    )

    asyncio.run(
        summarizer._summarize_audio_chunked("/fake/long.m4a", progress_cb=None)
    )

    # Expect parallelism between 2 and the configured cap.
    assert in_flight["max"] >= 2, (
        f"chunks ran serially (max in-flight = {in_flight['max']})"
    )
    assert in_flight["max"] <= summarizer._GEMINI_CHUNK_PARALLELISM, (
        f"chunks exceeded parallelism cap "
        f"(max in-flight = {in_flight['max']}, "
        f"cap = {summarizer._GEMINI_CHUNK_PARALLELISM})"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

class _FakeAudioFile:
    name = "files/fake-id"

    class _State:
        name = "ACTIVE"

    state = _State()

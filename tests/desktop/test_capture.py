"""
Unit tests for the desktop loopback capture POC (Task 2.1).

All tests are pure-logic with mocks — no PortAudio hardware required.
The hardware-gated test at the bottom is skipped in CI and can be run
manually on a Windows machine via `pytest tests/desktop/ -m hardware`.
"""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

# Skip the whole module cleanly when desktop/ deps are not installed.
# Users who want to run the desktop POC: pip install -r desktop/requirements.txt.
pytest.importorskip("sounddevice")
pytest.importorskip("soundfile")

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf

from desktop.capture import (
    DualStreamRecorder,
    SilenceDetector,
    UploadError,
    build_parser,
    main,
    upload_wav,
)


# ── SilenceDetector ──────────────────────────────────────────────────────────

SR = 48000


def test_silence_detector_triggers_after_threshold():
    """1s loud, then 240s of zeros → triggers near the 120s mark."""
    det = SilenceDetector(samplerate=SR, threshold_db=-50.0, duration_sec=120.0)

    # 1s of loud frames (amplitude 0.5 → ~ -6 dBFS, well above -50)
    loud = np.full(SR, 0.5, dtype=np.float32)
    assert det.feed(loud) is False  # arms but doesn't trigger

    # Feed in 1-second silent chunks until trigger (or fail-safe at 250s)
    triggered = False
    for sec in range(1, 251):
        if det.feed(np.zeros(SR, dtype=np.float32)):
            triggered = True
            break

    assert triggered, "Detector never triggered"
    # Should fire on the second after we cross 120s of silence
    # (allow ±2s slack for window-boundary effects)
    assert 119 <= sec <= 122, f"Triggered at second {sec}, expected ~120"


def test_silence_detector_not_armed_until_loud_frame():
    """300s of zeros from start → never triggers (not yet armed)."""
    det = SilenceDetector(samplerate=SR, threshold_db=-50.0, duration_sec=120.0)
    for _ in range(300):
        assert det.feed(np.zeros(SR, dtype=np.float32)) is False


def test_silence_resets_on_loud_frame():
    """60s silence + 1s loud + 60s silence → never triggers (silence reset)."""
    det = SilenceDetector(samplerate=SR, threshold_db=-50.0, duration_sec=120.0)
    # Arm
    det.feed(np.full(SR, 0.5, dtype=np.float32))
    # 60s silence (just under threshold)
    for _ in range(60):
        assert det.feed(np.zeros(SR, dtype=np.float32)) is False
    # 1s loud — resets silent counter
    assert det.feed(np.full(SR, 0.5, dtype=np.float32)) is False
    # 60s silence — total silent run is 60s, below 120s threshold
    for _ in range(60):
        assert det.feed(np.zeros(SR, dtype=np.float32)) is False


# ── DualStreamRecorder (mocked sounddevice) ─────────────────────────────────


class _FakeStream:
    """Minimal sd.InputStream replacement that pushes synthetic frames."""

    instances: list = []  # registry of created streams (for assertions)

    def __init__(self, *, callback, channels, samplerate, dtype, **kw):
        self.callback = callback
        self.channels = channels
        self.samplerate = samplerate
        self.dtype = dtype
        self._thread = None
        self._stop = threading.Event()
        _FakeStream.instances.append(self)

    def start(self):
        # Fire ~10 synthetic frames at 100ms cadence, then stop.
        def _loop():
            frame_samples = int(self.samplerate * 0.1)  # 100ms frame
            for _ in range(10):
                if self._stop.is_set():
                    return
                # Mild non-zero signal so the silence detector doesn't fire instantly.
                arr = np.full(
                    (frame_samples, self.channels) if self.channels > 1 else frame_samples,
                    0.1,
                    dtype=np.float32,
                )
                if self.channels > 1:
                    self.callback(arr, frame_samples, None, None)
                else:
                    self.callback(arr.reshape(-1, 1), frame_samples, None, None)
                time.sleep(0.05)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def close(self):
        self._stop.set()


@pytest.fixture
def fake_streams(monkeypatch):
    """Patch sd.InputStream and sd.WasapiSettings."""
    _FakeStream.instances = []
    monkeypatch.setattr(sd, "InputStream", _FakeStream)
    # WasapiSettings only exists on Windows; create a simple stand-in.
    monkeypatch.setattr(sd, "WasapiSettings", lambda **kw: object(), raising=False)
    yield _FakeStream


def test_dual_stream_writes_mono_wav(fake_streams, tmp_path, monkeypatch):
    """Recorder writes a mono 48 kHz PCM_16 WAV with content from the streams."""
    # Force the Windows code path so loopback stream is created too.
    monkeypatch.setattr(sys, "platform", "win32")

    out = tmp_path / "test.wav"
    recorder = DualStreamRecorder(
        output_path=out,
        samplerate=SR,
        silence_detector=None,
        max_duration_sec=1.0,
    )
    recorder.start()
    recorder.wait_for_stop()

    info = sf.info(str(out))
    assert info.channels == 1
    assert info.samplerate == SR
    assert info.subtype == "PCM_16"
    assert info.frames > 0  # something was written


# ── upload_wav ───────────────────────────────────────────────────────────────


def _wav(tmp_path: Path) -> Path:
    p = tmp_path / "x.wav"
    sf.write(str(p), np.zeros(SR // 10, dtype=np.float32), SR, subtype="PCM_16")
    return p


def test_upload_posts_multipart_with_cookie(tmp_path, monkeypatch):
    """Upload sends multipart with file, mode, language, and session_id cookie."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["cookie"] = request.headers.get("cookie", "")
        seen["body"] = request.content
        return httpx.Response(202, json={"task_id": "abc"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)

    result = upload_wav(
        "https://server.example",
        _wav(tmp_path),
        mode="gemini_direct",
        language="he",
        session_id="cookieval",
    )

    assert result == {"task_id": "abc"}
    assert seen["url"].endswith("/api/tasks/upload")
    assert "session_id=cookieval" in seen["cookie"]
    body = seen["body"]
    assert b'name="file"' in body
    assert b"audio/wav" in body
    assert b'name="mode"' in body and b"gemini_direct" in body
    assert b'name="language"' in body and b"he" in body


def test_upload_surfaces_server_detail(tmp_path, monkeypatch):
    """A 4xx response with JSON detail is surfaced in UploadError message."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "Unsupported file type"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)

    with pytest.raises(UploadError) as exc:
        upload_wav(
            "https://server.example",
            _wav(tmp_path),
            mode="gemini_direct",
            language="he",
            session_id="x",
        )
    assert "Unsupported file type" in str(exc.value)
    assert "400" in str(exc.value)


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_main_no_upload_skips_post(fake_streams, tmp_path, monkeypatch):
    """--no-upload runs the recorder but never invokes upload_wav."""
    monkeypatch.setattr(sys, "platform", "win32")

    out = tmp_path / "rec.wav"
    upload_called = []

    def fake_upload(*args, **kwargs):
        upload_called.append(True)
        return {}

    monkeypatch.setattr("desktop.capture.upload_wav", fake_upload)

    rc = main([
        "--url", "https://example",
        "--no-upload",
        "--duration", "1",
        "--output", str(out),
    ])

    assert rc == 0
    assert out.exists()
    assert upload_called == []


def test_main_missing_cookie_exits_2(monkeypatch, tmp_path):
    """No --cookie and no $ZTT_SESSION → exit 2 with clear stderr message."""
    monkeypatch.delenv("ZTT_SESSION", raising=False)
    rc = main([
        "--url", "https://example",
        "--duration", "1",
        "--output", str(tmp_path / "x.wav"),
    ])
    assert rc == 2


# ── Hardware-gated (Windows + real PortAudio) ───────────────────────────────


@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("CI"),
    reason="requires real WASAPI loopback hardware",
)
def test_real_wasapi_loopback_records_2s(tmp_path):
    """Sanity check on Asaf's machine — records 2s and verifies non-empty WAV."""
    out = tmp_path / "real.wav"
    recorder = DualStreamRecorder(
        output_path=out,
        samplerate=SR,
        max_duration_sec=2.0,
    )
    recorder.start()
    recorder.wait_for_stop()
    info = sf.info(str(out))
    assert info.frames > 0
    assert info.samplerate == SR

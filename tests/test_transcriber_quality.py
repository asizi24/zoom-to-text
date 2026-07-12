"""Unit tests for the transcription-quality upgrade.

Covers:
  - the decode parameters passed to faster-whisper (accuracy +
    anti-hallucination settings driven by config)
  - device/compute auto-resolution (CUDA when available, CPU fallback)
  - the audio normalization filter in the preprocessor's ffmpeg steps
"""
import asyncio
import sys
import types

import pytest

from app.config import settings
from app.services import audio_preprocessor, transcriber


# ── Decode parameters ─────────────────────────────────────────────────────────


class _FakeInfo:
    duration = 10.0
    language = "he"


class _CaptureModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return iter([]), _FakeInfo()


def test_transcribe_sync_uses_quality_settings():
    model = _CaptureModel()
    transcriber._transcribe_sync(model, "x.mp3", "he")

    kw = model.kwargs
    assert kw["beam_size"] == settings.whisper_beam_size
    assert kw["best_of"] == settings.whisper_best_of
    assert kw["vad_filter"] is True
    assert kw["vad_parameters"]["min_silence_duration_ms"] == settings.whisper_vad_min_silence_ms
    assert kw["vad_parameters"]["speech_pad_ms"] == settings.whisper_vad_speech_pad_ms
    assert kw["condition_on_previous_text"] == settings.whisper_condition_on_previous_text
    # Temperature ladder: starts greedy, escalates on rejected windows
    assert kw["temperature"][0] == 0.0
    assert kw["temperature"][-1] == 1.0
    assert kw["compression_ratio_threshold"] == 2.4
    # Empty initial prompt must become None, not ""
    assert kw["initial_prompt"] is None


def test_transcribe_sync_passes_initial_prompt(monkeypatch):
    monkeypatch.setattr(settings, "whisper_initial_prompt", "מונחים: ניתוב, סאבנט")
    model = _CaptureModel()
    transcriber._transcribe_sync(model, "x.mp3", "he")
    assert model.kwargs["initial_prompt"] == "מונחים: ניתוב, סאבנט"


# ── Device auto-resolution ────────────────────────────────────────────────────


def _fake_ct2(monkeypatch, cuda_devices: int):
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        types.SimpleNamespace(get_cuda_device_count=lambda: cuda_devices),
    )


def test_resolve_device_auto_without_cuda(monkeypatch):
    monkeypatch.setattr(settings, "whisper_device", "auto")
    monkeypatch.setattr(settings, "whisper_compute_type", "auto")
    _fake_ct2(monkeypatch, 0)
    assert transcriber._resolve_device() == ("cpu", "int8")


def test_resolve_device_auto_with_cuda(monkeypatch):
    monkeypatch.setattr(settings, "whisper_device", "auto")
    monkeypatch.setattr(settings, "whisper_compute_type", "auto")
    _fake_ct2(monkeypatch, 1)
    assert transcriber._resolve_device() == ("cuda", "float16")


def test_resolve_device_explicit_settings_win(monkeypatch):
    monkeypatch.setattr(settings, "whisper_device", "cpu")
    monkeypatch.setattr(settings, "whisper_compute_type", "int8")
    _fake_ct2(monkeypatch, 4)  # CUDA visible but explicitly overridden
    assert transcriber._resolve_device() == ("cpu", "int8")


def test_resolve_device_survives_broken_probe(monkeypatch):
    monkeypatch.setattr(settings, "whisper_device", "auto")
    monkeypatch.setattr(settings, "whisper_compute_type", "auto")

    def boom():
        raise RuntimeError("no cuda driver")

    monkeypatch.setitem(
        sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=boom)
    )
    assert transcriber._resolve_device() == ("cpu", "int8")


# ── Audio normalization ───────────────────────────────────────────────────────


def test_extract_audio_track_applies_normalization(tmp_path, monkeypatch):
    src = tmp_path / "lecture.mp4"
    src.write_bytes(b"fake video bytes")
    dest = tmp_path / "lecture.mp3"
    captured = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            dest.write_bytes(b"fake mp3")
            return b"", b""

    async def fake_exec(*cmd, stdout=None, stderr=None):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    out = asyncio.run(audio_preprocessor.extract_audio_track(str(src)))

    assert out == str(dest)
    assert not src.exists(), "source must be deleted after successful extraction"
    cmd = captured["cmd"]
    af = cmd[cmd.index("-af") + 1]
    assert "dynaudnorm" in af
    assert "highpass" in af


def test_extract_audio_track_skips_mp3_reencode(tmp_path):
    src = tmp_path / "already.mp3"
    src.write_bytes(b"mp3")
    out = asyncio.run(audio_preprocessor.extract_audio_track(str(src)))
    assert out == str(src)
    assert src.exists()


def test_remove_silence_filter_normalizes_after_silenceremove(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(audio_preprocessor.subprocess, "run", fake_run)
    audio_preprocessor._remove_silence("in.mp3")

    af = captured["cmd"][captured["cmd"].index("-af") + 1]
    assert "dynaudnorm" in af
    # Normalization must run AFTER silence removal, or boosted room noise
    # defeats the -40 dB silence detector.
    assert af.index("silenceremove") < af.index("dynaudnorm")

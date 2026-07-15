"""
Tests for the Setup Wizard's hardware probe. The nvidia-smi call is always
stubbed so results are deterministic on any host (including this dev box, which
actually has an RTX 4070 Ti).
"""
import subprocess

import app.services.hardware as hardware
from app.services.hardware import detect_hardware


def test_strong_gpu_recommends_local_ollama(monkeypatch):
    monkeypatch.setattr(
        hardware, "_query_nvidia_smi",
        lambda: [("NVIDIA GeForce RTX 4070 Ti", 12282)],
    )
    info = detect_hardware("gemma2:9b")
    assert info.strong_gpu is True
    assert info.cpu_only is False
    assert info.cuda_available is True
    assert info.recommended_backend == "ollama"
    assert info.recommended_model == "gemma2:9b"
    assert info.gpu_name == "NVIDIA GeForce RTX 4070 Ti"
    assert info.vram_mb == 12282


def test_weak_gpu_recommends_gemini(monkeypatch):
    monkeypatch.setattr(
        hardware, "_query_nvidia_smi",
        lambda: [("NVIDIA GeForce GTX 1050", 4096)],
    )
    info = detect_hardware()
    assert info.strong_gpu is False
    assert info.cpu_only is False        # a GPU exists, just not a strong one
    assert info.recommended_backend == "gemini"
    assert info.recommended_model == ""


def test_cpu_only_recommends_gemini(monkeypatch):
    monkeypatch.setattr(hardware, "_query_nvidia_smi", lambda: [])
    info = detect_hardware()
    assert info.cpu_only is True
    assert info.cuda_available is False
    assert info.gpu_name is None
    assert info.recommended_backend == "gemini"


def test_picks_strongest_gpu(monkeypatch):
    monkeypatch.setattr(
        hardware, "_query_nvidia_smi",
        lambda: [("weak", 4096), ("strong", 16000)],
    )
    info = detect_hardware()
    assert info.gpu_name == "strong"
    assert info.vram_mb == 16000
    assert info.gpu_count == 2


def test_query_parses_nvidia_smi_csv(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class _Proc:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 4070 Ti, 12282\nNVIDIA A100, 40960\n"

    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: _Proc())
    assert hardware._query_nvidia_smi() == [
        ("NVIDIA GeForce RTX 4070 Ti", 12282),
        ("NVIDIA A100", 40960),
    ]


def test_query_empty_without_binary(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    assert hardware._query_nvidia_smi() == []


def test_query_handles_timeout(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=8)

    monkeypatch.setattr(hardware.subprocess, "run", _boom)
    assert hardware._query_nvidia_smi() == []

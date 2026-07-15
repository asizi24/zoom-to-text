"""
Hardware probe for the first-boot Setup Wizard.

Detects whether the host has a CUDA-capable NVIDIA GPU strong enough to run a
local LLM (Ollama), and recommends a Smart Summary backend accordingly:

  • strong GPU (≥ ~8 GB VRAM, e.g. RTX 4070 Ti / 12 GB) → local Ollama
  • weak GPU or CPU-only                                 → Gemini cloud fallback

Probing is done with `nvidia-smi` (a subprocess, no torch/pynvml dependency —
consistent with the project's stdlib-only infrastructure preference). The probe
never raises: a missing binary, non-zero exit, or timeout all read as CPU-only.
"""
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# A local 7-9B model in 4-bit needs ~6 GB VRAM plus headroom; 8 GB is the floor
# below which we steer the user to the cloud fallback instead.
STRONG_GPU_MIN_VRAM_MB = 8000


@dataclass
class HardwareInfo:
    cuda_available: bool
    gpu_name: Optional[str]
    vram_mb: Optional[int]
    gpu_count: int
    strong_gpu: bool
    cpu_only: bool
    recommended_backend: str   # "ollama" | "gemini"
    recommended_model: str     # e.g. "gemma2:9b" (empty for the gemini path)
    detail: str                # human-readable Hebrew note for the wizard UI

    def as_dict(self) -> dict:
        return asdict(self)


def _query_nvidia_smi() -> list[tuple[str, int]]:
    """Return [(gpu_name, vram_mb), …] via nvidia-smi, or [] if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("nvidia-smi probe failed (treating as CPU-only): %s", exc)
        return []
    if proc.returncode != 0:
        return []

    gpus: list[tuple[str, int]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            vram_mb = int(float(parts[1]))
        except ValueError:
            continue
        gpus.append((parts[0], vram_mb))
    return gpus


def detect_hardware(default_model: str = "gemma2:9b") -> HardwareInfo:
    """Probe the host and recommend a Smart Summary backend."""
    gpus = _query_nvidia_smi()

    if not gpus:
        return HardwareInfo(
            cuda_available=False, gpu_name=None, vram_mb=None, gpu_count=0,
            strong_gpu=False, cpu_only=True,
            recommended_backend="gemini", recommended_model="",
            detail="לא זוהה כרטיס מסך NVIDIA — מומלץ להשתמש ב-Gemini בענן לסיכומים.",
        )

    # Recommend based on the strongest GPU present.
    name, vram_mb = max(gpus, key=lambda g: g[1])
    vram_gb = vram_mb // 1024
    strong = vram_mb >= STRONG_GPU_MIN_VRAM_MB

    if strong:
        return HardwareInfo(
            cuda_available=True, gpu_name=name, vram_mb=vram_mb, gpu_count=len(gpus),
            strong_gpu=True, cpu_only=False,
            recommended_backend="ollama", recommended_model=default_model,
            detail=f"זוהה GPU חזק ({name}, ‎{vram_gb}GB VRAM) — מומלץ מודל מקומי (Ollama).",
        )

    return HardwareInfo(
        cuda_available=True, gpu_name=name, vram_mb=vram_mb, gpu_count=len(gpus),
        strong_gpu=False, cpu_only=False,
        recommended_backend="gemini", recommended_model="",
        detail=f"זוהה GPU חלש ({name}, ‎{vram_gb}GB VRAM) — מומלץ Gemini בענן.",
    )

"""Desktop loopback capture POC for zoom-to-text (Task 2.1)."""
import argparse
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import httpx

__all__ = [
    "SilenceDetector",
    "DualStreamRecorder",
    "UploadError",
    "upload_wav",
    "build_parser",
    "main",
]

# ---------------------------------------------------------------------------
# Silence detection (loopback only)
# ---------------------------------------------------------------------------

class SilenceDetector:
    """RMS-based silence detector fed by loopback frames.

    Armed only after the first loud frame. Triggers when the loopback RMS
    stays below *threshold_db* for *duration_sec* consecutive seconds.
    RMS is computed in 0.5-second windows.
    """

    WINDOW_SEC = 0.5

    def __init__(
        self,
        samplerate: int,
        threshold_db: float = -50.0,
        duration_sec: float = 120.0,
    ) -> None:
        self.samplerate = samplerate
        self.threshold_db = threshold_db
        self.duration_sec = duration_sec

        self._window_samples = int(samplerate * self.WINDOW_SEC)
        self._silent_windows_needed = int(math.ceil(duration_sec / self.WINDOW_SEC))

        self._armed = False          # True after first loud frame
        self._silent_count = 0       # consecutive silent windows
        self._buffer = np.empty(0, dtype=np.float32)  # accumulates inter-frame samples

    # ------------------------------------------------------------------
    def _rms_db(self, samples: np.ndarray) -> float:
        """Return RMS amplitude in dBFS. Uses 1e-10 epsilon to avoid log10(0)."""
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        return 20.0 * math.log10(max(rms, 1e-10))

    # ------------------------------------------------------------------
    def feed(self, samples: np.ndarray) -> bool:
        """Feed a chunk of loopback samples. Return True when stop condition met."""
        # Flatten to mono for analysis
        mono = samples.mean(axis=1) if samples.ndim == 2 else samples.ravel()

        self._buffer = np.concatenate([self._buffer, mono.astype(np.float32)])

        triggered = False
        while len(self._buffer) >= self._window_samples:
            window = self._buffer[: self._window_samples]
            self._buffer = self._buffer[self._window_samples :]

            db = self._rms_db(window)
            is_loud = db >= self.threshold_db

            if is_loud:
                self._armed = True
                self._silent_count = 0  # reset on any loud window
            else:
                if self._armed:
                    self._silent_count += 1
                    if self._silent_count >= self._silent_windows_needed:
                        triggered = True
                        break

        return triggered


# ---------------------------------------------------------------------------
# Dual-stream recorder
# ---------------------------------------------------------------------------

class DualStreamRecorder:
    """Loopback (WASAPI on Windows) + mic, summed to mono 48 kHz PCM_16 WAV.

    Streaming-only: no large in-RAM buffer.
    """

    def __init__(
        self,
        output_path,
        samplerate: int = 48000,
        silence_detector: SilenceDetector | None = None,
        max_duration_sec: float | None = None,
        loopback_device=None,
        mic_device=None,
    ) -> None:
        self.output_path = Path(output_path)
        self.samplerate = samplerate
        self.silence_detector = silence_detector
        self.max_duration_sec = max_duration_sec
        self.loopback_device = loopback_device
        self.mic_device = mic_device

        self._loopback_q: queue.Queue = queue.Queue()
        self._mic_q: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._streams: list = []

    # ------------------------------------------------------------------
    def _loopback_callback(self, indata, frames, time_info, status):
        self._loopback_q.put(indata.copy())

    def _mic_callback(self, indata, frames, time_info, status):
        self._mic_q.put(indata.copy())

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open audio streams and start the writer thread."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open WAV file for streaming writes
        self._sf = sf.SoundFile(
            str(self.output_path),
            mode="w",
            samplerate=self.samplerate,
            channels=1,
            subtype="PCM_16",
        )

        # Loopback stream (Windows-only)
        self._has_loopback = False
        if sys.platform == "win32":
            try:
                loopback_stream = sd.InputStream(
                    device=self.loopback_device,
                    samplerate=self.samplerate,
                    channels=2,
                    dtype="float32",
                    callback=self._loopback_callback,
                    extra_settings=sd.WasapiSettings(loopback=True),
                )
                self._streams.append(loopback_stream)
                self._has_loopback = True
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: could not open loopback stream: {exc}. Falling back to mic-only.")
        else:
            print("Warning: WASAPI loopback is only available on Windows. Recording mic-only.")

        # Mic stream
        mic_stream = sd.InputStream(
            device=self.mic_device,
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            callback=self._mic_callback,
        )
        self._streams.append(mic_stream)

        for stream in self._streams:
            stream.start()

        self._start_time = time.monotonic()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    # ------------------------------------------------------------------
    def _writer_loop(self) -> None:
        """Pull frames from queues, mix to mono, write PCM_16 to WAV."""
        try:
            while not self._stop_event.is_set():
                # Duration cap
                if self.max_duration_sec is not None:
                    elapsed = time.monotonic() - self._start_time
                    if elapsed >= self.max_duration_sec:
                        self._stop_event.set()
                        break

                # Get mic frame (block briefly)
                try:
                    mic_frame = self._mic_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                mic_mono = mic_frame.ravel()  # already 1-channel

                if self._has_loopback:
                    # Best-effort alignment: consume matching loopback frame
                    try:
                        loop_frame = self._loopback_q.get_nowait()
                    except queue.Empty:
                        loop_frame = None

                    if loop_frame is not None:
                        loop_mono = loop_frame.mean(axis=1)

                        # Check silence on loopback
                        if self.silence_detector is not None:
                            if self.silence_detector.feed(loop_frame):
                                # pad any size difference then write last frame
                                mixed = self._mix(mic_mono, loop_mono)
                                self._sf.write(mixed)
                                self._stop_event.set()
                                break

                        # Ensure same length before mixing
                        mixed = self._mix(mic_mono, loop_mono)
                    else:
                        # No loopback frame yet — write mic only (/ 2 for headroom)
                        mixed = np.clip(mic_mono / 2.0, -1.0, 1.0)
                else:
                    mixed = np.clip(mic_mono / 2.0, -1.0, 1.0)

                self._sf.write(mixed)
        finally:
            self._sf.flush()
            self._sf.close()

    @staticmethod
    def _mix(mic_mono: np.ndarray, loop_mono: np.ndarray) -> np.ndarray:
        """Sum mic + loopback, divide by 2, clip to [-1, 1]. Handles length mismatch."""
        n = min(len(mic_mono), len(loop_mono))
        mixed = (mic_mono[:n].astype(np.float64) + loop_mono[:n].astype(np.float64)) / 2.0
        return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Signal streams and writer thread to stop."""
        self._stop_event.set()

    def wait_for_stop(self) -> None:
        """Block until silence, --duration cap, or KeyboardInterrupt."""
        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            self._stop_event.set()
        finally:
            # Stop all streams
            for stream in self._streams:
                try:
                    stream.stop()
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
            self._streams.clear()
            # Wait for writer to finish flushing
            if self._writer_thread is not None:
                self._writer_thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class UploadError(Exception):
    """Raised when the server returns a 4xx/5xx response."""


def upload_wav(
    server_url: str,
    wav_path,
    *,
    mode: str,
    language: str,
    session_id: str,
    timeout: float = 600.0,
) -> dict:
    """POST the WAV to /api/tasks/upload as multipart with cookie auth."""
    wav_path = Path(wav_path)
    url = f"{server_url.rstrip('/')}/api/tasks/upload"

    with httpx.Client(timeout=timeout) as client:
        with wav_path.open("rb") as fh:
            response = client.post(
                url,
                files={"file": (wav_path.name, fh, "audio/wav")},
                data={"mode": mode, "language": language},
                cookies={"session_id": session_id},
            )

    if response.is_error:
        detail = str(response.status_code)
        try:
            body = response.json()
            detail = body.get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        raise UploadError(f"HTTP {response.status_code}: {detail}")

    return response.json()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m desktop.capture",
        description="Record system audio + mic and upload to zoom-to-text.",
    )
    p.add_argument("--url", default="", help="Server URL, e.g. https://zoom-to-text.fly.dev")
    p.add_argument(
        "--duration",
        default="auto",
        help="Recording duration in seconds, or 'auto' for silence-based stop (default: auto)",
    )
    p.add_argument("--silence-threshold-db", type=float, default=-50.0, metavar="DB")
    p.add_argument("--silence-duration-sec", type=float, default=120.0, metavar="SEC")
    p.add_argument(
        "--mode",
        default="gemini_direct",
        choices=["gemini_direct", "whisper_local", "whisper_api", "ivrit_ai"],
    )
    p.add_argument("--language", default="he")
    p.add_argument("--cookie", default=None, help="session_id cookie value (or set $ZTT_SESSION)")
    p.add_argument("--output", default=None, help="Output WAV path (default: ./recordings/<UTC>.wav)")
    p.add_argument("--no-upload", action="store_true", help="Record only; skip upload")
    p.add_argument("--loopback-device", default=None, help="Loopback device name/index substring")
    p.add_argument("--mic-device", default=None, help="Mic device name/index substring")
    p.add_argument("--list-devices", action="store_true", help="Print audio devices and exit")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --list-devices
    if args.list_devices:
        print(sd.query_devices())
        return 0

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("recordings") / f"{ts}.wav"

    # Resolve duration
    max_duration: float | None = None
    if args.duration != "auto":
        try:
            max_duration = float(args.duration)
        except ValueError:
            print(f"Error: --duration must be 'auto' or a number (got {args.duration!r})", file=sys.stderr)
            return 1

    # Resolve cookie (only needed for upload)
    session_id: str | None = None
    if not args.no_upload:
        session_id = args.cookie or os.environ.get("ZTT_SESSION")
        if not session_id:
            print(
                "Error: session_id required for upload. "
                "Provide --cookie <session_id> or set $ZTT_SESSION. "
                "Use --no-upload to skip.",
                file=sys.stderr,
            )
            return 2

    # Build silence detector
    detector = SilenceDetector(
        samplerate=48000,
        threshold_db=args.silence_threshold_db,
        duration_sec=args.silence_duration_sec,
    )

    # Build recorder
    recorder = DualStreamRecorder(
        output_path=output_path,
        samplerate=48000,
        silence_detector=detector,
        max_duration_sec=max_duration,
        loopback_device=args.loopback_device,
        mic_device=args.mic_device,
    )

    print(f"Recording to {output_path} ... (Ctrl+C to stop)")
    recorder.start()
    recorder.wait_for_stop()
    print(f"Saved: {output_path}")

    if args.no_upload:
        return 0

    if not args.url:
        print("Error: --url is required when uploading.", file=sys.stderr)
        return 1

    print(f"Uploading to {args.url} ...")
    try:
        result = upload_wav(
            args.url,
            output_path,
            mode=args.mode,
            language=args.language,
            session_id=session_id,  # type: ignore[arg-type]
        )
        task_id = result.get("task_id", result)
        print(f"Uploaded. Task ID: {task_id}")
    except UploadError as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        print(f"WAV preserved at: {output_path}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

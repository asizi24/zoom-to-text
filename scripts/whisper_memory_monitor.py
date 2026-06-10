#!/usr/bin/env python3
"""
Whisper memory leak monitor.

Sends 3 WHISPER_LOCAL transcription tasks to a locally-running server,
waits for the idle window to expire, then compares RSS before vs. after.

Prerequisites
-------------
1.  Set AUTO_SHUTDOWN_IDLE_MINUTES=2 in .env  (so the wait is ~2 min)
2.  docker compose up -d
3.  Log in at http://localhost:8000, open DevTools > Application > Cookies,
    copy the value of the `session_id` cookie.

Usage
-----
    python scripts/whisper_memory_monitor.py --session <session_id_value>

Options
-------
    --session       session_id cookie value (required)
    --base-url      server URL             (default: http://localhost:8000)
    --container     docker container name  (default: zoom_transcriber)
    --idle-minutes  wait time in minutes   (default: 2;
                    must match AUTO_SHUTDOWN_IDLE_MINUTES in .env)

What to look for
----------------
After the script finishes, run:
    docker logs zoom_transcriber | grep -i "whisper"

You should see lines like:
    Whisper idle for 2.0 min ... — unloading (RSS before: 1850 MB)
    Whisper unloaded — RSS 1850 MB → 320 MB (freed 1530 MB)

If "freed" < 500 MB the model is NOT releasing RAM and you have a leak.
"""
import argparse
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave


# ── Helpers ───────────────────────────────────────────────────────────────────

def docker_rss_mb(container: str) -> float:
    """Container RSS in MiB via `docker stats --no-stream`. Returns -1.0 on error."""
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True, text=True, timeout=10,
        )
        raw = result.stdout.strip().split("/")[0].strip()
        if raw.endswith("MiB"):
            return float(raw[:-3])
        if raw.endswith("GiB"):
            return float(raw[:-3]) * 1024
        if raw.endswith("kB"):
            return float(raw[:-2]) / 1024
    except Exception as e:
        print(f"  [docker stats error] {e}", file=sys.stderr)
    return -1.0


def rss_str(container: str) -> str:
    mb = docker_rss_mb(container)
    return f"{mb:.0f} MiB" if mb >= 0 else "N/A"


def make_wav(duration_secs: int = 3, sample_rate: int = 16000) -> bytes:
    """Generate a minimal silent WAV (mono 16-bit PCM)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * sample_rate * duration_secs)
    return buf.getvalue()


def build_multipart(fields: dict[str, str], files: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body using only stdlib."""
    boundary = b"Boundary7MA4YWxkTrZu0gW"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="' + name.encode() + b'"\r\n'
            b"\r\n" + value.encode() + b"\r\n"
        )
    for name, (filename, data, ctype) in files.items():
        parts.append(
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="' + name.encode()
            + b'"; filename="' + filename.encode() + b'"\r\n'
            b"Content-Type: " + ctype.encode() + b"\r\n"
            b"\r\n" + data + b"\r\n"
        )
    body = b"".join(parts) + b"--" + boundary + b"--\r\n"
    ct = "multipart/form-data; boundary=" + boundary.decode()
    return body, ct


def upload_task(base_url: str, session: str, wav: bytes, n: int) -> str | None:
    """POST a WAV as a whisper_local task. Returns task_id or None on failure."""
    body, ct = build_multipart(
        {"mode": "whisper_local", "language": "he"},
        {"file": ("test.wav", wav, "audio/wav")},
    )
    req = urllib.request.Request(
        f"{base_url}/api/tasks/upload", data=body, method="POST"
    )
    req.add_header("Content-Type", ct)
    req.add_header("Cookie", f"session_id={session}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())
            tid = payload.get("task_id", "?")
            print(f"  task {n}: {tid}")
            return tid
    except urllib.error.HTTPError as e:
        snippet = e.read().decode()[:200]
        print(f"  task {n}: HTTP {e.code} — {snippet}")
    except Exception as e:
        print(f"  task {n}: {e}")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper memory leak monitor")
    parser.add_argument("--session", required=True, help="session_id cookie value")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--container", default="zoom_transcriber")
    parser.add_argument(
        "--idle-minutes", type=int, default=2,
        help="Minutes to wait for idle unload (must match AUTO_SHUTDOWN_IDLE_MINUTES in .env)",
    )
    args = parser.parse_args()

    print("=== Whisper memory monitor ===")
    print(f"Container    : {args.container}")
    print(f"Server       : {args.base_url}")
    print(f"Idle wait    : {args.idle_minutes} min (+ 70s watcher buffer)")
    print()

    loaded_mb: float = -1.0   # set after model snapshot; -1.0 = unavailable

    # ── Step 1: baseline ──────────────────────────────────────────────────────
    baseline = docker_rss_mb(args.container)
    print(f"[1/4] Baseline RSS     : {rss_str(args.container)}")

    # ── Step 2: submit 3 tasks ────────────────────────────────────────────────
    print("[2/4] Submitting 3 whisper_local tasks ...")
    wav = make_wav()
    task_ids = []
    for i in range(1, 4):
        tid = upload_task(args.base_url, args.session, wav, i)
        if tid:
            task_ids.append(tid)
        time.sleep(2)

    if not task_ids:
        print(
            "\n  ERROR: no tasks submitted.\n"
            "  Check --session value and that the server is running.\n"
            "  Hint: open DevTools > Application > Cookies > session_id",
            file=sys.stderr,
        )
        sys.exit(1)

    # Give the server a moment to start loading the model, then snapshot
    time.sleep(8)
    loaded_mb = docker_rss_mb(args.container)
    print(f"  RSS with model loaded: {loaded_mb:.0f} MiB" if loaded_mb >= 0 else "  RSS with model loaded: N/A")

    # ── Step 3: wait for idle unload ──────────────────────────────────────────
    # The idle watcher checks every 60 s, so add 70 s of buffer on top of
    # the configured idle window.
    wait_secs = args.idle_minutes * 60 + 70
    print(f"[3/4] Waiting {wait_secs}s for idle window to expire ...")
    deadline = time.time() + wait_secs
    while True:
        remaining = int(deadline - time.time())
        if remaining <= 0:
            break
        print(f"  {remaining:3d}s remaining — RSS: {rss_str(args.container)}", end="\r")
        time.sleep(15)
    print()

    # ── Step 4: final measurement ─────────────────────────────────────────────
    after_mb = docker_rss_mb(args.container)
    print(f"[4/4] RSS after unload : {after_mb:.0f} MiB" if after_mb >= 0 else "[4/4] RSS after unload : N/A")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=== Summary ===")
    if loaded_mb >= 0 and after_mb >= 0:
        freed = loaded_mb - after_mb
        print(f"  Peak RSS   : {loaded_mb:.0f} MiB")
        print(f"  After RSS  : {after_mb:.0f} MiB")
        print(f"  Freed      : {freed:.0f} MiB")
        if freed < 400:
            print("  RESULT: ⚠️  less than 400 MiB freed — potential memory leak")
        else:
            print("  RESULT: ✅ model memory released cleanly")
    else:
        print("  Could not read docker stats — check container name with `docker ps`")

    print()
    print(f"Check server logs with:")
    print(f"  docker logs {args.container} | grep -iE 'whisper|ivrit'")


if __name__ == "__main__":
    main()

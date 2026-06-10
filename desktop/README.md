# desktop/ — Loopback Capture POC (Task 2.1)

A standalone Python tool that records system audio + microphone on the
host machine and uploads the resulting WAV to the existing
`zoom-to-text` server. Server is unchanged — the tool POSTs to
`/api/tasks/upload`.

This is **client-side only**. It is **not** installed on the Fly.io
server (its dependencies live in `desktop/requirements.txt`, not the
top-level `requirements.txt`).

## When to use

- Recording any meeting platform without a bot integration:
  Zoom, Google Meet, Microsoft Teams, Discord, etc.
- When Chrome cookies for Zoom recordings stop working — capture
  the live meeting instead.

## Platform support

- **Windows 11**: full functionality (loopback via WASAPI + mic).
- **macOS / Linux**: degraded — mic-only with a printed warning.
  (To capture system audio there you'd need BlackHole / PulseAudio
  loopback configured at the OS level — not handled by this tool.)
- **Python 3.11+** required (PortAudio binaries ship in the
  `sounddevice` Windows wheels for cp311+).

## Install

```powershell
# From repo root, in a separate venv to keep server deps clean
python -m venv .venv-desktop
.venv-desktop\Scripts\activate
pip install -r desktop/requirements.txt
```

## Get a session cookie

The tool needs your `session_id` cookie from the live site to
authenticate uploads:

1. Sign in normally at `https://zoom-to-text.fly.dev/`
2. Open DevTools → Application → Cookies → `https://zoom-to-text.fly.dev`
3. Copy the value of `session_id`
4. Persist it for future shell sessions:

```powershell
setx ZTT_SESSION "<paste cookie value here>"
# Open a new shell to pick up the env var
```

The cookie is valid for 30 days. If uploads start returning 401, refresh.

## Usage

### Silence-stop record (recommended for meetings)

Stops automatically after 2 minutes of silence on the system audio.

```powershell
python -m desktop.capture --url https://zoom-to-text.fly.dev
```

### Fixed-duration record

```powershell
python -m desktop.capture --url https://zoom-to-text.fly.dev --duration 1800
```

### Record only, don't upload

Useful for offline test runs.

```powershell
python -m desktop.capture --duration 30 --no-upload --output test.wav
```

### List devices (find loopback / mic names)

```powershell
python -m desktop.capture --list-devices
```

### All flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--url` | `""` | Required when uploading. |
| `--duration` | `auto` | `auto` (silence-stop) or seconds. |
| `--silence-threshold-db` | `-50.0` | dBFS below which counts as silence. |
| `--silence-duration-sec` | `120.0` | Silent seconds before stop. |
| `--mode` | `gemini_direct` | One of: `gemini_direct`, `whisper_local`, `whisper_api`, `ivrit_ai`. |
| `--language` | `he` | Hebrew default. |
| `--cookie` | `$ZTT_SESSION` | Fallback when `--cookie` not given. |
| `--output` | `recordings/<UTC>.wav` | Auto-creates the `recordings/` directory. |
| `--no-upload` | off | Record only. |
| `--loopback-device` | (default output) | Substring or index. |
| `--mic-device` | (default input) | Substring or index. |
| `--list-devices` | off | Print devices and exit. |

## Output format

Mono 48 kHz PCM_16 WAV. Roughly 5.7 MB per minute. The server's upload
cap is 600 MB → about 1h44m per recording. For longer recordings,
split or use the `--duration` flag.

## Behaviour notes

- Captures stream to disk only — RAM stays under ~10 MB regardless of
  recording length.
- The silence detector watches **only** the system loopback. If the
  call ends and you keep talking into the mic, recording still stops.
- On upload failure the WAV is preserved at the printed path so you
  can re-upload manually (e.g. via the website's upload tab).
- `Ctrl+C` stops cleanly and flushes the WAV.

## Tests

Pure-logic unit tests (mocks for `sounddevice` + `httpx`):

```powershell
pytest tests/desktop/ -v
```

The hardware-gated test (`test_real_wasapi_loopback_records_2s`)
records 2 seconds of real audio. It runs only on Windows when `CI` is
unset.

## Troubleshooting

- **`PortAudio library not found`**: Python wheel didn't bundle it.
  Use Python 3.11+ on Windows, or install PortAudio via your OS
  package manager on macOS/Linux.
- **`Sample rate not supported`**: Your loopback device is locked at
  44.1 kHz (some pro audio interfaces). Pass
  `--loopback-device <substring>` to pick a 48 kHz device, or open
  Sound Control Panel → device → Advanced and set 48000 Hz.
- **HTTP 401 on upload**: Cookie expired. Refresh from DevTools.
- **HTTP 413 on upload**: WAV exceeded 600 MB. Record shorter
  segments.

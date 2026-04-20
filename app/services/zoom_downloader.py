"""
Zoom Recording Downloader.

Authentication strategy (in order of preference):
  1. Cookie injection — cookies extracted from the user's authenticated browser
     session via the Chrome extension. This is the most reliable method for
     institutional recordings (e.g. admin-ort-org-il.zoom.us).
  2. Direct download — yt-dlp without cookies (works for public/passcode-only links).

Audio is extracted with ffmpeg and saved as mp3 at 96kbps, which is more than
sufficient for speech and keeps file sizes small (~43 MB/hour).
"""
import asyncio
import logging
import os
import tempfile
from pathlib import Path

import yt_dlp

from app.config import settings

logger = logging.getLogger(__name__)


class ZoomDownloadError(Exception):
    """Raised when a download fails — message is shown directly to the user."""


# ── Public interface ──────────────────────────────────────────────────────────────

async def download_audio(
    url: str,
    task_id: str,
    cookies_netscape: str | None = None,
) -> str:
    """
    Download and extract audio from a Zoom recording URL.

    Returns the path to the extracted .mp3 file.
    The caller is responsible for deleting the file after processing.

    cookies_netscape: Cookie string in Netscape/Mozilla format, as provided by
                      the Chrome extension (document.cookie isn't enough — we need
                      the full header-level cookies including HttpOnly ones).
    """
    settings.downloads_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(settings.downloads_dir / f"{task_id}.%(ext)s")
    expected_output  = str(settings.downloads_dir / f"{task_id}.mp3")

    # Write cookies to a temp file — yt-dlp reads them from disk
    cookie_file_path: str | None = None
    if cookies_netscape:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="zoom_cookies_"
        ) as tmp:
            # Ensure the file has the required Netscape header
            if not cookies_netscape.startswith("# Netscape"):
                tmp.write("# Netscape HTTP Cookie File\n")
            tmp.write(cookies_netscape)
            cookie_file_path = tmp.name
        logger.info(f"Cookie file written: {cookie_file_path}")

    ydl_opts: dict = {
        # Prefer audio-only stream; fall back to lowest-quality video + best audio
        # (audio track is identical across all video resolutions, so no quality loss)
        "format": "bestaudio[ext=m4a]/bestaudio/worstvideo+bestaudio/worst",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "96",   # 96 kbps — clear speech, small file
        }],
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "socket_timeout": 90,
        "retries": 3,
        "fragment_retries": 5,
        "progress_hooks": [_make_progress_hook(task_id)],
        # Mimic a real Chrome browser session
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://zoom.us/",
        },
    }

    if cookie_file_path:
        ydl_opts["cookiefile"] = cookie_file_path

    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, _run_ydl, ydl_opts, url),
            timeout=600,  # 10 minutes max for download
        )
    except asyncio.TimeoutError:
        raise ZoomDownloadError("ההורדה לקחה יותר מ-10 דקות ופסקה — הקובץ גדול מדי או שהחיבור איטי")
    except yt_dlp.utils.DownloadError as exc:
        _raise_user_friendly_error(str(exc), bool(cookies_netscape))
    finally:
        # Always clean up the temp cookie file
        if cookie_file_path and Path(cookie_file_path).exists():
            os.unlink(cookie_file_path)

    if not Path(expected_output).exists():
        raise ZoomDownloadError(
            "החילוץ הצליח אך קובץ האודיו לא נמצא — ודא ש-ffmpeg מותקן."
        )

    size_mb = Path(expected_output).stat().st_size / 1024 / 1024
    logger.info(f"Downloaded: {expected_output} ({size_mb:.1f} MB)")
    return expected_output


async def cleanup_audio(file_path: str | None):
    """Safely delete a temp audio file after processing completes."""
    if not file_path:
        return
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as exc:
        logger.warning(f"Could not clean up {file_path}: {exc}")


# ── Internals ─────────────────────────────────────────────────────────────────────

def _make_progress_hook(task_id: str):
    """Log download progress so we can see it in Fly.io logs."""
    def hook(d: dict) -> None:
        if d["status"] == "downloading":
            pct = d.get("_percent_str", "?%").strip()
            speed = d.get("_speed_str", "?").strip()
            eta = d.get("_eta_str", "?").strip()
            logger.info(f"Task {task_id} — download {pct} speed={speed} eta={eta}")
        elif d["status"] == "finished":
            logger.info(f"Task {task_id} — download finished, extracting audio...")
    return hook


def _run_ydl(opts: dict, url: str):
    """Synchronous yt-dlp call — runs inside a thread pool executor."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def _raise_user_friendly_error(raw_error: str, had_cookies: bool) -> None:
    """Convert yt-dlp error strings into helpful messages for the user."""
    err = raw_error.lower()

    if any(k in err for k in ("password", "passcode", "403", "401", "forbidden", "login")):
        if had_cookies:
            raise ZoomDownloadError(
                "האימות נכשל למרות שנשלחו עוגיות סשן. "
                "רענן את דף ההקלטה בדפדפן ולחץ שוב על 'שלח למתמלל' בתוסף."
            )
        raise ZoomDownloadError(
            "ההקלטה דורשת אימות. "
            "פתח את ההקלטה בדפדפן תוך כדי שאתה מחובר ל-Zoom, "
            "ואז השתמש בתוסף Chrome לשלוח אותה לכאן."
        )

    if "404" in err or "not found" in err:
        raise ZoomDownloadError(
            "ההקלטה לא נמצאה (404). הקישור אולי פג תוקפו או נמחק."
        )

    if "private" in err or "unavailable" in err:
        raise ZoomDownloadError(
            "ההקלטה פרטית. השתמש בתוסף Chrome בזמן צפייה בה בדפדפן."
        )

    raise ZoomDownloadError(f"הורדה נכשלה: {raw_error[:300]}")

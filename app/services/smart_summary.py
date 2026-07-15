"""
On-demand Smart Summary — a Map-Reduce pipeline that turns a lesson's transcript
into a polished Obsidian note using the configured LLM (local Ollama OR Gemini).

Map-Reduce:
  • MAP    — split the transcript into ~4000-word chunks; summarize each chunk
             on its own (bounded context, so even a 3-hour lecture fits).
  • REDUCE — merge the partial summaries into one master Obsidian document:
             YAML frontmatter (added deterministically in Python), Markdown
             headings, fenced code blocks, and exactly 3 study flashcards.

The prompts enforce strict Hebrew-RTL / English-LTR discipline so mixed-language
technical lectures render cleanly in Obsidian instead of turning into a
bidirectional mess.

Execution model: a Smart Summary is triggered on an already-completed task and
runs as an in-process background asyncio task (NOT the durable transcription
queue — re-generating is cheap and idempotent). Status lives in the tasks table
(smart_summary_status); a restart resets any in-flight run to 'failed' so the UI
offers a re-run instead of spinning.
"""
import asyncio
import logging
import re
from datetime import date

from app import state
from app.logging_config import task_id_var
from app.models import LessonResult
from app.services.errors import PipelineError
from app.services.llm import get_summary_provider
from app.services.llm.base import LLMError

logger = logging.getLogger(__name__)

WORDS_PER_CHUNK = 4000
_REDUCE_TIMEOUT = 600.0


# ── Prompts ────────────────────────────────────────────────────────────────────

# Shared bidi discipline — the heart of the feature. Injected into both steps.
_BIDI_RULES = """━━━ כללי כיווניות טקסט (RTL/LTR) — קריטי, למניעת ערבוב מבולגן ━━━
1. הנרטיב וההסברים נכתבים בעברית מימין-לשמאל (RTL). כל פסקה עברית עומדת בפני עצמה.
2. מונחים טכניים, שמות טכנולוגיות, פונקציות ופקודות נשארים באנגלית (LTR) בדיוק כפי
   שנאמרו — אין לתרגם ואין לתעתק (React, לא "ריאקט"; useState, לא "השתמש-במצב").
3. כל קטע קוד באורך שורה או יותר עטוף ב-fenced code block עם שם השפה
   (```python, ```bash, ```sql) על שורות נפרדות משלו — לעולם לא בתוך פסקה עברית.
4. בתוך שורת קוד אין לערבב עברית — הקוד הוא LTR נקי. ההסבר בעברית בא בשורה נפרדת.
5. מונח אנגלי קצר בתוך משפט עברי מותר כ-inline code עם backticks (למשל `useState`)."""

_MAP_SYSTEM = f"""אתה מסכם קטע מתוך תמלול שיעור טכני המערבב עברית ואנגלית.
סכם אך ורק את הקטע שלפניך — נקודות המפתח, ההגדרות וכל קוד שמופיע בו.
שמור סימוני זמן [MM:SS] אם קיימים. אל תוסיף כותרת ראשית.

{_BIDI_RULES}

פלט: תקציר קצר ורשימת נקודות מפתח של הקטע בלבד, ב-Markdown."""

_REDUCE_SYSTEM = f"""אתה עורך לימודי המפיק סיכום-על מובנה לשיעור, מוכן להדבקה ב-Obsidian.
קיבלת סיכומי ביניים (או תמלול) של שיעור טכני המערבב עברית ואנגלית.

{_BIDI_RULES}

━━━ מבנה הפלט (Markdown בלבד — ללא YAML frontmatter, הוא יתווסף אוטומטית) ━━━
- השורה הראשונה בדיוק: `tags: <2-4 תגיות נושא, מילה אחת כל אחת, מופרדות בפסיקים>`
- `## 📝 סיכום` — 2-4 פסקאות נרטיב בעברית שמכסות את כל השיעור.
- `## 🗂️ נושאים עיקריים` — לכל נושא תת-כותרת `### <נושא>` עם הסבר ונקודות מפתח.
  שלב קטעי קוד רלוונטיים בתוך fenced code blocks עם השפה הנכונה.
- `## 🎴 כרטיסיות לחזרה` — בדיוק 3 כרטיסיות לימוד, כל אחת בפורמט:
  **שאלה:** <שאלה ממוקדת>
  **תשובה:** <תשובה של 1-2 משפטים>
  (שורה ריקה בין כרטיסיות)

הפק אך ורק את ה-Markdown של הגוף — בלי טקסט עוטף ובלי ```markdown חיצוני."""


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_by_words(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    """Split text into chunks of at most `words_per_chunk` whitespace-delimited
    words. Works for Hebrew and English alike (both are space-separated)."""
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def _source_text(result: LessonResult) -> str:
    """The best available text to summarize.

    Whisper-mode tasks carry a full transcript. GEMINI_DIRECT tasks don't, so
    fall back to the generated lesson (summary + chapters) — Smart Summary then
    still produces a tidy Obsidian note from what we have.
    """
    if result.transcript and result.transcript.strip():
        return result.transcript
    parts: list[str] = []
    if result.summary:
        parts.append(result.summary)
    for ch in result.chapters:
        parts.append(f"{ch.title}\n{ch.content}")
        parts.extend(f"- {kp}" for kp in ch.key_points)
    return "\n\n".join(parts)


# ── Map-Reduce ─────────────────────────────────────────────────────────────────

async def _map_chunk(chunk: str, provider, index: int, total: int) -> str:
    prompt = f"סכם את הקטע הבא (חלק {index} מתוך {total}):\n\n{chunk}"
    return await provider.complete(prompt, system=_MAP_SYSTEM, temperature=0.2)


async def _reduce(segments: list[str], provider) -> str:
    joined = "\n\n---\n\n".join(s for s in segments if s.strip())
    prompt = (
        "להלן חומר הגלם של השיעור (סיכומי ביניים או תמלול). "
        "הפק ממנו את סיכום-העל המובנה כפי שהוגדר:\n\n" + joined
    )
    return await provider.complete(
        prompt, system=_REDUCE_SYSTEM, temperature=0.3, timeout=_REDUCE_TIMEOUT
    )


async def generate_smart_summary(
    source_text: str,
    provider,
    *,
    title: str,
    source: str = "",
    words_per_chunk: int = WORDS_PER_CHUNK,
) -> str:
    """Run Map-Reduce and return a complete Obsidian markdown document."""
    chunks = chunk_by_words(source_text, words_per_chunk)
    if not chunks:
        raise PipelineError("אין טקסט לסיכום")

    if len(chunks) == 1:
        # Short enough to reduce directly — the single chunk IS the material.
        body = await _reduce(chunks, provider)
    else:
        partials: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            logger.info("Smart Summary MAP %d/%d", i, len(chunks))
            partials.append(await _map_chunk(chunk, provider, i, len(chunks)))
        body = await _reduce(partials, provider)

    return _build_obsidian_doc(body, title=title, source=source, backend=provider.name)


# ── Obsidian document assembly ─────────────────────────────────────────────────

def _yaml_escape(value: str) -> str:
    return value.replace('"', "'").replace("\n", " ").strip()


def _slugify_tag(tag: str) -> str:
    # Obsidian tags can't contain spaces; keep Hebrew + latin word chars + hyphen.
    cleaned = re.sub(r"[^\w֐-׿-]+", "_", tag.strip(), flags=re.UNICODE)
    return cleaned.strip("_")


def _unwrap_outer_fence(body: str) -> str:
    """Strip a stray ```markdown … ``` fence some models wrap the whole doc in."""
    stripped = body.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]                       # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]                  # drop closing fence
        return "\n".join(lines).strip()
    return stripped


def _extract_tags(body: str) -> tuple[list[str], str]:
    """Pull an optional leading `tags: a, b, c` line into structured tags and
    return (tags, body_without_that_line). Absent → ([], body)."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        match = re.match(r"(?i)^tags?\s*:\s*(.+)$", line.strip())
        if match:
            raw = re.split(r"[,،;]", match.group(1))
            tags = [t for t in (_slugify_tag(r) for r in raw) if t][:6]
            rest = "\n".join(lines[:i] + lines[i + 1:]).strip()
            return tags, rest
        break  # first content line isn't a tags line → no tags to extract
    return [], body


def _build_obsidian_doc(body: str, *, title: str, source: str, backend: str) -> str:
    body = _unwrap_outer_fence(body)
    extra_tags, body = _extract_tags(body)

    tags = ["שיעור", "zoom-to-text", *extra_tags]
    front = ["---", f'title: "{_yaml_escape(title)}"', f"date: {date.today().isoformat()}"]
    if source:
        front.append(f'source: "{_yaml_escape(source)}"')
    front.append("tags:")
    front.extend(f"  - {t}" for t in tags)
    front.append(f"generated_by: {backend}")
    front.append("---")

    return "\n".join(front) + f"\n\n# {title}\n\n" + body.strip() + "\n"


# ── Background job runner ──────────────────────────────────────────────────────

# Keep strong references so fire-and-forget tasks aren't garbage-collected.
_jobs: set[asyncio.Task] = set()
# Serialize generations (one Ollama call at a time). Recreated if the running
# loop changes — asyncio primitives are bound to a loop, and the test suite
# spins up a fresh loop per app instance.
_run_lock: asyncio.Lock | None = None
_run_lock_loop: object | None = None


def _get_lock() -> asyncio.Lock:
    global _run_lock, _run_lock_loop
    loop = asyncio.get_running_loop()
    if _run_lock is None or _run_lock_loop is not loop:
        _run_lock = asyncio.Lock()
        _run_lock_loop = loop
    return _run_lock


def enqueue(task_id: str) -> None:
    """Launch the Smart Summary background job for a task."""
    job = asyncio.create_task(_run_job(task_id), name=f"smart-summary-{task_id}")
    _jobs.add(job)
    job.add_done_callback(_jobs.discard)


def _title_from_task(task) -> str:
    raw = (task.url or "").replace("upload:", "").strip()
    raw = re.sub(r"^https?://", "", raw)
    raw = re.sub(r"\.[a-z0-9]{2,5}$", "", raw, flags=re.IGNORECASE)
    return raw[:80].strip() or f"סיכום שיעור {date.today().isoformat()}"


async def _run_job(task_id: str) -> None:
    ctx = task_id_var.set(task_id)
    try:
        async with _get_lock():
            await state.set_smart_summary_status(task_id, "running")
            task = await state.get_task(task_id)
            if task is None or task.result is None:
                await state.set_smart_summary_status(task_id, "failed", "אין תוצאה למשימה")
                return
            source = _source_text(task.result)
            if not source.strip():
                await state.set_smart_summary_status(
                    task_id, "failed", "אין טקסט לסיכום עבור משימה זו"
                )
                return

            provider = await get_summary_provider()
            markdown = await generate_smart_summary(
                source, provider,
                title=_title_from_task(task),
                source=(task.url or "").replace("upload:", ""),
            )
            await state.save_smart_summary(task_id, markdown)
            logger.info("Smart Summary completed for %s (%d chars)", task_id, len(markdown))
    except (LLMError, PipelineError) as exc:
        message = getattr(exc, "user_message", None) or str(exc)
        await state.set_smart_summary_status(task_id, "failed", message)
        logger.warning("Smart Summary failed for %s: %s", task_id, message)
    except Exception:
        await state.set_smart_summary_status(task_id, "failed", "שגיאה ביצירת הסיכום החכם")
        logger.exception("Smart Summary crashed for %s", task_id)
    finally:
        task_id_var.reset(ctx)

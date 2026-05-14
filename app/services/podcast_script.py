"""Generate a two-host conversational podcast script from a lesson result.

B6.4 ships the *script* only — TTS is deferred. The output is a list of
dialogue turns (`host_a` / `host_b`) suitable for any TTS pipeline (ElevenLabs,
Polly, Piper) or to be read aloud by the user.

We use the active LLM provider for generation. If the LLM call fails or the
response can't be parsed, we fall back to a deterministic local script built
from the summary + chapters so the endpoint never 500s on a downstream wobble.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from app.models import LessonResult
from app.services.llm_providers import get_provider

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 60.0
_MIN_TURNS = 4
_MAX_TURNS = 16


_PROMPT_TEMPLATE = """אתה כותב שני פודקסטרים בעברית: מארח A (סקרן, שואל שאלות) ומארח B (מבין, מסביר). מטרת השיחה: לסכם את ההרצאה הבאה בצורה נעימה ושוחה לאוזן, כשני אנשים שמדברים.

הנחיות:
- בין 6 ל-{max_turns} דברי שיחה (תורות).
- כל תור הוא שורה אחת קצרה. אל תכתוב פסקאות.
- מארח A פותח עם שאלה או מסגרת; מארח B מסביר; הם מתחלפים.
- אל תוסיף סיומת תרגום, פתיח או הסבר על עצמך.
- החזר JSON תקין בלבד (ללא ```), במבנה הבא:
{{"turns": [{{"speaker": "host_a", "text": "..."}}, {{"speaker": "host_b", "text": "..."}}, ...]}}

תקציר ההרצאה:
{summary}

פרקים:
{chapters}
"""


def _build_prompt(result: LessonResult) -> str:
    summary = (result.summary or "").strip()[:4000]
    lines: list[str] = []
    for idx, ch in enumerate(result.chapters or [], start=1):
        lines.append(f"{idx}. {ch.title}: {(ch.content or '')[:400]}")
    chapters_text = "\n".join(lines)[:4000] or "(אין פרקים)"
    return _PROMPT_TEMPLATE.format(
        summary=summary or "(אין תקציר)",
        chapters=chapters_text,
        max_turns=_MAX_TURNS,
    )


def _strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _normalize_turns(raw_turns: list[dict]) -> list[dict]:
    out: list[dict] = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip().lower()
        text = str(turn.get("text", "")).strip()
        # Tolerate alt labels: "a" / "A" / "host a"
        if speaker in {"a", "host a", "host-a"}:
            speaker = "host_a"
        elif speaker in {"b", "host b", "host-b"}:
            speaker = "host_b"
        if speaker not in {"host_a", "host_b"} or not text:
            continue
        out.append({"speaker": speaker, "text": text[:800]})
        if len(out) >= _MAX_TURNS:
            break
    return out


def _fallback_script(result: LessonResult) -> list[dict]:
    """Deterministic local fallback when the LLM is unavailable.

    Strictly alternates host_a / host_b lines built from the summary and
    chapter titles + first sentence of each chapter.
    """
    turns: list[dict] = []
    summary = (result.summary or "").strip()
    if summary:
        turns.append({"speaker": "host_a", "text": "בוא נדבר על ההרצאה הזאת — מה התובנה המרכזית?"})
        turns.append({"speaker": "host_b", "text": summary[:500]})
    for idx, ch in enumerate(result.chapters or [], start=1):
        speaker = "host_a" if idx % 2 == 1 else "host_b"
        snippet = (ch.content or "").strip().split(".")[0][:400] or ch.title
        turns.append({"speaker": speaker, "text": f"בפרק \"{ch.title}\": {snippet}"})
        if len(turns) >= _MIN_TURNS + 4:
            break
    if not turns:
        turns = [
            {"speaker": "host_a", "text": "אין מספיק תוכן בהרצאה כדי לבנות שיחה."},
            {"speaker": "host_b", "text": "כדאי להריץ עיבוד מחדש או לעלות קובץ ארוך יותר."},
        ]
    return turns[:_MAX_TURNS]


async def build_podcast_script(result: LessonResult) -> dict:
    """Return `{"turns": [...], "model": "<provider_name>|fallback"}`.

    Never raises — on any LLM failure we fall back to the deterministic
    `_fallback_script`. The endpoint surfaces this through `model="fallback"`
    so the UI can show a "best effort" hint if it wants.
    """
    provider = get_provider()
    prompt = _build_prompt(result)
    try:
        raw = await asyncio.wait_for(
            provider.generate_text(prompt, temperature=0.5),
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(f"Podcast LLM call failed, falling back: {exc}")
        return {"turns": _fallback_script(result), "model": "fallback"}

    stripped = _strip_code_fences(raw)
    if not stripped:
        return {"turns": _fallback_script(result), "model": "fallback"}
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return {"turns": _fallback_script(result), "model": "fallback"}

    turns = _normalize_turns(data.get("turns") or [])
    if len(turns) < _MIN_TURNS:
        return {"turns": _fallback_script(result), "model": "fallback"}
    return {"turns": turns, "model": provider.name}

"""Cross-lecture glossary builder (B4).

Given a user's recent completed lectures, asks the active LLM provider to
extract a single deduped glossary of technical terms with concise Hebrew
definitions, each annotated with which lectures (task ids) it was found in.

The result is cached per user in ``user_glossaries`` so the home page can
load it without paying for an LLM call every time.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable

from app import state
from app.models import TaskResponse
from app.services.llm_providers import get_provider

logger = logging.getLogger(__name__)

# Cap the number of tasks we send into the prompt — Gemini's context window
# is wide but we don't need every lecture, and tokens cost money.
_MAX_TASKS = 30
_MAX_TERMS = 60
_TIMEOUT = 90.0


_SYSTEM_PROMPT = """\
אתה בונה רשימת מונחים (glossary) חוצת-שיעורים עבור תלמיד.
קלט: כותרות וסיכומים קצרים של מספר שיעורים, כל אחד עם מזהה ייחודי.
מטרה: לחלץ עד 60 מונחים מקצועיים חשובים שחוזרים על עצמם או כדאי לזכור,
ולחבר לכל אחד הגדרה קצרה בעברית.

עקרונות:
- הגדרות קצרות (משפט אחד, עד 25 מילים).
- מונחים בעלי משמעות מקצועית בלבד — לא שמות פרטיים, לא ביטויי לשון.
- אם אותו מונח מופיע בכמה שיעורים — הוסף את כל מזהי השיעורים ב-sources.
- מיין מהמונח המופיע הכי הרבה פעמים לפחות.

החזר JSON תקין בלבד, ללא טקסט מסביב, במבנה הבא:
{
  "terms": [
    {"term": "...", "definition": "...", "sources": ["task_id_1", "task_id_2"]}
  ]
}
"""


def _format_tasks_for_prompt(tasks: Iterable[TaskResponse]) -> str:
    parts: list[str] = []
    for t in tasks:
        if t.result is None or not t.result.summary:
            continue
        title = ""
        if t.result.chapters:
            title = t.result.chapters[0].title or ""
        parts.append(
            f"=== {t.task_id} ===\n"
            f"כותרת: {title}\n"
            f"סיכום: {t.result.summary[:600]}"
        )
    return "\n\n".join(parts)


def _extract_json(text: str) -> dict:
    """Strip ```json fences and parse — gracefully fall back on raw text."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _coerce_terms(payload: dict, *, valid_task_ids: set[str]) -> list[dict]:
    """Filter LLM output: drop malformed entries; clamp to ``_MAX_TERMS``."""
    raw_terms = payload.get("terms") or []
    out: list[dict] = []
    for entry in raw_terms:
        if not isinstance(entry, dict):
            continue
        term = (entry.get("term") or "").strip()
        definition = (entry.get("definition") or "").strip()
        if not term or not definition:
            continue
        sources = entry.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        sources = [s for s in sources if isinstance(s, str) and s in valid_task_ids]
        out.append({
            "term": term[:120],
            "definition": definition[:400],
            "sources": sources[:10],
        })
        if len(out) >= _MAX_TERMS:
            break
    return out


async def build_glossary_for_user(user_id: str) -> dict:
    """Build (and cache) the glossary for ``user_id``.

    Returns ``{"terms": [...], "updated_at": "...", "task_count": N}``. If the
    user has no completed lectures the function returns an empty payload and
    writes that to the cache so we don't keep re-trying.
    """
    tasks = await state.list_completed_tasks_with_results(user_id=user_id, limit=_MAX_TASKS)
    if not tasks:
        updated_at = await state.set_user_glossary(user_id, [])
        return {"terms": [], "updated_at": updated_at, "task_count": 0}

    valid_task_ids = {t.task_id for t in tasks}
    prompt = (
        _SYSTEM_PROMPT
        + "\n\nשיעורים:\n"
        + _format_tasks_for_prompt(tasks)
    )

    provider = get_provider()
    try:
        text = await provider.generate_text(prompt, timeout=_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — bubble up but log so the route can map to 502
        logger.warning("glossary: provider failed for %s: %s", user_id, exc)
        raise

    try:
        payload = _extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("glossary: bad JSON from provider: %s", exc)
        payload = {"terms": []}

    terms = _coerce_terms(payload, valid_task_ids=valid_task_ids)
    updated_at = await state.set_user_glossary(user_id, terms)
    return {"terms": terms, "updated_at": updated_at, "task_count": len(tasks)}

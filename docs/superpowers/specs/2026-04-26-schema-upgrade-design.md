# Schema Upgrade — Design Spec

**Date:** 2026-04-26
**Cycle:** B (builds on the LLM Provider Abstraction shipped in Cycle A)
**Status:** Pending user approval
**Source plan:** `UPGRADE_PROMPT.md` Task 1.1

---

## 1. Goal

Extend the JSON schema returned by the LLM beyond the existing
`(summary, chapters, quiz, flashcards)` to capture business-meeting and
academic-discussion artifacts:

- `action_items` — `{owner, task, deadline, priority, source_quote}[]`
- `decisions` — `{decision, context, stakeholders, source_quote}[]`
- `open_questions` — `{question, raised_by, context}[]`
- `sentiment_analysis` — `{overall_tone, per_speaker_sentiment[], shifts_in_tone[]}`
- `objections_tracked` — `{objection, raised_by, response_given, resolved}[]`

All five fields are **optional** in the response model — old recordings
keep working, and partial provider responses don't crash the pipeline.

A new field `content_type` (`lecture | meeting | discussion`) is also
auto-detected by the LLM and used to **modulate the prompt's emphasis**
in subsequent stages of the pipeline (action items dominate in
meetings; key claims dominate in lectures).

The persistence pipeline gains a debug field `raw_llm_response`
(structured per call) so failures and prompt-quality issues can be
reproduced offline.

## 2. Non-goals

Out of scope for this spec:

- **No UI changes.** Fields are produced and persisted; the existing
  results page in `static/index.html` ignores them. UI work is a
  separate task (`/frontend-design` later).
- **Quiz/flashcards/critique pipeline unchanged.** The
  `critique_exam` / `revise_exam` flow keeps its current behavior.
- **No diarization** (that's Task 1.2).
- **No new LLM provider work.** Uses the existing `LLMProvider`
  abstraction shipped in Cycle A.
- **No schema migration in SQLite.** `result_json` is already a JSON
  blob — new fields appear in newly-completed tasks, old tasks remain
  readable and render with the new fields absent.

## 3. Architecture

### 3.1 Two-call parallel strategy

Today, summarization is a single LLM call returning
`(summary, chapters, quiz)` at `temperature=0.3`. This spec splits
it into two calls run concurrently via `asyncio.gather`:

```
                ┌────────────────────────────────────────┐
                │ Call 1 — Synthesis (existing)          │
                │   temperature = 0.3                    │
                │   returns: summary, chapters, quiz,    │
                │            content_type                │
                └────────────────────────────────────────┘
   audio/text  →
                ┌────────────────────────────────────────┐
                │ Call 2 — Extraction (new)              │
                │   temperature = 0.2                    │
                │   returns: action_items, decisions,    │
                │            open_questions,             │
                │            sentiment_analysis,         │
                │            objections_tracked          │
                └────────────────────────────────────────┘
```

Both calls receive the same input (audio file or transcript) and
the same `content_type` hint via prompt prefix. Wall-clock latency is
`max(call_1, call_2)` ≈ same as today; cost ≈ +30% Gemini tokens.

`content_type` is detected **inside Call 1** (the synthesis call already
reads the entire input) and **passed in the prompt of Call 2 only when
running serially**; in the parallel path, both calls use a self-contained
detection step in their own prompt. This keeps the calls independent.

### 3.2 Module responsibilities

```
app/models.py
    LessonResult — gains 6 optional fields (5 extraction + content_type)
    + 5 new sub-models (ActionItem, Decision, OpenQuestion,
                        SentimentAnalysis, Objection)

app/services/summarizer.py
    summarize_audio()       — orchestrates the parallel pair
    summarize_transcript()  — orchestrates the parallel pair
    _extract_artifacts_*()  — new helpers for Call 2 (audio + text)
    _merge_results()        — combines Call 1 + Call 2 into LessonResult
                              with raw_llm_response populated

app/services/prompts/  (new package, optional refactor)
    synthesis.py            — Call 1 prompt (Hebrew + English few-shots)
    extraction.py           — Call 2 prompt (Hebrew + English few-shots)
```

Both calls go through the existing `LLMProvider` ABC — no new provider
work. The Gemini provider's `_with_retry` already handles 429/5xx
transient failures.

## 4. Pydantic schema changes (`app/models.py`)

```python
class ActionItem(BaseModel):
    owner:        str
    task:         str
    deadline:     Optional[str] = None    # free-form: "EOW", "2026-05-01", etc.
    priority:     Optional[str] = None    # "high" | "medium" | "low"
    source_quote: Optional[str] = None    # transcribed sentence that triggered this

class Decision(BaseModel):
    decision:     str
    context:      Optional[str] = None
    stakeholders: list[str] = []
    source_quote: Optional[str] = None

class OpenQuestion(BaseModel):
    question:  str
    raised_by: Optional[str] = None
    context:   Optional[str] = None

class PerSpeakerSentiment(BaseModel):
    speaker:   str                         # "Speaker A" or named if known
    sentiment: str                         # "positive" | "neutral" | "negative" | "mixed"
    rationale: Optional[str] = None

class ToneShift(BaseModel):
    at:          str                       # rough timestamp or paragraph anchor
    from_tone:   str = Field(..., alias="from")
    to_tone:     str = Field(..., alias="to")
    trigger:     Optional[str] = None

class SentimentAnalysis(BaseModel):
    overall_tone:          str
    per_speaker_sentiment: list[PerSpeakerSentiment] = []
    shifts_in_tone:        list[ToneShift] = []

class Objection(BaseModel):
    objection:       str
    raised_by:       Optional[str] = None
    response_given:  Optional[str] = None
    resolved:        Optional[bool] = None

class RawLLMResponse(BaseModel):
    summary_call:    Optional[str] = None  # raw text returned by Call 1
    extraction_call: Optional[str] = None  # raw text returned by Call 2

class LessonResult(BaseModel):
    # ── existing fields unchanged ──
    transcript:        Optional[str] = None
    summary:           str = ""
    chapters:          list[Chapter] = []
    quiz:              list[QuizQuestion] = []
    flashcards:        list[Flashcard] = []
    language:          str = "he"
    exam_critique_log: Optional[dict] = None
    # ── new optional fields ──
    content_type:        Optional[str] = None  # "lecture" | "meeting" | "discussion"
    action_items:        list[ActionItem] = []
    decisions:           list[Decision] = []
    open_questions:      list[OpenQuestion] = []
    sentiment_analysis:  Optional[SentimentAnalysis] = None
    objections_tracked:  list[Objection] = []
    raw_llm_response:    Optional[RawLLMResponse] = None
```

Defaults are empty list / `None` so existing JSON blobs in SQLite
deserialize cleanly without migration.

## 5. Prompt design

### 5.1 Synthesis prompt (Call 1)

The existing prompt stays mostly intact. A new instruction asks the
model to start its JSON with a `content_type` discriminator:

> Begin with `"content_type"` set to one of: `"lecture"` (formal
> academic delivery, one primary speaker), `"meeting"` (decision-making
> with multiple participants), or `"discussion"` (open-ended dialogue).

No few-shot examples added here — the existing prompt already includes
Hebrew + English demonstrations.

### 5.2 Extraction prompt (Call 2)

New prompt with **two few-shot examples** — one Hebrew meeting, one
English lecture — demonstrating empty arrays when a category doesn't
apply:

```
You are extracting structured artifacts from a recording transcript or
audio file. Return STRICT JSON with the following keys, ALL optional
(use empty arrays / null when absent):

  - action_items, decisions, open_questions,
    sentiment_analysis, objections_tracked

Rules:
  - Every string MUST be in the source language (Hebrew → Hebrew).
  - source_quote MUST be a verbatim short snippet (≤140 chars) when present.
  - DO NOT invent owners, deadlines, or stakeholders. If unclear, omit
    the field for that item.
  - For lectures, action_items and objections_tracked typically stay [].
  - For meetings, sentiment_analysis is usually populated.

[Few-shot example 1 — Hebrew product meeting]
[Few-shot example 2 — English physics lecture]

Now extract from the recording above.
```

`temperature=0.2` per the source plan — factual extraction, low
creativity.

## 6. Error handling

Per Q4 brainstorming decision **(B)**:

1. Each call goes through `LLMProvider._with_retry` (already handles
   429, 5xx, network errors with exponential backoff — existing
   behavior).
2. If the **synthesis call** fails after retries → propagates as
   today: task transitions to `FAILED` with the error message stored.
3. If the **extraction call** fails after retries → graceful skip:
   - `action_items`, `decisions`, `open_questions`,
     `objections_tracked` remain `[]`.
   - `sentiment_analysis` remains `None`.
   - `raw_llm_response.extraction_call` remains `None`.
   - A WARNING-level log entry is written with the exception and
     task_id.
   - The task still completes successfully — user sees summary +
     chapters + quiz, just no business artifacts.

This satisfies the spec's success criterion "if Gemini doesn't return
a new field — the system doesn't crash."

## 7. `raw_llm_response` persistence

Per Q5 brainstorming decision **(A + flag)**:

- `LessonResult.raw_llm_response` is a structured object with
  `summary_call` and `extraction_call` keys (both optional).
- Populated **only when** `LLM_DEBUG_RAW_RESPONSES=true` in `.env`
  (default: `false` in production to save SQLite space).
- When the flag is off, `raw_llm_response` is `None` — same shape as
  for old completed tasks.
- New `app/config.py` field:
  ```python
  llm_debug_raw_responses: bool = Field(False, env="LLM_DEBUG_RAW_RESPONSES")
  ```

This is **debug instrumentation only** — never returned in the public
`/api/tasks/{id}` response (filtered out at the API layer to avoid
leaking model output to clients).

## 8. Backward compatibility

| Concern                                 | Behavior                                                  |
|-----------------------------------------|-----------------------------------------------------------|
| Old SQLite tasks without new fields     | Deserialize cleanly — defaults fill missing keys.        |
| Frontend (`static/index.html`)           | Ignores new fields — no JS changes needed in this spec.  |
| `/api/tasks/{id}` response shape         | Same keys as today PLUS the new optional ones.           |
| Quiz / flashcards / critique pipeline    | Unchanged — Call 1 still produces them.                  |
| `LLM_PROVIDER=openrouter` / `ollama`     | Same two-call orchestration; provider answers each call. |
| Audio-direct mode (`GEMINI_DIRECT`)      | Both calls upload to Gemini Files API once and reference. |
| Whisper modes                            | Both calls receive the transcript text.                  |
| `summarize_audio()` / `summarize_transcript()` public signatures | Unchanged. Internals split into helpers. |

## 9. Configuration additions (`.env.example`)

```bash
# Schema upgrade (Task 1.1)
# Persist raw LLM responses for debugging. Disable in production
# to save SQLite space — raw responses can be large.
LLM_DEBUG_RAW_RESPONSES=false
```

## 10. Test plan

Unit tests (`tests/test_summarizer_schema_upgrade.py` — new):

1. `test_lesson_result_loads_old_json` — feed a JSON blob produced by
   the current summarizer; verify `LessonResult` parses without errors
   and new fields default to empty.
2. `test_extraction_call_failure_does_not_crash` — mock provider where
   Call 2 raises after retries; verify task completes with summary +
   `action_items=[]`.
3. `test_raw_response_disabled_by_default` — flag off → field is `None`.
4. `test_raw_response_enabled_persists_both_calls` — flag on → both
   keys populated.
5. `test_content_type_propagates_to_lesson_result` — synthesis call
   returns `content_type="meeting"` → field appears in `LessonResult`.
6. `test_extraction_prompt_temperature_is_02` — assert provider call
   was made with `temperature=0.2`.

Integration test (`tests/test_pipeline_real.py` — extend existing):

7. Run a 5-minute real recording end-to-end with stub provider; assert
   the resulting `LessonResult.action_items` is a list and JSON
   round-trips through SQLite.

Manual verification (per spec §3 of `UPGRADE_PROMPT.md`):

- Run on a real meeting recording (Asaf has business calls in
  history) → verify `action_items` and `decisions` are non-empty.
- Run on an existing academic lecture → verify `action_items=[]`,
  summary still good.

## 11. Rollout

1. Land schema in `models.py` (no behavior change yet — fields default
   empty).
2. Land prompts + Call 2 plumbing behind `LLM_DEBUG_RAW_RESPONSES=true`
   in `.env` first; observe two real recordings.
3. Flip to default behavior, leave the flag off.
4. Update `CLAUDE.md` Architecture section to mention the two-call
   pipeline (Task 3.3 of `UPGRADE_PROMPT.md`).

## 12. Risks & open questions

- **Cost.** +30% Gemini tokens per recording. At ~$0.003/recording
  today, this is ~$0.001 increase — negligible at current volume.
- **Prompt drift between calls.** Two independent prompts can disagree
  on `content_type`. Mitigation: only Call 1's `content_type` reaches
  `LessonResult`; Call 2 uses its own copy internally for emphasis.
- **Audio-direct mode (`GEMINI_DIRECT`).** Both calls re-upload the
  audio file. Optimization (out of scope): use Gemini Files API's
  reusable file handles to upload once and reference twice.
- **Few-shot example length.** Hebrew + English meeting examples can
  add ~3k tokens to every Call 2. Acceptable today; revisit if cost
  becomes meaningful.

## 13. Acceptance criteria (from `UPGRADE_PROMPT.md` Task 1.1)

- [x] Old recording without new fields plays cleanly through History.
- [x] Quiz unchanged.
- [x] If LLM doesn't return a new field — the system doesn't crash.
- [ ] Tested on two real recordings (one lecture, one meeting).

The first three are guaranteed by design; the fourth is the manual
acceptance gate before merging.

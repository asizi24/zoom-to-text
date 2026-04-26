# Gemini-based Text Diarization — Design Spec

**Date:** 2026-04-26
**Cycle:** B (continuation of the schema upgrade)
**Status:** Pending user approval
**Source plan:** `UPGRADE_PROMPT.md` Task 1.2

---

## 1. Goal

Add **textual** speaker diarization to the WHISPER pipelines by sending
the raw transcript to Gemini with a dedicated prompt that:

1. Identifies how many speakers are present.
2. Tags each utterance/paragraph with `Speaker A`, `Speaker B`, …
3. Proposes named labels (`אסף`, `Dan`, etc.) when the transcript
   contains explicit semantic cues (introductions, role mentions,
   addressed names).

The diarized transcript becomes available to the downstream synthesis +
extraction calls (so `sentiment_analysis.per_speaker_sentiment` and
`action_items.owner` produce more accurate output) AND is persisted on
the `LessonResult` so the UI can later display speaker labels next to
each paragraph.

This task explicitly **does NOT** use Pyannote, audio embeddings, or
biometric speaker separation — those are reserved for the future home
server (Task 2.2). Today we run on a 512MB Fly.io machine; only
text-based diarization is in scope.

## 2. Non-goals

- **Audio diarization** (Pyannote / WhisperX) — out of scope.
- **GEMINI_DIRECT mode** — already gets implicit speaker awareness from
  the audio model. No changes to the audio path.
- **UI changes** — the new fields are persisted; rendering them is a
  separate task.
- **Per-speaker statistics** beyond what `sentiment_analysis` already
  produces.
- **Provider-agnostic diarization.** Today only Gemini is wired up; for
  OpenRouter/Ollama we skip diarization (LessonResult fields stay null)
  and rely on the existing transcript flow.

## 3. Architecture

### 3.1 Where diarization runs

Inside `summarize_transcript` — as a serial step **before** the parallel
synthesis + extraction pair:

```
        WHISPER_LOCAL / WHISPER_API path:

        transcribe()
           ↓
        ┌──────────────────────────────────────┐
        │  summarize_transcript(text)          │
        │                                      │
        │   diarize(text)  ─→ (text', map)     │  ← NEW (this spec)
        │                                      │
        │   asyncio.gather(                    │
        │     synthesize(text'),               │
        │     extract(text'),                  │
        │   )                                  │
        │                                      │
        │   merge → LessonResult               │
        └──────────────────────────────────────┘
```

The diarized transcript is what synthesis + extraction receive. Both
benefit: synthesis chapters can reference speakers,
`per_speaker_sentiment` becomes much more accurate, and
`action_items.owner` can read directly from the labels.

### 3.2 Why not in `processor.py`?

- The spec explicitly calls diarization "a new pre-processing step in
  the pipeline" alongside summarization, not transcription.
- Keeping all LLM logic inside `summarizer.py` mirrors how Task 1.1
  was structured and keeps `processor.py` ASR-vs-LLM clean.
- `LessonResult.diarized_transcript` is naturally produced inside the
  summarizer, which is what writes `LessonResult` today.

### 3.3 Where it does NOT run

- **`summarize_audio` (GEMINI_DIRECT)** — audio mode already passes
  raw audio to Gemini; the model "hears" speaker turns. Adding a
  separate text diarization step would require us to first transcribe
  the audio, defeating the purpose of GEMINI_DIRECT.
- **Non-Gemini providers** — OpenRouter and Ollama paths skip
  diarization for now (open follow-up: provider-agnostic diarization
  in a later cycle).
- **`ENABLE_DIARIZATION=false`** — operator opt-out.

## 4. Schema additions (`app/models.py`)

```python
class LessonResult(BaseModel):
    # ... existing + Task 1.1 fields ...
    diarized_transcript: Optional[str]            = None
    speaker_map:         Optional[dict[str, str]] = None
```

`diarized_transcript` — full transcript with each paragraph prefixed by
its speaker tag, e.g.:

```
Speaker A: שלום לכולם, אני אסף מהצוות...
Speaker B: היי, דן כאן...
Speaker A: דן, אתה לוקח את החלק של ה-API?
```

`speaker_map` — optional `{"Speaker A": "אסף", "Speaker B": "דן"}`
populated only when the transcript contained clear self-identifications
or addressed names. When ambiguous, the entry is omitted (the UI shows
the anonymous tag).

Both fields are **optional** so old SQLite blobs deserialize unchanged.

## 5. Configuration (`app/config.py`, `.env.example`)

```python
# Enable Gemini-based text diarization for transcripts (WHISPER modes).
# When True, summarize_transcript runs an extra Gemini call to label
# speakers before synthesis + extraction.
enable_diarization: bool = True
```

```bash
# Diarization (Task 1.2) — text-based, Gemini-only
ENABLE_DIARIZATION=true
```

Default `True` per the source plan.

## 6. Prompt design

### 6.1 Single-call structured output

One Gemini call returns both the annotated transcript and the speaker
map as a single JSON object:

```
{
  "speaker_count": 2,
  "speaker_map": {"Speaker A": "אסף", "Speaker B": "דן"},
  "diarized_transcript": "Speaker A: ...\nSpeaker B: ...\n..."
}
```

`speaker_map` may be `{}` when no explicit names were detected.
`speaker_count` is informational; the source of truth for the count is
the keys of `speaker_map` plus any `Speaker X` tags in
`diarized_transcript`.

`temperature = 0.1` (lower than extraction) — diarization is mechanical
turn-detection, not creative.

### 6.2 Few-shot examples

Two examples in the prompt:

1. **Hebrew product meeting** — three speakers, two names detected from
   self-introduction.
2. **English physics lecture** — one speaker (the lecturer); occasional
   student questions tagged as `Speaker B`; no name map.

## 7. Error handling

Per the same pattern as Task 1.1 extraction:

1. The diarization call goes through `_generate_with_retry` (handles
   429/5xx).
2. If it still fails after retries → graceful skip:
   - WARNING-level log entry with task_id and the exception.
   - `diarized_transcript` and `speaker_map` remain `None`.
   - Synthesis + extraction receive the **raw transcript** as
     fallback — pipeline still completes.
3. JSON parse errors are treated identically to retry exhaustion.

## 8. Backward compatibility

| Concern                                  | Behavior                                            |
|------------------------------------------|-----------------------------------------------------|
| Old SQLite tasks                         | Deserialize cleanly; new fields default `None`.     |
| Frontend (`static/index.html`)            | Ignores new fields — no JS changes.                 |
| `/api/tasks/{id}` shape                   | Same keys + 2 new optional ones.                    |
| `GEMINI_DIRECT` mode                      | Unchanged.                                           |
| `summarize_transcript` public signature   | Unchanged. Accepts the same `(transcript, cb)`.    |
| `ENABLE_DIARIZATION=false` operator override | Skips the diarization call entirely.             |
| Non-Gemini provider                       | Skips diarization; falls through to today's path.  |

## 9. Test plan

Unit (`tests/test_diarization.py` — new):

1. `_parse_diarization_response` — full payload parses correctly.
2. `_parse_diarization_response` — empty `speaker_map` accepted.
3. `_parse_diarization_response` — markdown fences stripped.
4. Diarization sync helper invokes Gemini with `temperature=0.1`.
5. Prompt contains both Hebrew and English few-shots.

Integration (extends `tests/test_two_call_pipeline.py`):

6. `summarize_transcript` calls diarize → synth → extract in that order
   (diarization output reaches both downstream calls).
7. Diarization failure → both downstream calls receive raw transcript;
   `diarized_transcript` stays `None`; task still completes.
8. `ENABLE_DIARIZATION=False` → diarize is NOT called; raw transcript
   reaches downstream calls.

Manual verification:

- Run on a real meeting recording → verify `speaker_map` is populated
  and `diarized_transcript` has clear speaker turns.
- Run on a single-lecturer recording → verify `speaker_map` is empty
  or has only one entry.

## 10. Cost note

+1 Gemini call per WHISPER-mode recording. Diarization tokens ≈
input transcript size + small JSON output. At $0.001/recording today,
this adds ~$0.0005. Negligible.

## 11. Acceptance criteria (`UPGRADE_PROMPT.md` Task 1.2)

- [x] Pre-processing step runs in WHISPER_LOCAL + WHISPER_API.
- [x] Identifies speaker count, tags each paragraph, proposes human
      labels when semantic clues exist.
- [x] Saved as `result_json.diarized_transcript`.
- [x] Flag `ENABLE_DIARIZATION` in config (default True).
- [x] Pyannote / audio embeddings explicitly out of scope.

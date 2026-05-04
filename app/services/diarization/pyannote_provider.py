"""
Pyannote-based acoustic speaker diarization — code-only, home server / GPU.

Never imported at server boot. Activated only when DIARIZATION_PROVIDER=pyannote.
Heavy deps (pyannote.audio, torch) live in requirements-heavy.txt and must
never be added to the top-level requirements.txt.

Output contract (matches Gemini text diarizer):
  diarize_audio(audio_path, transcript) -> (diarized_transcript, speaker_map)

speaker_map is always {} because Pyannote identifies speaker *segments* but
not speaker *names*. The Gemini text diarizer infers names from self-
introductions in the transcript; Pyannote cannot do that acoustically.
"""
import importlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_pipeline(model: str, hf_token: str):
    """
    Lazy-load pyannote.audio.Pipeline.

    Raises ImportError if pyannote.audio is not installed (expected on Fly.io).
    """
    audio_module = importlib.import_module("pyannote.audio")
    return audio_module.Pipeline.from_pretrained(
        model, use_auth_token=hf_token or None
    )


def _annotation_to_turns(annotation) -> list[tuple[float, float, str]]:
    """
    Convert a pyannote Annotation to a chronologically sorted list of
    (start_sec, end_sec, raw_speaker_label) tuples.
    """
    turns = [
        (seg.start, seg.end, lbl)
        for seg, _, lbl in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t[0])
    return turns


def _assign_sentences_to_speakers(
    sentences: list[str],
    turns: list[tuple[float, float, str]],
    total_duration: float,
) -> list[tuple[str, str]]:
    """
    Proportionally assign *sentences* to speakers based on pyannote turn durations.

    Strategy: each turn's share of the total audio time determines how many
    sentences it receives (rounded). This is a best-effort heuristic — precise
    alignment would require word-level Whisper timestamps.

    Returns: [(normalized_speaker_label, sentence), ...]
    """
    if not sentences:
        return []

    if not turns:
        return [("Speaker A", s) for s in sentences]

    # Map raw pyannote labels to "Speaker A", "Speaker B", ... by first appearance.
    seen: list[str] = []
    for _, _, lbl in turns:
        if lbl not in seen:
            seen.append(lbl)
    label_map = {lbl: f"Speaker {_LABELS[i % len(_LABELS)]}" for i, lbl in enumerate(seen)}

    if total_duration <= 0:
        total_duration = turns[-1][1]
    if total_duration <= 0:
        return [("Speaker A", s) for s in sentences]

    n = len(sentences)
    result: list[tuple[str, str]] = []
    idx = 0

    for i, (start, end, raw_lbl) in enumerate(turns):
        speaker = label_map[raw_lbl]
        duration = max(0.0, end - start)
        fraction = duration / total_duration
        # Last turn takes all remaining sentences to avoid under-assignment.
        if i == len(turns) - 1:
            count = n - idx
        else:
            count = max(1, round(fraction * n))
        for s in sentences[idx : idx + count]:
            result.append((speaker, s))
        idx += count
        if idx >= n:
            break

    return result


def diarize_audio(
    audio_path: str | Path,
    transcript: str,
) -> tuple[str, dict[str, str]]:
    """
    Run Pyannote speaker diarization on *audio_path*; map labels onto *transcript*.

    Returns (diarized_transcript, speaker_map) with the same shape as the
    Gemini text diarizer. speaker_map is always {} — Pyannote does not detect
    speaker names.

    Raises:
      ImportError: if pyannote.audio is not installed.
      Any exception from the Pyannote pipeline propagates up; the caller
      (summarizer.py) wraps this in a best-effort try/except.
    """
    from app.config import settings

    pipeline = _load_pipeline(settings.pyannote_model, settings.hf_token)
    annotation = pipeline(str(audio_path))

    turns = _annotation_to_turns(annotation)
    total_duration = turns[-1][1] if turns else 0.0

    # Split transcript into sentences on sentence-ending punctuation.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
    if not sentences and transcript.strip():
        sentences = [transcript.strip()]

    assigned = _assign_sentences_to_speakers(sentences, turns, total_duration)

    # Group consecutive sentences from the same speaker into one line.
    lines: list[str] = []
    current_speaker: str | None = None
    current_sentences: list[str] = []

    for speaker, sentence in assigned:
        if speaker != current_speaker:
            if current_speaker is not None and current_sentences:
                lines.append(f"{current_speaker}: {' '.join(current_sentences)}")
            current_speaker = speaker
            current_sentences = [sentence]
        else:
            current_sentences.append(sentence)

    if current_speaker is not None and current_sentences:
        lines.append(f"{current_speaker}: {' '.join(current_sentences)}")

    diarized_transcript = "\n".join(lines)
    return diarized_transcript, {}

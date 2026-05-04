"""
Diarization provider package.

Import the provider module directly — this package exports nothing at import
time so that heavy deps (pyannote.audio, torch) are never loaded on server boot.
"""

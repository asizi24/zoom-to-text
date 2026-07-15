"""
API routers, split by concern:

    tasks.py      — task lifecycle: submit (URL/upload), list, get, delete,
                    cancel, retry
    events.py     — live updates: SSE stream + transcript-delta polling
    chat.py       — ask/chat about a completed lesson (+ history)
    audio.py      — persistent audio streaming with HTTP Range support
    flashcards.py — flashcards JSON + Anki/CSV export

app/api/routes.py aggregates them into the single `router` that main.py
mounts under /api, and re-exports the helpers tests import from it.
"""

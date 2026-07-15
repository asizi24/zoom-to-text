"""
Data-access layer, split by domain concern:

    tasks.py — task rows: CRUD, sticky terminal states, live transcript
               preview, audio-path tracking, retention queries
    jobs.py  — the durable job queue riding on tasks.payload_json
    auth.py  — users, magic tokens, sessions
    chat.py  — per-task chat history

All repositories share the single cached aiosqlite connection owned by
app.state (`state._get_db()`); connection lifecycle (open/close/init/
migrations) stays there. app.state also re-exports every function here, so
callers and tests keep the stable `state.<fn>` seam — new code may import
from the specific repository instead.
"""

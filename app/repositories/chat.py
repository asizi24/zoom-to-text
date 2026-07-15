"""
Chat-history repository: per-task multi-turn chat persisted in tasks.chat_history.
"""
import json
import logging

from app import state

logger = logging.getLogger(__name__)

# Keep at most this many messages in the stored history (user + model turns combined).
# Older messages are trimmed from the front so the most recent context is preserved.
_MAX_CHAT_MESSAGES = 40


async def get_chat_history(task_id: str) -> list[dict]:
    """
    Return the stored chat history for a task as a list of
    {"role": "user"|"model", "content": "..."} dicts.
    Returns an empty list if no history yet.
    """
    db = await state._get_db()
    async with db.execute(
        "SELECT chat_history FROM tasks WHERE id=?", [task_id]
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or not row["chat_history"]:
        return []
    try:
        return json.loads(row["chat_history"])
    except (ValueError, TypeError):
        return []


async def append_chat_message(task_id: str, role: str, content: str) -> None:
    """
    Append one message to the task's chat history and persist it.
    Trims the history to _MAX_CHAT_MESSAGES (oldest messages dropped first).

    Single-statement append via SQLite's JSON1 functions: the old
    read-modify-write interleaved at the awaits, so two concurrent chats on
    the same task could each read the same history and silently drop the
    other's message. json_insert('$[#]') appends atomically; the CASE trims
    one from the front once the cap is reached (appends are one-at-a-time, so
    the cap holds). A NULL or corrupt column falls back to a fresh array.
    """
    message = json.dumps({"role": role, "content": content}, ensure_ascii=False)
    db = await state._get_db()
    await db.execute(
        """
        UPDATE tasks
           SET chat_history = (
               SELECT CASE
                          WHEN json_array_length(appended.j) > ?
                          THEN json_remove(appended.j, '$[0]')
                          ELSE appended.j
                      END
                 FROM (
                     SELECT json_insert(
                                CASE WHEN json_valid(COALESCE(chat_history, ''))
                                     THEN chat_history ELSE '[]' END,
                                '$[#]', json(?)
                            ) AS j
                 ) AS appended
           )
         WHERE id = ?
        """,
        [_MAX_CHAT_MESSAGES, message, task_id],
    )
    await db.commit()


async def clear_chat_history(task_id: str) -> None:
    """Delete the chat history for a task (user-initiated reset)."""
    db = await state._get_db()
    await db.execute(
        "UPDATE tasks SET chat_history=NULL WHERE id=?", [task_id]
    )
    await db.commit()

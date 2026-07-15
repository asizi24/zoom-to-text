"""
Facade-contract regression tests for the Phase 4 decomposition.

app.state and app.api.routes are the stable seams the rest of the app (and
this suite's monkeypatching) depend on. If a future refactor moves a function
out of the repositories without re-exporting it, these fail loudly instead of
surfacing as an AttributeError deep inside a worker.
"""
import inspect


def test_state_facade_exports_full_api():
    from app import state

    expected = {
        # lifecycle (owned by state itself)
        "DB_PATH", "_get_db", "close_db", "init_db", "ping",
        # tasks
        "create_task", "update_task", "complete_task", "fail_task",
        "cancel_task", "requeue_task", "get_task", "get_task_for_user",
        "backfill_task_owners", "list_tasks", "delete_task",
        "append_partial_transcript", "get_partial_transcript",
        "set_audio_path", "get_audio_path", "clear_audio_path",
        "list_reclaimable_media",
        # jobs
        "set_job_payload", "get_job_payload", "clear_job_payload",
        "finalize_job_payload", "reset_interrupted_tasks",
        "list_payload_file_paths",
        # auth
        "get_or_create_user", "create_magic_token", "consume_magic_token",
        "create_session", "get_session_user", "delete_session",
        "purge_expired_auth",
        # chat
        "get_chat_history", "append_chat_message", "clear_chat_history",
    }
    missing = {name for name in expected if not hasattr(state, name)}
    assert not missing, f"state facade lost re-exports: {sorted(missing)}"


def test_repositories_share_the_state_connection():
    """Every repository must go through state._get_db so tests patching
    state.DB_PATH / state._db keep controlling all data access."""
    from app.repositories import auth, chat, jobs, tasks

    for module in (auth, chat, jobs, tasks):
        source = inspect.getsource(module)
        assert "state._get_db()" in source, f"{module.__name__} bypasses the shared connection"
        assert "aiosqlite.connect" not in source, f"{module.__name__} opens its own connection"


def test_routes_aggregator_compat_exports():
    from app.api import routes

    for name in ("router", "_parse_range", "_path_under_audio_root",
                 "_AUDIO_ROOT", "_ALLOWED_EXTENSIONS", "AskRequest"):
        assert hasattr(routes, name), f"routes.py lost compat export: {name}"


def test_all_api_routes_still_registered():
    """The aggregated router must expose the complete pre-split route set."""
    from app.api.routes import router

    paths = {(r.path, m) for r in router.routes for m in r.methods}
    expected = {
        ("/tasks", "POST"), ("/tasks", "GET"),
        ("/tasks/upload", "POST"),
        ("/tasks/{task_id}", "GET"), ("/tasks/{task_id}", "DELETE"),
        ("/tasks/{task_id}/cancel", "POST"),
        ("/tasks/{task_id}/retry", "POST"),
        ("/tasks/{task_id}/transcript", "GET"),
        ("/tasks/{task_id}/events", "GET"),
        ("/tasks/{task_id}/ask", "POST"),
        ("/tasks/{task_id}/chat", "POST"), ("/tasks/{task_id}/chat", "GET"),
        ("/tasks/{task_id}/chat", "DELETE"),
        ("/tasks/{task_id}/audio", "GET"),
        ("/tasks/{task_id}/flashcards", "GET"),
        ("/tasks/{task_id}/flashcards/export.apkg", "GET"),
        ("/tasks/{task_id}/flashcards/export.csv", "GET"),
    }
    missing = expected - paths
    assert not missing, f"routes lost in the split: {sorted(missing)}"

"""
Tests for the runtime-mutable configuration store (app_config kv table +
app/services/runtime_config.py). This is what the first-boot Setup Wizard
writes its choices into, so the round-trip and typed accessors must be solid.
"""
import asyncio

from app import state
from app.services import runtime_config


def _run_with_db(tmp_path, monkeypatch, coro_factory):
    async def wrapper():
        monkeypatch.setattr(state, "DB_PATH", tmp_path / "cfg.db")
        monkeypatch.setattr(state, "_db", None, raising=False)
        await state.init_db()
        try:
            await coro_factory()
        finally:
            await state.close_db()

    try:
        asyncio.run(wrapper())
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_kv_roundtrip_and_upsert(tmp_path, monkeypatch):
    async def scenario():
        assert await runtime_config.get("missing") is None
        assert await runtime_config.get("missing", "fallback") == "fallback"
        await runtime_config.set(runtime_config.KEY_SUMMARY_BACKEND, "gemini")
        assert await runtime_config.get(runtime_config.KEY_SUMMARY_BACKEND) == "gemini"
        # Setting the same key again upserts (no duplicate-key error, value replaced)
        await runtime_config.set(runtime_config.KEY_SUMMARY_BACKEND, "ollama")
        assert await runtime_config.get(runtime_config.KEY_SUMMARY_BACKEND) == "ollama"

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_bool_accessor_and_setup_flag(tmp_path, monkeypatch):
    async def scenario():
        assert await runtime_config.is_setup_complete() is False
        assert await runtime_config.get_bool("flag", default=True) is True
        for truthy in ("1", "true", "YES", "on"):
            await runtime_config.set("flag", truthy)
            assert await runtime_config.get_bool("flag") is True
        for falsy in ("0", "false", "no", ""):
            await runtime_config.set("flag", falsy)
            assert await runtime_config.get_bool("flag") is False
        await runtime_config.mark_setup_complete()
        assert await runtime_config.is_setup_complete() is True

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_delete_and_all(tmp_path, monkeypatch):
    async def scenario():
        await runtime_config.set("a", "1")
        await runtime_config.set("b", "2")
        assert await runtime_config.all() == {"a": "1", "b": "2"}
        await runtime_config.delete("a")
        assert await runtime_config.get("a") is None
        assert set((await runtime_config.all()).keys()) == {"b"}
        # Deleting a missing key is a no-op, not an error
        await runtime_config.delete("does-not-exist")

    _run_with_db(tmp_path, monkeypatch, scenario)

"""
Structured logging — stdlib only, no new dependencies.

Two output formats, selected by settings.log_format:
  text — the classic human-readable line for local terminals (default)
  json — one JSON object per line, ready for log aggregation (Docker/Loki/etc.)

Correlation ids ride on contextvars, not on logger names or call sites:
  request_id — bound for the lifetime of each HTTP request by
               RequestContextMiddleware (installed in app/main.py)
  task_id    — bound for the lifetime of each pipeline job by the worker loop
               (app/services/worker.py)

Formatting happens synchronously on the caller's stack, so any log line emitted
inside those scopes automatically carries the ids — in both formats, with no
changes to the ~200 existing logger calls. Lines emitted from plain threads
(e.g. the Whisper executor pool) have no bound context and simply omit them.
"""
import json
import logging
import re
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg (+ ids, + exc)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        task_id = task_id_var.get()
        if task_id:
            payload["task_id"] = task_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # ensure_ascii=False: pipeline messages are Hebrew — keep them readable
        # in the aggregated output instead of \uXXXX soup.
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """The pre-existing human format, plus a [req=… task=…] tag when bound."""

    def __init__(self):
        super().__init__("%(asctime)s [%(levelname)s] %(name)s:%(ctx)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        parts = []
        request_id = request_id_var.get()
        if request_id:
            parts.append(f"req={request_id}")
        task_id = task_id_var.get()
        if task_id:
            parts.append(f"task={task_id}")
        record.ctx = f" [{' '.join(parts)}]" if parts else ""
        return super().format(record)


def setup_logging(log_format: str = "text") -> None:
    """Install a single root handler with the chosen formatter.

    Non-destructive to foreign handlers: only handlers this function installed
    before (marked with _ztt_handler) are replaced — pytest's caplog capture
    handler, for one, must survive an app import mid-test. uvicorn's own
    loggers are switched to propagate so access/error lines share the same
    format and destination as application logs.
    """
    is_json = log_format.lower() == "json"
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if is_json else TextFormatter())
    handler._ztt_handler = True  # ownership mark — see docstring
    if is_json:
        # JSON output is machine-consumed — force UTF-8 so Hebrew survives even
        # when the platform default is a locale codepage (Windows stderr uses
        # cp125x + backslashreplace, which corrupts the "one JSON object per
        # line" contract). No-op on Linux/Docker where UTF-8 is the default.
        try:
            handler.stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, "_ztt_handler", False)]
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True


# ── Request-id binding (pure ASGI — safe for SSE/streaming responses) ─────────

_REQUEST_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")


class RequestContextMiddleware:
    """Bind a request_id for each HTTP request and echo it as X-Request-ID.

    An inbound X-Request-ID (e.g. from a reverse proxy) is sanitized and
    honored so ids correlate across hops; otherwise a fresh id is minted.
    Implemented as raw ASGI rather than BaseHTTPMiddleware so long-lived
    streaming responses (the SSE endpoints) pass through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for key, value in scope.get("headers", []):
            if key == b"x-request-id":
                incoming = _REQUEST_ID_SAFE.sub("", value.decode("latin-1"))[:64] or None
                break
        request_id = incoming or new_request_id()
        token = request_id_var.set(request_id)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)

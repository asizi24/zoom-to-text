"""
Standard API error envelope + exception-handler registration.

Every error response from the API has the same JSON shape:

    {"detail": "<human message>", "code": "<machine slug>", "request_id": "<id>"}

`detail` stays the primary key — the frontend and the test suite already read
it — and is always a STRING (FastAPI's default validation handler puts a list
there; we flatten it and move the raw list to an `errors` field instead).
`code` gives clients a stable machine-readable discriminator; `request_id`
lets a user-reported error be matched to the exact log lines that produced it.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_config import request_id_var
from app.services.errors import PipelineError

logger = logging.getLogger(__name__)

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "not_ready",
}


def error_response(
    status: int,
    detail: str,
    code: str | None = None,
    extra: dict | None = None,
) -> JSONResponse:
    """Build an envelope response. Exported for endpoints that construct
    error bodies directly (e.g. /ready)."""
    body = {
        "detail": detail,
        "code": code or _STATUS_CODES.get(status, f"http_{status}"),
        "request_id": request_id_var.get(),
    }
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status, content=body)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    response = error_response(exc.status_code, str(exc.detail))
    for key, value in (exc.headers or {}).items():
        response.headers[key] = value
    return response


async def _validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = ".".join(str(part) for part in first.get("loc", []) if part != "body")
    msg = first.get("msg", "Invalid request")
    detail = f"{loc}: {msg}" if loc else msg
    return error_response(422, detail, extra={"errors": jsonable_encoder(errors)})


async def _pipeline_error_handler(request: Request, exc: PipelineError):
    # A PipelineError escaping a route means an upstream AI/media step failed
    # synchronously (background-pipeline failures are stored in the task row,
    # not raised here). 502: our service is fine, the upstream call was not.
    logger.error(
        f"Pipeline error on {request.method} {request.url.path}: "
        f"{exc.user_message} ({exc.detail})"
    )
    return error_response(502, exc.user_message, code="pipeline_error")


async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so unhandled errors return the clean envelope, never raw HTML."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return error_response(500, "Internal server error. Please try again.")


def register_exception_handlers(app: FastAPI) -> None:
    # StarletteHTTPException covers fastapi.HTTPException (a subclass) and
    # Starlette's own 404/405 for unknown routes.
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(PipelineError, _pipeline_error_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

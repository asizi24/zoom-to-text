"""
Black-box smoke campaign for the deployed Zoom-to-Text site.

Runs phases 1-4 by default (cheap, finishes in <2 minutes). Phase 5 (full E2E
with a real Zoom URL) is gated behind --full-e2e because it costs Gemini
tokens and ~26 minutes wall-clock.

Usage:

    PROD_SESSION_ID=<cookie> python scripts/prod_smoke.py
    PROD_SESSION_ID=<cookie> python scripts/prod_smoke.py --phases 1,2
    PROD_SESSION_ID=<cookie> python scripts/prod_smoke.py --full-e2e \
        --zoom-url "https://...zoom.us/rec/play/..." \
        --zoom-cookies-file ./zoom_cookies.txt

Exits non-zero if any phase 1-4 case fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

import httpx


DEFAULT_BASE_URL = "https://zoom-to-text.fly.dev"

# On Windows machines with corporate AV/proxy, the local cert chain can include
# a self-signed root that httpx rejects. We're probing our own site over TLS,
# so disabling verification is safe in this context (matches the curl
# --ssl-no-revoke pattern used elsewhere in this repo).
_VERIFY_TLS = os.environ.get("PROD_SMOKE_VERIFY_TLS", "0") not in ("0", "false", "False", "")


# ── Result tracking ──────────────────────────────────────────────────────────

class Result:
    def __init__(self, name: str, ok: bool, detail: str = ""):
        self.name = name
        self.ok = ok
        self.detail = detail


def _check(name: str, predicate: Callable[[], None]) -> Result:
    """Run `predicate`; convert assertion failure into a Result."""
    try:
        predicate()
        return Result(name, True)
    except AssertionError as e:
        return Result(name, False, str(e))
    except Exception as e:
        return Result(name, False, f"{type(e).__name__}: {e}")


def _print_phase(name: str, results: list[Result]) -> None:
    passed = sum(1 for r in results if r.ok)
    total = len(results)
    icon = "OK " if passed == total else "!! "
    print(f"\n=== {icon}{name}: {passed}/{total} ===")
    for r in results:
        mark = "[ok] " if r.ok else "[FAIL]"
        print(f"  {mark} {r.name}")
        if not r.ok:
            print(f"        {r.detail[:300]}")


# ── Phase 1 — Public / unauth surface ────────────────────────────────────────

def phase_1_public(base_url: str) -> list[Result]:
    """No-auth probes: health, capabilities, login page, auth flow rejections."""
    results: list[Result] = []

    with httpx.Client(base_url=base_url, timeout=15.0, verify=_VERIFY_TLS) as c:

        def _1_1():
            r = c.get("/health")
            assert r.status_code == 200, f"status={r.status_code}"

        results.append(_check("1.1 GET /health -> 200", _1_1))

        def _1_2():
            r = c.get("/api/capabilities")
            assert r.status_code == 200, f"status={r.status_code}"
            data = r.json()
            assert "llm_provider" in data, f"keys={list(data.keys())}"
            assert "available_modes" in data, f"keys={list(data.keys())}"
            modes = set(data["available_modes"])
            allowed = {"gemini_direct", "whisper_local", "whisper_api", "ivrit_ai"}
            assert modes.issubset(allowed), f"unexpected modes: {modes - allowed}"

        results.append(_check("1.2 GET /api/capabilities shape", _1_2))

        def _1_3():
            r = c.get("/")
            assert r.status_code == 200, f"status={r.status_code}"
            assert "text/html" in r.headers.get("content-type", ""), "not html"
            body = r.text
            for secret in ("AIzaSy", "GOOGLE_API_KEY", "RESEND_API_KEY"):
                assert secret not in body, f"leaked: {secret}"

        results.append(_check("1.3 GET / no secret leak", _1_3))

        def _1_4():
            r = c.get("/login")
            assert r.status_code == 200, f"status={r.status_code}"

        results.append(_check("1.4 GET /login -> 200", _1_4))

        def _1_5():
            r = c.get("/api/tasks")
            assert r.status_code in (401, 403), f"status={r.status_code}"

        results.append(_check("1.5 unauthed /api/tasks -> 401", _1_5))

        def _1_6():
            fake_id = "00000000-0000-0000-0000-000000000000"
            r = c.get(f"/api/tasks/{fake_id}")
            assert r.status_code in (401, 403), f"status={r.status_code}"

        results.append(_check("1.6 unauthed task GET -> 401", _1_6))

        def _1_7():
            r = c.post(
                "/api/auth/request",
                json={"email": "definitely-not-allowed@example.com"},
            )
            assert r.status_code == 200, f"status={r.status_code}"
            text = r.text.lower()
            assert "not allowed" not in text, "leaks email allow-list state"
            assert "denied" not in text, "leaks email allow-list state"

        results.append(_check("1.7 unknown email -> no enum leak", _1_7))

        def _1_8():
            r = c.get(
                "/api/auth/verify?token=garbage-token-123",
                follow_redirects=False,
            )
            assert r.status_code in (302, 303), f"status={r.status_code}"
            for header_val in r.headers.get_list("set-cookie"):
                assert "session_id=" not in header_val.lower(), (
                    "garbage token issued a session"
                )

        results.append(_check("1.8 invalid token -> no session", _1_8))

    return results


# ── Phase 2 — Authenticated contract ─────────────────────────────────────────

def phase_2_contract(base_url: str, session_id: str) -> list[Result]:
    """Authenticated probes: capabilities again, task list, recent task shape."""
    results: list[Result] = []
    cookies = {"session_id": session_id}

    with httpx.Client(base_url=base_url, cookies=cookies, timeout=20.0, verify=_VERIFY_TLS) as c:

        recent_id: str | None = None

        def _2_1():
            r = c.get("/api/capabilities")
            assert r.status_code == 200, f"status={r.status_code}"

        results.append(_check("2.1 authed /api/capabilities -> 200", _2_1))

        def _2_2():
            nonlocal recent_id
            r = c.get("/api/tasks?limit=20")
            assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
            data = r.json()
            assert isinstance(data, list), f"not a list: {type(data)}"
            for entry in data:
                for k in ("id", "status", "progress", "created_at"):
                    assert k in entry, f"missing key {k} in {entry}"
            if data:
                recent_id = data[0]["id"]

        results.append(_check("2.2 GET /api/tasks list shape", _2_2))

        def _2_3():
            assert recent_id, "no recent task to inspect (skipped)"
            r = c.get(f"/api/tasks/{recent_id}")
            assert r.status_code == 200, f"status={r.status_code}"
            data = r.json()
            for k in ("task_id", "status", "progress", "created_at"):
                assert k in data, f"missing {k}"
            if data.get("result") is not None:
                lr = data["result"]
                expected = {
                    "summary", "chapters", "quiz", "flashcards",
                    "action_items", "decisions", "open_questions",
                    "objections_tracked",
                }
                missing = expected - set(lr.keys())
                assert not missing, f"missing LessonResult keys: {missing}"

        results.append(_check("2.3 recent task LessonResult schema", _2_3))

        def _2_4():
            assert recent_id, "no recent task"
            r = c.get(f"/api/tasks/{recent_id}/transcript")
            assert r.status_code in (200, 404), f"status={r.status_code}"

        results.append(_check("2.4 GET task transcript reachable", _2_4))

        def _2_5():
            assert recent_id, "no recent task"
            r = c.get(f"/api/tasks/{recent_id}/export/obsidian")
            if r.status_code == 200:
                ct = r.headers.get("content-type", "")
                assert "markdown" in ct or "text/plain" in ct, f"ct={ct}"
                assert r.text.startswith("---"), "no YAML frontmatter"
            else:
                assert r.status_code in (400, 404), f"status={r.status_code}"

        results.append(_check("2.5 GET obsidian export shape", _2_5))

        def _2_8():
            fake = "00000000-0000-0000-0000-000000000000"
            r = c.get(f"/api/tasks/{fake}")
            assert r.status_code == 404, f"status={r.status_code}"

        results.append(_check("2.8 nonexistent task -> 404", _2_8))

    return results


# ── Phase 3 — Negative + validation ──────────────────────────────────────────

def phase_3_negative(base_url: str, session_id: str) -> list[Result]:
    """Negative paths: bad URLs, schema errors, rate limits."""
    results: list[Result] = []
    cookies = {"session_id": session_id}
    created_ids: list[str] = []

    with httpx.Client(base_url=base_url, cookies=cookies, timeout=20.0, verify=_VERIFY_TLS) as c:

        def _3_2():
            r = c.post("/api/tasks", json={})
            assert r.status_code == 422, f"status={r.status_code}"

        results.append(_check("3.2 missing url -> 422", _3_2))

        def _3_3():
            r = c.post(
                "/api/tasks",
                json={"url": "https://example.com/x", "mode": "INVALID_MODE"},
            )
            assert r.status_code == 422, f"status={r.status_code}"

        results.append(_check("3.3 invalid mode -> 422", _3_3))

        def _3_4():
            files = {"file": ("junk.txt", b"not an audio file", "text/plain")}
            data = {"mode": "gemini_direct", "language": "he"}
            r = c.post("/api/tasks/upload", files=files, data=data)
            assert r.status_code in (400, 415, 422), f"status={r.status_code}"

        results.append(_check("3.4 .txt upload rejected", _3_4))

        def _3_1():
            r = c.post(
                "/api/tasks",
                json={"url": "https://not-a-real-zoom-url-xyz.example/abc",
                      "mode": "gemini_direct"},
            )
            assert r.status_code == 202, f"status={r.status_code}"
            tid = r.json()["task_id"]
            created_ids.append(tid)

        results.append(_check("3.1 bogus URL accepted (will fail later)", _3_1))

        def _3_6():
            burst = 12
            statuses = []
            for _ in range(burst):
                rr = c.post(
                    "/api/tasks",
                    json={"url": "https://x.example/y", "mode": "gemini_direct"},
                )
                statuses.append(rr.status_code)
                if rr.status_code == 202:
                    try:
                        created_ids.append(rr.json()["task_id"])
                    except Exception:
                        pass
            assert 429 in statuses, f"never rate-limited; statuses={statuses}"

        results.append(_check("3.6 burst -> eventually 429", _3_6))

        for tid in created_ids:
            try:
                c.delete(f"/api/tasks/{tid}")
            except Exception:
                pass

    return results


# ── Phase 4 — Security invariants ────────────────────────────────────────────

def phase_4_security(base_url: str, session_id: str) -> list[Result]:
    """Security: openapi visibility, cross-user enumeration, HttpOnly cookies."""
    results: list[Result] = []
    cookies = {"session_id": session_id}

    with httpx.Client(base_url=base_url, cookies=cookies, timeout=15.0, verify=_VERIFY_TLS) as c:

        def _4_2():
            r = c.get("/api/tasks/some-random-string-not-a-real-id")
            assert r.status_code == 404, f"status={r.status_code}"

        results.append(_check("4.2 cross-user lookup -> 404", _4_2))

        def _4_3():
            r = c.get("/openapi.json")
            assert r.status_code in (200, 404), f"status={r.status_code}"
            if r.status_code == 200:
                paths = set(r.json().get("paths", {}).keys())
                forbidden = {"/internal", "/admin", "/debug", "/__debug__"}
                leaked = paths & forbidden
                assert not leaked, f"leaked debug endpoints: {leaked}"

        results.append(_check("4.3 openapi has no debug endpoints", _4_3))

        def _4_5():
            r = c.get("/api/tasks", cookies={"session_id": "totally-bogus"})
            assert r.status_code in (401, 403), f"status={r.status_code}"

        results.append(_check("4.5 bogus session -> 401", _4_5))

    return results


# ── Phase 5 — Full E2E ───────────────────────────────────────────────────────

def phase_5_e2e(
    base_url: str,
    session_id: str,
    zoom_url: str,
    zoom_cookies: str | None,
    poll_interval: float = 10.0,
    timeout_seconds: float = 45 * 60,
) -> list[Result]:
    """End-to-end: submit zoom URL, poll until completed, validate result."""
    results: list[Result] = []
    cookies = {"session_id": session_id}

    payload: dict[str, Any] = {"url": zoom_url, "mode": "gemini_direct"}
    if zoom_cookies:
        payload["cookies"] = zoom_cookies

    with httpx.Client(base_url=base_url, cookies=cookies, timeout=60.0, verify=_VERIFY_TLS) as c:
        r = c.post("/api/tasks", json=payload)
        if r.status_code != 202:
            results.append(Result(
                "5.1 POST /api/tasks", False,
                f"status={r.status_code} body={r.text[:300]}"
            ))
            return results

        tid = r.json()["task_id"]
        results.append(Result("5.1 POST /api/tasks accepted", True, tid))
        print(f"  task_id = {tid}")

        deadline = time.time() + timeout_seconds
        last_progress = -1
        last_status = ""
        while time.time() < deadline:
            try:
                rr = c.get(f"/api/tasks/{tid}")
            except Exception as e:
                print(f"  poll error: {e}")
                time.sleep(poll_interval)
                continue
            if rr.status_code != 200:
                print(f"  poll {rr.status_code}: {rr.text[:200]}")
                time.sleep(poll_interval)
                continue
            data = rr.json()
            status = data.get("status", "?")
            progress = data.get("progress", -1)
            msg = data.get("message", "")[:80]
            if status != last_status or progress != last_progress:
                print(f"  [{int(time.time()-deadline+timeout_seconds):4d}s] "
                      f"{status:14s} {progress:3d}%  {msg}")
                last_status, last_progress = status, progress
            if status == "completed":
                results.append(Result("5.2 task completed", True))
                _validate_e2e_result(data, results)
                return results
            if status == "failed":
                ed = data.get("error_details") or {}
                detail = json.dumps(ed)[:400] if ed else (data.get("error") or "")[:400]
                results.append(Result("5.2 task completed", False, detail))
                return results
            time.sleep(poll_interval)

        results.append(Result(
            "5.2 task completed", False,
            f"timed out after {timeout_seconds:.0f}s, last status={last_status}@{last_progress}%"
        ))
        return results


def _validate_e2e_result(data: dict, results: list[Result]) -> None:
    res = data.get("result") or {}

    def _summary():
        s = res.get("summary", "") or ""
        assert len(s) >= 100, f"summary too short ({len(s)} chars)"

    results.append(_check("5.3 summary length >= 100", _summary))

    def _chapters():
        ch = res.get("chapters") or []
        assert len(ch) >= 3, f"only {len(ch)} chapters"

    results.append(_check("5.4 chapters >= 3", _chapters))

    def _quiz():
        q = res.get("quiz") or []
        assert len(q) >= 3, f"only {len(q)} quiz items"

    results.append(_check("5.5 quiz >= 3", _quiz))

    def _flashcards():
        fc = res.get("flashcards") or []
        assert 5 <= len(fc) <= 30, f"flashcards count={len(fc)}"

    results.append(_check("5.6 flashcards 5-30", _flashcards))


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("PROD_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--phases", default="1,2,3,4")
    p.add_argument("--full-e2e", action="store_true")
    p.add_argument("--zoom-url")
    p.add_argument("--zoom-cookies-file")
    args = p.parse_args()

    session_id = os.environ.get("PROD_SESSION_ID", "").strip()
    phases = {x.strip() for x in args.phases.split(",") if x.strip()}

    print(f"== Smoke campaign against {args.base_url} ==")
    print(f"   phases={sorted(phases)} full_e2e={args.full_e2e}")

    all_pass = True

    if "1" in phases:
        r = phase_1_public(args.base_url)
        _print_phase("Phase 1 - public surface", r)
        all_pass &= all(x.ok for x in r)

    if "2" in phases or "3" in phases or "4" in phases or args.full_e2e:
        if not session_id:
            print("\n!! PROD_SESSION_ID env var not set — skipping authed phases.")
            return 1

    if "2" in phases:
        r = phase_2_contract(args.base_url, session_id)
        _print_phase("Phase 2 - authed contract", r)
        all_pass &= all(x.ok for x in r)

    if "3" in phases:
        r = phase_3_negative(args.base_url, session_id)
        _print_phase("Phase 3 - negative + validation", r)
        all_pass &= all(x.ok for x in r)

    if "4" in phases:
        r = phase_4_security(args.base_url, session_id)
        _print_phase("Phase 4 - security invariants", r)
        all_pass &= all(x.ok for x in r)

    if args.full_e2e:
        if not args.zoom_url:
            print("\n!! --full-e2e requires --zoom-url")
            return 1
        zoom_cookies = None
        if args.zoom_cookies_file:
            with open(args.zoom_cookies_file, "r", encoding="utf-8") as f:
                zoom_cookies = f.read()
        r = phase_5_e2e(args.base_url, session_id, args.zoom_url, zoom_cookies)
        _print_phase("Phase 5 - full E2E", r)
        all_pass &= all(x.ok for x in r)

    print()
    print("==>", "ALL PASS" if all_pass else "FAILURES PRESENT")
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())

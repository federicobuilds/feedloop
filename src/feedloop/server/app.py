"""HTTP boundary over one Engine: explicit POST delivery, cached GET observation,
measured views, committed watch batches, durable feedback, discovery, the
scorecard and the tuner controls, plus the packaged no-build client and the
media folder. Standard library only.

Every mutation is authorized by ``ledger.authorize_mutation`` (shared key header
plus an exact browser origin) before the Engine is touched. Failure bodies carry
a code, never an exception message or a path. Route handlers compose Engine
methods and never rank, attribute, journal or tune on their own.
"""
from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

from feedloop import ledger, serving, tuning
from feedloop.engine import Engine, initialize_stores
from feedloop.slots import item_key
from feedloop.sources.filesystem import FilesystemSource, TextHashEncoder, build_text_hash_space

MAX_BODY_BYTES = 1 << 20
OBSERVATION_TTL_S = 600.0
OBSERVATION_CAP = 32
SCORECARD_TTL_S = 5.0
MEDIA_PREFIX = "/media/"
API_KEY_HEADER = "x-ai-api-key"
DENIED = {"mutation_credential_required": 401, "mutation_origin_denied": 403}
DEMO_ATTRIBUTION = {"window_s": 5.0, "policy_revision": "demo-explicit-w5-v1", "min_advance_s": 0.0}
DEMO_SPACE = "sidecar_text"
DEMO_ROLES = {"visual": "visual", "semantic": DEMO_SPACE, "voice": "audioembed", "sound": "audiomix"}

Response = tuple[int, list[tuple[str, str]], bytes]


def web_root() -> Path:
    """The packaged client: ``web/`` beside this module after installation, or the
    checkout's top-level ``web/`` when running from source."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "web", here.parents[2] / "web"):
        if (candidate / "index.html").is_file():
            return candidate
    raise FileNotFoundError("web client resources are not installed")


def _json_bytes(value) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def _reply(status: int, value, extra: list[tuple[str, str]] | None = None) -> Response:
    return status, [("Content-Type", "application/json")] + (extra or []), _json_bytes(value)


def _detail(code: str, status: int) -> Response:
    return _reply(status, {"detail": code})


def _int(values, name, default, low, high):
    raw = values.get(name, [None])[0]
    if raw is None or raw == "":
        return default
    if not raw.lstrip("-").isdigit():
        raise ValueError(name)
    return max(low, min(high, int(raw)))


def build_engine(folder, state_dir, *, api_key=None, clock=time.time, attribution=None, config=None, initialize=False, cutover_ts=None,
                 automatic_tuning=True, demo_space=True):
    """Compose the filesystem source and an Engine over local stores. ``initialize``
    creates the stores explicitly (the ``demo`` command); ``serve`` never does."""
    source = FilesystemSource(folder, state_dir, media_prefix=MEDIA_PREFIX, clock=clock)
    ledger_path, tuner_path = source.state_dir / "ledger.sqlite", source.state_dir / "tuner.sqlite"
    if initialize:
        if not ledger_path.exists():
            initialize_stores(ledger_path=ledger_path, tuner_path=tuner_path, cutover_ts=clock() if cutover_ts is None else cutover_ts, clock=clock)
        if demo_space and DEMO_SPACE not in source.spaces():
            build_text_hash_space(source, DEMO_SPACE)
    encoder = TextHashEncoder([DEMO_SPACE]) if DEMO_SPACE in source.spaces() else None
    roles = DEMO_ROLES if DEMO_SPACE in source.spaces() else None
    engine = Engine(catalog=source, signals=source, spaces=source, encoder=encoder, ledger_path=ledger_path, tuner_path=tuner_path,
                    read_current=source.read_current, apply_change=source.apply_change, attribution=attribution, config=config,
                    automatic_tuning=automatic_tuning, clock=clock, **({"space_roles": roles} if roles else {}))
    return source, engine


class FeedloopApp:
    """Transport-independent request handling; ``handle`` maps one request to one response."""

    def __init__(self, engine: Engine, *, api_key: str, allowed_origins, source: FilesystemSource | None = None,
                 web_dir: Path | None = None, clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic):
        self.engine, self.source = engine, source
        self.api_key, self.allowed_origins = api_key, tuple(allowed_origins)
        self.web_dir = Path(web_dir) if web_dir is not None else web_root()
        self.clock, self.monotonic = clock, monotonic
        self.lock = threading.Lock()
        self.observations: dict[str, tuple[float, bytes]] = {}
        self.scorecard_cache: tuple[float, bytes] | None = None
        engine.tuner.listeners.append(self._drop_scorecard)

    def _drop_scorecard(self):
        with self.lock:
            self.scorecard_cache = None

    # ------------------------------------------------------------ authority
    def authorize(self, headers: Mapping[str, str]) -> Response | None:
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        try:
            ledger.authorize_mutation(headers=raw, configured_key=self.api_key, allowed_origins=self.allowed_origins)
        except ledger.ContractError as exc:
            code = str(exc) if str(exc) in ("mutation_auth_unconfigured", "mutation_origin_unconfigured", *DENIED) else "mutation_auth_unconfigured"
            return _detail(code, DENIED.get(code, 503))
        return None

    # -------------------------------------------------------------- routing
    def handle(self, method: str, target: str, headers: Mapping[str, str] | None = None, body: bytes | None = None) -> Response:
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        parts = urlsplit(target)
        path, query = unquote(parts.path), parse_qs(parts.query, keep_blank_values=True)
        try:
            if path.startswith("/api/"):
                return self._api(method, path[5:], query, headers, body)
            if path.startswith(MEDIA_PREFIX):
                return self._media(method, path[len(MEDIA_PREFIX):], headers)
            return self._static(method, path)
        except Exception:
            return _detail("internal_error", 500)

    def _api(self, method, route, query, headers, body) -> Response:
        if route == "health" and method == "GET":
            return _reply(200, {"ok": True})
        if route == "config" and method == "GET":
            return _reply(200, self.config())
        if route == "feed":
            if method == "GET":
                return self.observe_feed(query)
            if method == "POST":
                return self._mutation(headers, body, self.deliver_feed)
        if route == "view" and method == "POST":
            return self._mutation(headers, body, lambda payload: _reply(200, self.engine.view(payload)))
        if route == "watch" and method == "POST":
            return self._mutation(headers, body, self.commit_watch)
        if route == "feedback" and method == "POST":
            return self._mutation(headers, body, self.feedback)
        if route == "feedback/reconcile" and method == "POST":
            return self._mutation(headers, body, self.reconcile)
        if route == "tick" and method == "POST":
            return self._mutation(headers, body, lambda _payload: self.tick(), optional_body=True)
        if route == "search" and method == "GET":
            return self.search(query)
        if route == "similar" and method == "GET":
            return self.similar(query)
        if route == "scorecard" and method == "GET":
            return self.scorecard()
        if route == "tuner/reset" and method == "POST":
            return self._mutation(headers, body, lambda _payload: self.tuner("reset", None), optional_body=True)
        if route == "tuner/revert" and method == "POST":
            ledger_id = query.get("ledger_id", [""])[0]
            if not ledger_id.isdigit():
                return _detail("invalid_ledger_id", 400)
            return self._mutation(headers, body, lambda _payload: self.tuner("revert", int(ledger_id)), optional_body=True)
        return _detail("not_found", 404)

    def _mutation(self, headers, body, action, *, optional_body=False) -> Response:
        denied = self.authorize(headers)
        if denied is not None:
            return denied
        if body is None:
            return _detail("request_too_large", 413)
        if not body and optional_body:
            return action(None)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return _detail("invalid_json", 400)
        return action(payload)

    # ----------------------------------------------------------------- feed
    @staticmethod
    def observation_key(payload, kinds) -> str:
        def decoded(name, default):
            value = payload.get(name, default)
            return json.loads(value) if isinstance(value, str) and value else value or default
        return json.dumps([payload.get("limit", 24), bool(payload.get("images", False)), serving.feed_intent(payload.get("intent"), kinds=kinds),
                           payload.get("surface", "feed"), payload.get("offset", 0), decoded("cursor", None), payload.get("session_id", ""),
                           decoded("eligibility", {}), payload.get("filter_identity")], sort_keys=True, separators=(",", ":"), allow_nan=False)

    def annotate(self, result):
        """Each served item carries the source's named reason when its sidecar could not be
        read (``sidecar_reason``), shown on the card like every other unmeasured value."""
        items = result.get("items") if isinstance(result, dict) else None
        if not items or self.source is None or not hasattr(self.source, "sidecar_reasons"):
            return result
        reasons = self.source.sidecar_reasons([(item["kind"], item["id"]) for item in items if isinstance(item, dict)])
        for item in items:
            if isinstance(item, dict):
                item["sidecar_reason"] = reasons.get((item["kind"], item["id"]))
        return result

    def deliver_feed(self, payload) -> Response:
        result = self.annotate(self.engine.feed(payload))
        if isinstance(payload, dict) and result.get("status") in ("ok", "partial", "empty"):
            try:
                key = self.observation_key(payload, self.engine.kinds)
            except (ValueError, TypeError):
                key = None
            if key is not None:
                observation = {**result, "request_id": None, "client_request_id": None,
                               "items": [{**item, "request_id": None, "served_item_id": None, "viewed_event_id": None} for item in result.get("items", [])]}
                with self.lock:
                    if key not in self.observations and len(self.observations) >= OBSERVATION_CAP:
                        del self.observations[min(self.observations, key=lambda k: self.observations[k][0])]
                    self.observations[key] = (self.monotonic(), _json_bytes(observation))
        return _reply(200, result)

    def observe_feed(self, query) -> Response:
        """Cached observation only: never ranks, records, opens a store or starts work."""
        try:
            payload = dict(limit=_int(query, "limit", 24, 1, 60), images=query.get("images", ["0"])[0] in ("1", "true"),
                           intent=query.get("intent", [""])[0], surface=query.get("surface", ["feed"])[0], offset=_int(query, "offset", 0, 0, 10 ** 9),
                           cursor=query.get("cursor", [""])[0], session_id=query.get("session_id", [""])[0],
                           eligibility=query.get("eligibility", [""])[0], filter_identity=query.get("filter_identity", [None])[0])
            if payload["surface"] not in serving.VIEW_SURFACES:
                return _reply(200, {"status": "error", "error_code": "invalid_surface", "items": []})
            key = self.observation_key(payload, self.engine.kinds)
        except (ValueError, TypeError):
            return _reply(200, {"status": "error", "error_code": "invalid_feed_request", "items": []})
        with self.lock:
            cached = self.observations.get(key)
        if cached and self.monotonic() - cached[0] <= OBSERVATION_TTL_S:
            return 200, [("Content-Type", "application/json")], cached[1]
        return _reply(200, {"status": "unavailable", "error_code": "feed_observation_unavailable", "items": [], "request_id": None})

    # ------------------------------------------------------------ recording
    def commit_watch(self, payload) -> Response:
        """A player batch becomes authoritative watch history only after the source's own
        validation and the ledger's binding checks both accept every row; a rejected batch
        is reported with its reason and writes no source history; an uncommitted source
        write is refused, never reported as imported."""
        if self.source is None:
            return _detail("watch_capture_unavailable", 503)
        required = {"capture_id", "stream_session_id", "session_id", "events"}
        if not isinstance(payload, dict) or set(payload) != required or not isinstance(payload["events"], list) or not 0 < len(payload["events"]) <= 2000:
            return _reply(200, {"status": "error", "error_code": "invalid_watch_batch"})
        now = self.clock()
        rows = []
        for event in payload["events"]:
            if not isinstance(event, dict):
                return _reply(200, {"status": "error", "error_code": "invalid_watch_batch"})
            rows.append({"id": event.get("id"), "stream_session_id": payload["stream_session_id"], "type": event.get("type"),
                         "item_id": event.get("item_id"), "occurred_at": event.get("occurred_at"), "position": event.get("position"),
                         "duration": event.get("duration"), "session_id": payload["session_id"], "viewed_event_id": event.get("viewed_event_id"),
                         "previous_event_id": event.get("previous_event_id"), "playback_rate": event.get("playback_rate", 1),
                         "canonical_session_id": payload["session_id"]})
        # Order of proof: the source validates the batch without writing, the ledger validates
        # every view and session binding (quarantining unproven rows), and only a batch the
        # ledger accepted in full becomes authoritative watch history in the source.
        check = self.source.commit_watch(rows, received_at=now, dry_run=True)
        if check.get("status") != "valid":
            return _reply(200, {"status": "rejected", "error_code": check.get("error_code", "watch_batch_rejected"), "stored_steps": 0, "capture_id": payload["capture_id"]})
        batch = {"capture_id": payload["capture_id"], "source_id": "player", "received_at": now, "source_revision": "feedloop-player-v1",
                 "status": "committed", "reason": None, "events_json": rows}
        try:
            receipt = self.engine.record([{"type": "watch_capture", "batch": batch}])[0]
        except ledger.ContractError as exc:
            return _reply(200, {"status": "rejected", "error_code": str(exc), "stored_steps": 0, "capture_id": payload["capture_id"]})
        if receipt.get("status") == "duplicate":
            return _reply(200, {**receipt, "stored_steps": 0, "capture_id": payload["capture_id"]})
        if receipt.get("quarantined"):
            return _reply(200, {"status": "rejected", "error_code": "watch_view_or_session_unproven", "quarantined": receipt["quarantined"], "outcomes": receipt.get("outcomes", 0),
                                "stored_steps": 0, "capture_id": payload["capture_id"]})
        commit = self.source.commit_watch(rows, received_at=now)
        if commit.get("status") != "committed":
            return _detail("sync_commit_unconfirmed", 503)
        return _reply(200, {**receipt, "stored_steps": commit["stored"], "capture_id": payload["capture_id"]})

    def feedback(self, payload) -> Response:
        if not isinstance(payload, dict):
            return _reply(200, {"status": "indeterminate", "error_code": "invalid_payload", "operation_id": None})
        try:
            result = self.engine.record([{"type": "feedback", "operation": payload}])[0]
        except (ValueError, TypeError):
            return _reply(200, {"status": "indeterminate", "error_code": "feedback_unavailable", "operation_id": payload.get("operation_id")})
        return _reply(200, result)

    def reconcile(self, payload) -> Response:
        if not isinstance(payload, dict) or set(payload) != {"operation_id"} or self.engine.read_current is None:
            return _reply(200, {"status": "indeterminate", "error_code": "invalid_payload", "operation_id": None})
        try:
            result = ledger.reconcile_feedback(self.engine.ledger_path, operation_id=payload["operation_id"], read_current=self.engine.read_current,
                                               kinds=self.engine.kinds)
        except ledger.ContractError as exc:
            return _reply(200, {"status": "indeterminate", "error_code": str(exc), "operation_id": payload["operation_id"]})
        return _reply(200, result)

    def tick(self) -> Response:
        result = self.engine.tick()
        with self.lock:
            self.scorecard_cache = None
        return _reply(200, result)

    # ------------------------------------------------------------ discovery
    def search(self, query) -> Response:
        """Exactly what the facade returns: a means-only space has no window rows and is
        reported as no-feature by ``Engine.search``; the route never ranks on its own."""
        text = query.get("q", [""])[0]
        mode = query.get("mode", ["look"])[0]
        try:
            offset, limit = _int(query, "offset", 0, 0, 10 ** 6), _int(query, "limit", 20, 1, 60)
        except ValueError:
            return _reply(200, {"status": "error", "error_code": "invalid_search_request", "items": []})
        return _reply(200, self.engine.search(text, mode, offset=offset, limit=limit))

    def similar(self, query) -> Response:
        try:
            kind = query.get("kind", [self.engine.primary])[0]
            key = item_key(kind, int(query.get("id", [""])[0]), self.engine.kinds)
            offset, limit = _int(query, "offset", 0, 0, 10 ** 6), _int(query, "limit", 20, 1, 60)
            result = self.engine.similar(key, offset=offset, limit=limit)
        except ValueError:
            return _reply(200, {"status": "error", "error_code": "invalid_similar_request", "items": []})
        return _reply(200, self.annotate(result))

    # ---------------------------------------------------------- scorecard
    def scorecard(self) -> Response:
        with self.lock:
            cached = self.scorecard_cache
        if cached and self.monotonic() - cached[0] <= SCORECARD_TTL_S:
            return 200, [("Content-Type", "application/json")], cached[1]
        body = _json_bytes(self.engine.scorecard())
        with self.lock:
            self.scorecard_cache = (self.monotonic(), body)
        return 200, [("Content-Type", "application/json")], body

    def tuner(self, action, ledger_id) -> Response:
        result = self.engine.tuner.write(action, ledger_id)
        with self.lock:
            self.scorecard_cache = None
        return _reply(200, result)

    def config(self) -> dict:
        engine = self.engine
        return {"kinds": list(engine.kinds), "primary": engine.primary, "attribution": dict(engine.attribution),
                "tuner_ripen_s": tuning.TUNER_RIPEN_S,
                "space_roles": dict(engine.roles), "spaces": list(engine.spaces.spaces()), "encoder": engine.encoder is not None,
                "space_meta": {space: self.source.space_meta(space) for space in engine.spaces.spaces()} if self.source is not None else {},
                "visibility_policy": serving.VISIBILITY_POLICY, "automatic_tuning": bool(engine.tuner.automatic)}

    # --------------------------------------------------------------- files
    def _file(self, root: Path, relative: str, headers=None) -> Response:
        """Serve one regular file under ``root``. Every component of the requested path is
        checked with ``lstat`` before resolution so a symlink anywhere in it is refused; the
        resolved target must lie inside the resolved root and outside any hidden directory."""
        parts = Path(relative).parts
        if not parts or any(part in ("", ".", "..") or part.startswith(".") for part in parts):
            return _detail("not_found", 404)
        current = root
        for part in parts:
            current = current / part
            try:
                if os.path.islink(current) or not (current.is_file() or current.is_dir()):
                    return _detail("not_found", 404)
            except OSError:
                return _detail("not_found", 404)
        candidate = current.resolve()
        try:
            inside = candidate.relative_to(root.resolve())
        except ValueError:
            return _detail("not_found", 404)
        if any(part.startswith(".") for part in inside.parts) or not candidate.is_file():
            return _detail("not_found", 404)
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        size = candidate.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = (headers or {}).get("range")
        if range_header and range_header.startswith("bytes=") and size:
            first, _, last = range_header[6:].partition("-")
            if first.isdigit() and (last == "" or last.isdigit()):
                start, end = int(first), int(last) if last else size - 1
                if start > end or start >= size:
                    return 416, [("Content-Range", f"bytes */{size}")], b""
                end = min(end, size - 1)
                status = 206
        with open(candidate, "rb") as handle:
            handle.seek(start)
            data = handle.read(end - start + 1)
        extra = [("Content-Type", mime), ("Accept-Ranges", "bytes"), ("Cache-Control", "no-store")]
        if status == 206:
            extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
        return status, extra, data

    def _static(self, method, path) -> Response:
        if method not in ("GET", "HEAD"):
            return _detail("method_not_allowed", 405)
        relative = path.lstrip("/") or "index.html"
        if relative in ("feed", "home", "search", "similar", "engine"):
            relative = "index.html"
        return self._file(self.web_dir, relative)

    def _media(self, method, relative, headers) -> Response:
        """Only catalogued media, addressed as ``/media/<kind>/<id>``. The state directory,
        sidecars, dotfiles and every other path are not items and are never served."""
        if method not in ("GET", "HEAD"):
            return _detail("method_not_allowed", 405)
        if self.source is None:
            return _detail("not_found", 404)
        kind, _, raw_id = relative.partition("/")
        if kind not in self.engine.kinds or not raw_id.isdigit():
            return _detail("not_found", 404)
        path = self.source.media_path((kind, int(raw_id)))
        if path is None:
            return _detail("not_found", 404)
        return self._file(self.source.folder, path.relative_to(self.source.folder).as_posix(), headers)


class _Handler(BaseHTTPRequestHandler):
    app: FeedloopApp
    server_version = "feedloop"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _dispatch(self):
        length = self.headers.get("Content-Length")
        body = b""
        if length is not None:
            if not length.isdigit() or int(length) > MAX_BODY_BYTES:
                body = None
                self.rfile.read(min(int(length), MAX_BODY_BYTES) if length.isdigit() else 0)
            else:
                body = self.rfile.read(int(length))
        status, headers, data = self.app.handle(self.command, self.path, dict(self.headers.items()), body)
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    do_GET = do_POST = do_HEAD = _dispatch


def run_server(app: FeedloopApp, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Bind a threaded server on ``host:port`` (0 picks a free port). The caller runs
    ``serve_forever`` and ``shutdown``; nothing is started here beyond the socket."""
    handler = type("FeedloopHandler", (_Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def origins_for(host: str, port: int):
    names = {host, "127.0.0.1", "localhost"} if host in ("127.0.0.1", "localhost", "0.0.0.0", "") else {host}
    return tuple(sorted(f"http://{name}:{port}" for name in names))


def new_api_key() -> str:
    return secrets.token_urlsafe(24)


__all__ = ["FeedloopApp", "build_engine", "run_server", "web_root", "origins_for", "new_api_key", "MAX_BODY_BYTES", "OBSERVATION_TTL_S",
           "DEMO_ATTRIBUTION", "DEMO_SPACE", "DEMO_ROLES", "MEDIA_PREFIX", "API_KEY_HEADER"]

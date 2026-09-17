"""Prospective evidence and durable feedback, with no import-time I/O.

Only initialize_event_store creates storage. Other commands require an initialized
file; readers open mode=ro and return status=unavailable rather than creating it.
The existing rec_impressions.db is the intended path; no path is opened implicitly.

record_served accepts the rec_requests fields, with config/arms as JSON mappings
instead of config_json/arms_json; config_hash is computed (or checked if supplied).
Items require kind,id,source_rank,score,arm,control,explore,category,duration_s,trace.
record_event accepts event_type,source,source_event_id,session_id,kind,item_id,
occurred_at,payload and optional event_id,parent_id,request_id,corrects_id.
View payload: visible_fraction,dwell_ms,foreground,display_rank,placement.
These browser assertions are validated, NOT independently verified by the server.
Outcome payload: provenance ('confirmed_delta_v1' or 'unknown'), signal and exactly
one signal schema: watch(watched_s_delta,started_at), rating(rating_before,
rating_after), engagement(engagement_delta=1), correction(no extra fields, corrects_id required).
Unknown provenance is retained but cannot earn attribution. Source adapters must
establish canonical session identity and actual deltas; host enums are not used.
These are trusted engine APIs, not browser endpoints: only a verified source adapter
may assert confirmed_delta_v1. Browser observations never prove outcome provenance.
An experiment_id fixes configuration, arm definitions and ranking/eligibility code;
changing those requires a new experiment_id, not reuse of the same arm labels.
ranking_content_id identifies an immutable ordered cached page, including its
experiment/configuration/session/revisions. New delivery IDs do not relabel content.
Trusted adapters register explicit session mappings, including their revision;
unresolved mappings cannot earn attribution. Later merges invalidate old evidence.

Feedback operations require operation_id,kind,item_id,action,session_id and optional
request_id,viewed_event_id. action=rating also requires rating100 (including None);
action=engagement increments once; action=undo requires undo_of. read_current((kind,id))
returns {status:'ok',rating100:None|int,engagement_count:int}. apply_change((kind,id),
change) returns the same authoritative fields with status='confirmed', or a typed
status 'conflict'/'indeterminate'. change is {action:'rating',rating100:...} or
{action:'engagement',delta:1|-1}. Neither callback is called under a SQLite transaction.

Cooperating processes serialize by persisted item ownership. This is NOT CAS
against independent host writers and cannot detect ABA. Reconciliation matching
a rating confirms observation only, so it emits unknown evidence. Matching an engagement
counter NEVER proves our increment/decrement applied and remains indeterminate.
Uncertain increments are never retried. A divergent/unchanged value stays
indeterminate and owned until an operator resolves the external ambiguity.
Process-liveness checks assume cooperating processes share a POSIX PID namespace.
If the recorded PID is still alive (including PID reuse), unfinished dispatch stays
owned. Non-POSIX unfinished dispatch needs operator review. No timed lease steals it.

kinds names every item kind the ledger accepts, primary first; the primary kind is
the only one that earns watch credit. source_revision values are recorded opaquely;
the host verifies them before importing a capture.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit
import uuid

from feedloop.taste import DEFAULT_KINDS


SCHEMA_VERSION = 1
_REQUEST_FIELDS = (
    "request_id", "client_request_id", "schema_version", "created_at", "session_id",
    "surface", "recommender", "context_id", "config_json", "config_hash",
    "ranker_revision", "feature_revision", "preference_revision", "preference_cutoff",
    "intent_revision", "eligibility_revision", "seed", "experiment_id", "arms_json",
    "ranking_content_id", "canonical_session_id", "session_mapping_revision",
)
_EVENT_FIELDS = (
    "event_id", "event_type", "occurred_at", "session_id", "kind", "item_id",
    "request_id", "parent_id", "corrects_id", "source", "source_event_id", "payload_json",
    "canonical_session_id", "session_mapping_revision",
)
_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS rec_metadata (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        schema_version INTEGER NOT NULL, cutover_ts REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS rec_requests (
        request_id TEXT PRIMARY KEY, client_request_id TEXT NOT NULL,
        schema_version INTEGER NOT NULL, created_at REAL NOT NULL, session_id TEXT NOT NULL,
        surface TEXT NOT NULL, recommender TEXT NOT NULL, context_id TEXT NOT NULL,
        config_json TEXT NOT NULL, config_hash TEXT NOT NULL, ranker_revision TEXT NOT NULL,
        feature_revision TEXT NOT NULL, preference_revision TEXT NOT NULL,
        preference_cutoff REAL NOT NULL, intent_revision TEXT NOT NULL,
        eligibility_revision TEXT NOT NULL, seed INTEGER NOT NULL,
        experiment_id TEXT, arms_json TEXT NOT NULL, ranking_content_id TEXT NOT NULL,
        canonical_session_id TEXT, session_mapping_revision TEXT,
        UNIQUE(session_id,client_request_id))""",
    "CREATE INDEX IF NOT EXISTS rec_experiment_requests ON rec_requests(experiment_id)",
    "CREATE INDEX IF NOT EXISTS rec_cached_content ON rec_requests(ranking_content_id)",
    """CREATE TABLE IF NOT EXISTS rec_eligibility_snapshots (
        snapshot_id TEXT PRIMARY KEY, protocol_version INTEGER NOT NULL CHECK(protocol_version=1),
        saved_filter_id TEXT NOT NULL, mode TEXT NOT NULL,
        predicate_sha256 TEXT NOT NULL, membership_sha256 TEXT NOT NULL,
        observed_at REAL NOT NULL, created_at REAL NOT NULL,
        kind_counts TEXT NOT NULL, membership_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS rec_request_eligibility (
        request_id TEXT NOT NULL REFERENCES rec_requests(request_id),
        role TEXT NOT NULL CHECK(role IN ('generation','delivery')),
        snapshot_id TEXT NOT NULL REFERENCES rec_eligibility_snapshots(snapshot_id),
        PRIMARY KEY(request_id,role))""",
    """CREATE TABLE IF NOT EXISTS rec_watch_capture_imports (
        source_id TEXT NOT NULL,capture_id TEXT NOT NULL,received_at REAL NOT NULL,
        imported_at REAL NOT NULL,content_hash TEXT NOT NULL,status TEXT NOT NULL,
        reason TEXT,PRIMARY KEY(source_id,capture_id))""",
    """CREATE TABLE IF NOT EXISTS rec_watch_steps (
        source_id TEXT NOT NULL,event_id TEXT NOT NULL,capture_id TEXT NOT NULL,
        previous_event_id TEXT,payload_json TEXT NOT NULL,status TEXT NOT NULL,
        reason TEXT,outcome_id TEXT,stream_session_id TEXT NOT NULL,item_id INTEGER NOT NULL,
        occurred_at REAL NOT NULL,PRIMARY KEY(source_id,event_id))""",
    "CREATE INDEX IF NOT EXISTS rec_watch_event_id ON rec_watch_steps(event_id)",
    "CREATE INDEX IF NOT EXISTS rec_watch_stream ON rec_watch_steps(source_id,stream_session_id,item_id,occurred_at) WHERE status!='quarantined'",
    """CREATE TABLE IF NOT EXISTS rec_watch_view_bindings (
        viewed_id TEXT PRIMARY KEY,canonical_session_id TEXT NOT NULL,
        session_mapping_revision TEXT NOT NULL,source_id TEXT NOT NULL,
        capture_id TEXT NOT NULL,bound_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS rec_session_mappings (
        mapping_seq INTEGER PRIMARY KEY, alias_session_id TEXT NOT NULL,
        canonical_session_id TEXT NOT NULL, mapping_revision TEXT NOT NULL, recorded_at REAL NOT NULL,
        UNIQUE(alias_session_id,mapping_revision))""",
    """CREATE TABLE IF NOT EXISTS rec_events (
        event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL
            CHECK(event_type IN ('served','viewed','outcome')),
        occurred_at REAL NOT NULL, received_at REAL NOT NULL, session_id TEXT NOT NULL,
        kind TEXT NOT NULL, item_id INTEGER NOT NULL,
        request_id TEXT REFERENCES rec_requests(request_id),
        parent_id TEXT REFERENCES rec_events(event_id), corrects_id TEXT REFERENCES rec_events(event_id),
        source TEXT NOT NULL, source_event_id TEXT NOT NULL, payload_json TEXT NOT NULL,
        canonical_session_id TEXT, session_mapping_revision TEXT,
        UNIQUE(source,source_event_id))""",
    "CREATE UNIQUE INDEX IF NOT EXISTS rec_one_serve ON rec_events(request_id,kind,item_id) WHERE event_type='served'",
    "CREATE UNIQUE INDEX IF NOT EXISTS rec_one_view ON rec_events(parent_id) WHERE event_type='viewed'",
    "CREATE UNIQUE INDEX IF NOT EXISTS rec_one_correction ON rec_events(corrects_id) WHERE corrects_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS rec_item_time ON rec_events(session_id,kind,item_id,occurred_at)",
    """CREATE TABLE IF NOT EXISTS rec_event_receipts (
        source TEXT NOT NULL, source_event_id TEXT NOT NULL,
        event_id TEXT NOT NULL REFERENCES rec_events(event_id), submitted_json TEXT NOT NULL,
        PRIMARY KEY(source,source_event_id))""",
    """CREATE TABLE IF NOT EXISTS rec_attributions (
        outcome_id TEXT PRIMARY KEY REFERENCES rec_events(event_id),
        viewed_id TEXT NOT NULL REFERENCES rec_events(event_id),
        policy_revision TEXT NOT NULL, attributed_at REAL NOT NULL, through_ts REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS rec_attribution_runs (
        through_ts REAL PRIMARY KEY, window_s REAL NOT NULL, policy_revision TEXT NOT NULL,
        completed_at REAL NOT NULL, event_seq INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS rec_sync_captures (
        capture_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_revision TEXT NOT NULL,
        event_ids_json TEXT NOT NULL, owner_token TEXT NOT NULL, received_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS rec_sync_claims (
        source_id TEXT NOT NULL, client_event_id TEXT NOT NULL,
        capture_id TEXT NOT NULL REFERENCES rec_sync_captures(capture_id),
        PRIMARY KEY(source_id,client_event_id))""",
    """CREATE TABLE IF NOT EXISTS rec_sync_results (
        capture_id TEXT PRIMARY KEY REFERENCES rec_sync_captures(capture_id),
        status TEXT NOT NULL CHECK(status IN ('captured','quarantined')),
        reason TEXT, completed_at REAL NOT NULL, result_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS feedback_operations (
        operation_id TEXT PRIMARY KEY, kind TEXT NOT NULL, item_id INTEGER NOT NULL,
        operation_json TEXT NOT NULL, owner_pid INTEGER NOT NULL, created_at REAL NOT NULL,
        owner_token TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS feedback_steps (
        step_id INTEGER PRIMARY KEY, operation_id TEXT NOT NULL REFERENCES feedback_operations(operation_id),
        state TEXT NOT NULL CHECK(state IN ('planned','sent','confirmed','conflict','indeterminate')),
        ts REAL NOT NULL, facts_json TEXT NOT NULL, owner_token TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS feedback_operation_steps ON feedback_steps(operation_id,step_id)",
    """CREATE TABLE IF NOT EXISTS feedback_ownership (
        kind TEXT NOT NULL, item_id INTEGER NOT NULL, active_operation TEXT,
        latest_operation TEXT, owner_token TEXT, PRIMARY KEY(kind,item_id))""",
)


class ContractError(ValueError):
    """A sanitized public error code. Never includes payloads or callback text."""


def authorize_mutation(*, headers, configured_key, allowed_origins) -> None:
    """Pure caller-auth/CSRF gate for every privileged mutation.

    headers is Request.scope['headers'], preserving duplicates. A nonblank server
    key, one explicit x-ai-api-key header and one exact allowed Origin are mandatory.
    Neither query parameters, cookies, host library access nor permissive host CORS grant
    authority. The custom secret header plus exact Origin check prevents browser
    CSRF; this does not bind client-supplied session IDs to an authenticated user.
    """
    _require(isinstance(configured_key, str) and configured_key.strip(), "mutation_auth_unconfigured")
    _require(isinstance(allowed_origins, (tuple, list, set, frozenset)) and allowed_origins,
             "mutation_origin_unconfigured")
    for origin in allowed_origins:
        try:
            parsed = urlsplit(origin)
            valid = (isinstance(origin, str) and parsed.scheme in ("http", "https") and parsed.hostname
                     and parsed.netloc == parsed.hostname + (":" + str(parsed.port) if parsed.port is not None else "")
                     and not parsed.path and not parsed.query and not parsed.fragment
                     and origin == f"{parsed.scheme}://{parsed.netloc}" and "*" not in origin)
        except (ValueError, TypeError, AttributeError):
            valid = False
        _require(valid, "mutation_origin_unconfigured")
    keys = [v for k, v in headers if k.lower() == b"x-ai-api-key"]
    origins = [v for k, v in headers if k.lower() == b"origin"]
    _require(len(keys) == 1 and isinstance(keys[0], bytes) and
             hmac.compare_digest(keys[0], configured_key.encode("utf-8")), "mutation_credential_required")
    _require(len(origins) == 1 and origins[0] in [o.encode("utf-8") for o in allowed_origins], "mutation_origin_denied")


def _require(condition, code="invalid_payload"):
    if not condition:
        raise ContractError(code)


def _text(value):
    _require(isinstance(value, str) and 0 < len(value) <= 512 and value.strip() == value
             and all(ord(c) >= 32 for c in value), "invalid_identity")
    return value


def _number(value, *, minimum=0):
    _require(type(value) in (int, float), "invalid_number")
    try:
        valid = math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    _require(valid, "invalid_number")
    return value


def _integer(value, *, minimum=0):
    _require(type(value) is int and minimum <= value <= 9223372036854775807, "invalid_integer")
    return value


def _rating(value):
    _require(value is None or type(value) is int and 0 <= value <= 100, "invalid_rating")
    return value


def _keys(value, required, optional=()):
    _require(isinstance(value, Mapping), "invalid_mapping")
    _require(set(required) <= set(value) <= set(required) | set(optional), "invalid_fields")


def _json(value):
    def validate(v):
        if isinstance(v, Mapping):
            for k, x in v.items():
                _require(isinstance(k, str), "invalid_json_key")
                _require(k.lower().replace("_", "") not in {"query", "rawquery", "querytext", "rawquerytext", "password", "apikey", "authorization"},
                         "sensitive_field")
                validate(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                validate(x)
        elif type(v) in (int, float):
            _number(v, minimum=-float("inf"))
        else:
            _require(v is None or type(v) in (str, bool), "invalid_json")
    try:
        validate(value)
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, RecursionError, OverflowError):
        raise ContractError("invalid_json") from None
    _require(len(result) <= 262144, "payload_too_large")
    return result


def _identity(kind, item_id, kinds):
    _require(kind in kinds, "invalid_kind")
    return kind, _integer(item_id, minimum=1)


@contextmanager
def _connection(db_path, *, write=False, initialize=False):
    conn = None
    try:
        uri = Path(db_path).absolute().as_uri() + ("?mode=rwc" if initialize else "?mode=rw" if write else "?mode=ro")
        conn = sqlite3.connect(uri, uri=True, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        if not write:
            conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        if not initialize:
            row = conn.execute("SELECT * FROM rec_metadata WHERE singleton=1").fetchone()
            _require(row is not None and row["schema_version"] == SCHEMA_VERSION, "schema_unavailable")
        yield conn
        conn.commit()
    except sqlite3.Error:
        raise ContractError("store_unavailable") from None
    finally:
        if conn is not None:
            conn.close()


def initialize_event_store(db_path: str, *, cutover_ts: float) -> None:
    """Explicit migration only. Existing legacy tables and evidence are untouched."""
    _number(cutover_ts)
    with _connection(db_path, write=True, initialize=True) as conn:
        for sql in _SCHEMA:
            conn.execute(sql)
        old = conn.execute("SELECT * FROM rec_metadata WHERE singleton=1").fetchone()
        if old:
            _require(old["schema_version"] == SCHEMA_VERSION and old["cutover_ts"] == cutover_ts,
                     "cutover_conflict")
        else:
            conn.execute("INSERT INTO rec_metadata VALUES(1,?,?)", (SCHEMA_VERSION, cutover_ts))
        for table in ("rec_metadata", "rec_requests", "rec_session_mappings", "rec_events", "rec_event_receipts", "rec_attributions",
                      "rec_attribution_runs", "rec_sync_captures", "rec_sync_claims", "rec_sync_results",
                      "feedback_operations", "feedback_steps", "rec_watch_capture_imports",
                      "rec_watch_steps", "rec_watch_view_bindings", "rec_eligibility_snapshots",
                      "rec_request_eligibility"):
            for action in ("UPDATE", "DELETE"):
                conn.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()} "
                             f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'append_only'); END")


def _snapshot_id(value):
    _require(isinstance(value, str) and value.startswith("elig:v1:") and len(value) == 72
             and all(c in "0123456789abcdef" for c in value[8:]), "invalid_eligibility_snapshot")
    return value


def _snapshot_content(saved_filter_id, mode, predicate_sha256, eligible_ids, kinds):
    _require(isinstance(saved_filter_id, str) and 1 <= len(saved_filter_id) <= 20
             and saved_filter_id.isascii() and saved_filter_id.isdecimal() and int(saved_filter_id) > 0,
             "invalid_saved_filter")
    _require(mode in kinds, "invalid_saved_filter")
    _require(isinstance(predicate_sha256, str) and len(predicate_sha256) == 64
             and all(c in "0123456789abcdef" for c in predicate_sha256), "invalid_predicate_hash")
    _keys(eligible_ids, set(kinds))
    members = {}
    for kind in kinds:
        ids = eligible_ids[kind]
        _require(isinstance(ids, (list, tuple)), "invalid_eligibility_membership")
        members[kind] = sorted(_integer(i, minimum=1) for i in ids)
        _require(len(set(members[kind])) == len(ids), "duplicate_eligibility_member")
    _require(not any(members[kind] for kind in kinds if kind != mode), "eligibility_kind_conflict")
    # Membership is stored once, not as an event payload; the event JSON limit is unchanged.
    serialized = json.dumps(members, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    identity = "elig:v1:" + hashlib.sha256(_json([1, saved_filter_id, mode, predicate_sha256, digest]).encode()).hexdigest()
    return identity, digest, serialized, {kind: len(ids) for kind, ids in members.items()}


def _validated_snapshot(row):
    _require(row is not None, "eligibility_snapshot_missing")
    try:
        members = json.loads(row["membership_json"])
        identity, digest, serialized, counts = _snapshot_content(
            row["saved_filter_id"], row["mode"], row["predicate_sha256"], members, tuple(members))
        _require(row["protocol_version"] == 1 and row["snapshot_id"] == identity and
                 row["membership_sha256"] == digest and row["membership_json"] == serialized and
                 json.loads(row["kind_counts"]) == counts, "eligibility_snapshot_corrupt")
        _number(row["observed_at"])
        _number(row["created_at"])
        _require(row["observed_at"] <= row["created_at"], "eligibility_snapshot_corrupt")
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ContractError("eligibility_snapshot_corrupt") from None
    return {"snapshot_id": identity, "eligible_ids": members, "metadata": {
        key: row[key] for key in ("protocol_version", "saved_filter_id", "mode", "predicate_sha256",
                                  "membership_sha256", "observed_at", "created_at")
    } | {"counts": counts}}


def put_eligibility_snapshot(db_path, *, saved_filter_id, mode, predicate_sha256, eligible_ids, observed_at,
                             kinds=DEFAULT_KINDS):
    """Publish one complete trusted host observation; caller authorizes before resolving it."""
    identity, digest, serialized, counts = _snapshot_content(saved_filter_id, mode, predicate_sha256, eligible_ids, kinds)
    _number(observed_at)
    now = _number(time.time())
    _require(observed_at <= now, "eligibility_snapshot_future")
    with _connection(db_path, write=True) as conn:
        _require(observed_at >= conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0], "before_cutover")
        old = conn.execute("SELECT * FROM rec_eligibility_snapshots WHERE snapshot_id=?", (identity,)).fetchone()
        if old is not None:
            _validated_snapshot(old)
            _require(old["membership_json"] == serialized, "eligibility_snapshot_conflict")
        else:
            conn.execute("INSERT INTO rec_eligibility_snapshots VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (identity, 1, saved_filter_id, mode, predicate_sha256, digest, observed_at, now,
                          _json(counts), serialized))
    return {"snapshot_id": identity, "counts": counts}


def read_eligibility_snapshot(db_path, *, snapshot_id):
    """Read and verify immutable membership; absent/corrupt storage raises without repair."""
    _snapshot_id(snapshot_id)
    with _connection(db_path) as conn:
        return _validated_snapshot(conn.execute(
            "SELECT * FROM rec_eligibility_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone())


def _request_eligibility(conn, request_id):
    # Pre-extension inline ledgers remain readable until explicit additive migration.
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rec_request_eligibility'").fetchone():
        return {}
    return dict(conn.execute("SELECT role,snapshot_id FROM rec_request_eligibility WHERE request_id=? ORDER BY role",
                             (request_id,)))


def _session_mappings(conn, through_ts=None):
    rows = conn.execute("SELECT * FROM rec_session_mappings WHERE (? IS NULL OR recorded_at<=?) ORDER BY mapping_seq",
                        (through_ts, through_ts))
    return {r["alias_session_id"]: dict(r) for r in rows}


def _resolve_session(session_id, mappings):
    path, seen, current = [], set(), session_id
    while current in mappings:
        _require(current not in seen, "session_mapping_cycle")
        seen.add(current)
        row = mappings[current]
        path.append((current, row["canonical_session_id"], row["mapping_revision"]))
        if row["canonical_session_id"] == current:
            return current, hashlib.sha256(_json(path).encode()).hexdigest()
        current = row["canonical_session_id"]
    return None, None


def record_session_mapping(db_path: str, *, alias_session_id: str,
                           canonical_session_id: str, mapping_revision: str) -> None:
    """Explicit trusted identity evidence, never fingerprint inference or a read side effect."""
    for value in (alias_session_id, canonical_session_id, mapping_revision):
        _text(value)
    with _connection(db_path, write=True) as conn:
        _put_session_mapping(conn, alias_session_id, canonical_session_id, mapping_revision)


def _put_session_mapping(conn, alias_session_id, canonical_session_id, mapping_revision):
    old = conn.execute("SELECT canonical_session_id FROM rec_session_mappings WHERE alias_session_id=? AND mapping_revision=?",
                       (alias_session_id, mapping_revision)).fetchone()
    if old:
        _require(old[0] == canonical_session_id, "mapping_revision_conflict")
        return
    mappings = _session_mappings(conn)
    target, _ = _resolve_session(canonical_session_id, mappings)
    _require(target is None or target == canonical_session_id, "target_not_canonical")
    now = time.time()
    _require(now >= conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0], "before_cutover")
    if canonical_session_id not in mappings and alias_session_id != canonical_session_id:
        conn.execute("INSERT INTO rec_session_mappings(alias_session_id,canonical_session_id,mapping_revision,recorded_at) VALUES(?,?,?,?)",
                     (canonical_session_id, canonical_session_id, mapping_revision, now))
    conn.execute("INSERT INTO rec_session_mappings(alias_session_id,canonical_session_id,mapping_revision,recorded_at) VALUES(?,?,?,?)",
                 (alias_session_id, canonical_session_id, mapping_revision, now))
    _resolve_session(alias_session_id, _session_mappings(conn))


def _mapping_valid(row, mappings):
    canonical, revision = _resolve_session(row["session_id"], mappings)
    return canonical is not None and canonical == row["canonical_session_id"] and revision == row["session_mapping_revision"]


def watch_capture_cutover(db_path):
    with _connection(db_path) as conn:
        return conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0]


def imported_watch_captures(db_path, *, source_id):
    with _connection(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT capture_id FROM rec_watch_capture_imports WHERE source_id=? ORDER BY capture_id", (source_id,))]


def _watch_bound_rows(conn, events, requests, through_ts):
    """Project explicit committed identity proof without rewriting old exposures."""
    children = {}
    for row in events.values():
        children.setdefault(row["parent_id"], []).append(row)
    for binding in conn.execute("SELECT * FROM rec_watch_view_bindings WHERE bound_at<=?", (through_ts,)):
        viewed = events.get(binding["viewed_id"])
        if viewed is None:
            continue
        served = events.get(viewed["parent_id"])
        req = requests.get(viewed["request_id"])
        for row in (viewed, served, req, *children.get(viewed["event_id"], ())):
            if row is not None and row["canonical_session_id"] is None and row["session_id"] == viewed["session_id"]:
                row["canonical_session_id"] = binding["canonical_session_id"]
                row["session_mapping_revision"] = binding["session_mapping_revision"]


def import_watch_capture(db_path, *, batch, kinds=DEFAULT_KINDS):
    """Import only a committed source outbox receipt; never infer from watch totals."""
    _keys(batch, {"capture_id", "source_id", "received_at", "source_revision", "status", "reason", "events_json"})
    for name in ("capture_id", "source_id", "source_revision"):
        _text(batch[name])
    _number(batch["received_at"])
    _require(batch["status"] in ("committed", "quarantined") and isinstance(batch["events_json"], list), "watch_receipt_invalid")
    content_hash = hashlib.sha256(_json(batch).encode()).hexdigest()
    outcomes, quarantined = 0, 0
    with _connection(db_path, write=True) as conn:
        old = conn.execute("SELECT content_hash FROM rec_watch_capture_imports WHERE source_id=? AND capture_id=?",
                           (batch["source_id"], batch["capture_id"])).fetchone()
        if old:
            _require(old[0] == content_hash, "watch_receipt_conflict")
            return {"status": "duplicate", "outcomes": 0, "quarantined": 0}
        cutover = conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0]
        _require(cutover <= batch["received_at"] <= time.time(), "watch_receipt_clock_invalid")
        for row in sorted(batch["events_json"], key=lambda r: (r.get("occurred_at", 0), str(r.get("id", "")))):
            reason, outcome_id, status = batch["reason"], None, "quarantined"
            _keys(row, {"id", "stream_session_id", "type", "item_id", "occurred_at", "position", "duration",
                        "session_id", "viewed_event_id", "previous_event_id", "playback_rate", "canonical_session_id"})
            _text(row["id"])
            if conn.execute("SELECT 1 FROM rec_watch_steps WHERE event_id=?", (row["id"],)).fetchone():
                quarantined += 1
                continue
            conn.execute("SAVEPOINT watch_row")
            try:
                _require(batch["status"] == "committed" and not reason, "watch_receipt_quarantined")
                for name in ("stream_session_id", "session_id", "viewed_event_id", "canonical_session_id"):
                    _text(row[name])
                for name in ("occurred_at", "position", "duration", "playback_rate"):
                    _number(row[name])
                _require(cutover <= row["occurred_at"] <= batch["received_at"] and
                         0 <= row["position"] <= row["duration"] and row["duration"] > 0 and
                         0 < row["playback_rate"] <= 4, "watch_clock_or_position_invalid")
                _require(type(row["item_id"]) is int and row["item_id"] > 0, "watch_item_invalid")
                latest = conn.execute("""SELECT occurred_at FROM rec_watch_steps
                    WHERE source_id=? AND stream_session_id=? AND item_id=? AND status!='quarantined'
                    ORDER BY occurred_at DESC LIMIT 1""", (batch["source_id"], row["stream_session_id"], row["item_id"])).fetchone()
                _require(latest is None or row["occurred_at"] > latest[0], "watch_stream_clock_regressed")
                viewed = conn.execute("SELECT * FROM rec_events WHERE event_id=?", (row["viewed_event_id"],)).fetchone()
                _require(viewed is not None and viewed["event_type"] == "viewed" and viewed["kind"] == kinds[0] and
                    viewed["item_id"] == row["item_id"] and viewed["session_id"] == row["session_id"] and
                    viewed["occurred_at"] <= row["occurred_at"] and viewed["received_at"] <= batch["received_at"], "watch_view_unproven")
                mapping_revision = hashlib.sha256(_json([batch["source_id"], row["stream_session_id"],
                    row["session_id"], row["canonical_session_id"]]).encode()).hexdigest()
                _put_session_mapping(conn, row["session_id"], row["canonical_session_id"], mapping_revision)
                # Canonical identity is a committed host fact even if this
                # particular interval must be quarantined. Future views can bind
                # to the new alias; old view identities are never rewritten.
                conn.execute("RELEASE watch_row")
                conn.execute("SAVEPOINT watch_row")
                _require(viewed["canonical_session_id"] in (None, row["canonical_session_id"]), "watch_alias_conflict")
                canonical, revision = _resolve_session(row["session_id"], _session_mappings(conn))
                binding = conn.execute("SELECT * FROM rec_watch_view_bindings WHERE viewed_id=?", (row["viewed_event_id"],)).fetchone()
                if binding is not None:
                    _require(binding["canonical_session_id"] == canonical and binding["session_mapping_revision"] == revision, "watch_alias_conflict")
                else:
                    conn.execute("INSERT INTO rec_watch_view_bindings VALUES(?,?,?,?,?,?)", (row["viewed_event_id"], canonical,
                        revision, batch["source_id"], batch["capture_id"], time.time()))
                prior = conn.execute("SELECT * FROM rec_watch_steps WHERE source_id=? AND event_id=?",
                                     (batch["source_id"], row["previous_event_id"])).fetchone()
                previous = json.loads(prior["payload_json"]) if prior is not None else {}
                same_scope = all(previous.get(k) == row[k] for k in ("stream_session_id", "session_id", "canonical_session_id",
                                 "item_id", "viewed_event_id", "duration", "playback_rate"))
                # The first observed post-exposure progress point is only an
                # anchor. Never guess its start or award time before that point.
                if row["type"] == "view_start" or (row["type"] == "view_progress" and
                        (prior is None or prior["status"] not in ("anchor", "credited") or not same_scope)):
                    status = "anchor"
                else:
                    _require(prior is not None and prior["status"] in ("anchor", "credited"), "watch_chain_gap")
                    _require(same_scope, "watch_chain_scope_changed")
                    _require(previous["type"] in ("view_start", "view_progress", "view_seek"), "watch_not_playing")
                    _require(not conn.execute("SELECT 1 FROM rec_watch_steps WHERE source_id=? AND previous_event_id=? AND status!='quarantined'",
                                             (batch["source_id"], row["previous_event_id"])).fetchone(), "watch_chain_fork")
                    elapsed = row["occurred_at"] - previous["occurred_at"]
                    _require(0 < elapsed <= 15, "watch_chain_clock_gap")
                    if row["type"] == "view_seek":
                        status = "anchor"
                    else:
                        _require(row["type"] in ("view_progress", "view_pause", "view_complete"), "watch_type_invalid")
                        delta = row["position"] - previous["position"]
                        _require(0 <= delta <= elapsed * row["playback_rate"] + .5, "watch_seek_or_rate_unproven")
                        status = "credited" if row["type"] == "view_progress" else "paused"
                        if delta > 0:
                            event = {"event_type": "outcome", "source": "sync_capture", "source_event_id": hashlib.sha256(
                                _json([batch["source_id"], row["id"]]).encode()).hexdigest(), "session_id": row["session_id"],
                                "kind": kinds[0], "item_id": row["item_id"], "occurred_at": row["occurred_at"], "parent_id": row["viewed_event_id"],
                                "payload": {"signal": "watch", "provenance": "confirmed_delta_v1",
                                            "watched_s_delta": min(delta, elapsed * row["playback_rate"]), "started_at": previous["occurred_at"]}}
                            outcome_id = _insert_event(conn, _event(conn, event, internal=True, kinds=kinds))
                            outcomes += 1
                conn.execute("RELEASE watch_row")
            except ContractError as exc:
                conn.execute("ROLLBACK TO watch_row")
                conn.execute("RELEASE watch_row")
                reason, status = str(exc), "quarantined"
                quarantined += 1
            conn.execute("INSERT INTO rec_watch_steps VALUES(?,?,?,?,?,?,?,?,?,?,?)", (batch["source_id"], row["id"], batch["capture_id"],
                         row["previous_event_id"], _json(row), status, reason, outcome_id,
                         row["stream_session_id"], row["item_id"], row["occurred_at"]))
        conn.execute("INSERT INTO rec_watch_capture_imports VALUES(?,?,?,?,?,?,?)", (batch["source_id"], batch["capture_id"],
            batch["received_at"], time.time(), content_hash, "quarantined" if quarantined or batch["status"] == "quarantined" else "imported", batch["reason"]))
    return {"status": "imported", "outcomes": outcomes, "quarantined": quarantined}


def _same_session(a, b):
    return a["session_id"] == b["session_id"] or (a["canonical_session_id"] is not None and
                                                a["canonical_session_id"] == b["canonical_session_id"])


def _request(request):
    required = set(_REQUEST_FIELDS) - {"config_json", "config_hash", "arms_json", "canonical_session_id", "session_mapping_revision"} | {"config", "arms"}
    _keys(request, required, {"config_hash"})
    req = dict(request)
    for k in required - {"schema_version", "created_at", "preference_cutoff", "seed", "config", "arms", "experiment_id"}:
        _text(req[k])
    _require(type(req["schema_version"]) is int and req["schema_version"] == SCHEMA_VERSION, "schema_unavailable")
    _number(req["created_at"])
    _number(req["preference_cutoff"])
    _require(req["preference_cutoff"] <= req["created_at"], "future_preference")
    _integer(req["seed"])
    if req["experiment_id"] is not None:
        _text(req["experiment_id"])
    _require(isinstance(req["config"], Mapping) and isinstance(req["arms"], Mapping))
    for key in ("finished_ratio", "abandon_ratio", "rating_strength", "dislike_min_watch_seconds"):
        if key in req["config"]:
            _number(req["config"][key])
            if key in ("finished_ratio", "abandon_ratio"):
                _require(req["config"][key] <= 1, "invalid_verdict_config")
    _require(set(req["arms"]) <= {"base", "cand"}, "invalid_arms")
    _require(req["experiment_id"] is None and not req["arms"] or
             req["experiment_id"] is not None and set(req["arms"]) == {"base", "cand"}, "invalid_arms")
    _require(all(isinstance(v, Mapping) for v in req["arms"].values()), "invalid_arms")
    req["config_json"] = _json(req.pop("config"))
    req["arms_json"] = _json(req.pop("arms"))
    digest = hashlib.sha256(req["config_json"].encode()).hexdigest()
    _require("config_hash" not in req or req["config_hash"] == digest, "config_hash_conflict")
    req["config_hash"] = digest
    return req


def _item(item, req, kinds):
    _keys(item, {"kind", "id", "source_rank", "score", "arm", "control", "explore", "category", "duration_s", "trace"})
    kind, iid = _identity(item["kind"], item["id"], kinds)
    _integer(item["source_rank"])
    _number(item["score"], minimum=-float("inf"))
    _number(item["duration_s"])
    _text(item["category"])
    _require(type(item["control"]) is bool and type(item["explore"]) is bool)
    _require(item["arm"] in ("", "base", "cand"), "invalid_arm")
    _require(not item["arm"] or req["experiment_id"] is not None, "arm_without_experiment")
    _require(not (item["control"] or item["explore"]) or not item["arm"], "invalid_arm")
    _require(isinstance(item["trace"], Mapping))
    return kind, iid, _json({k: v for k, v in item.items() if k not in ("kind", "id")})


def record_served(db_path: str, *, request: Mapping[str, Any],
                  items: Sequence[Mapping[str, Any]], eligibility_snapshots=None,
                  kinds=DEFAULT_KINDS) -> dict[tuple[str, int], str]:
    req = _request(request)
    roles = {} if eligibility_snapshots is None else eligibility_snapshots
    _keys(roles, set(), {"generation", "delivery"})
    roles = {role: _snapshot_id(value) for role, value in roles.items() if value is not None}
    _require(isinstance(items, Sequence) and not isinstance(items, (str, bytes)) and len(items) <= 10000,
             "invalid_items")
    rows = [_item(it, req, kinds) for it in items]
    _require(len(rows) <= 10000 and len({(k, i) for k, i, _ in rows}) == len(rows), "duplicate_item")
    for it in items:
        for role, field in (("generation", "ranking_provenance"), ("delivery", "delivery_context")):
            context = it["trace"].get(field)
            if isinstance(context, Mapping) and "eligible_ids" in context:
                spec = context["eligible_ids"]
                if isinstance(spec, Mapping) and "snapshot_id" in spec:
                    _keys(spec, {"snapshot_id"})
                    _require(roles.get(role) == _snapshot_id(spec["snapshot_id"]), "eligibility_binding_conflict")
                else:
                    _require(role not in roles, "eligibility_binding_conflict")
    config_spec = request["config"].get("eligibility")
    if isinstance(config_spec, str):
        try:
            config_spec = json.loads(config_spec)
        except ValueError:
            config_spec = None
    if isinstance(config_spec, Mapping) and "snapshot_id" in config_spec:
        _keys(config_spec, {"snapshot_id"})
        _require(roles.get("delivery") == _snapshot_id(config_spec["snapshot_id"]), "eligibility_binding_conflict")
    with _connection(db_path, write=True) as conn:
        _require(req["created_at"] >= conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0], "before_cutover")
        snapshots = {identity: _validated_snapshot(conn.execute(
            "SELECT * FROM rec_eligibility_snapshots WHERE snapshot_id=?", (identity,)).fetchone())
            for identity in set(roles.values())}
        for role, identity in roles.items():
            snapshot = snapshots[identity]
            cutoff = req["preference_cutoff"] if role == "generation" else req["created_at"]
            _require(snapshot["metadata"]["created_at"] <= cutoff, "eligibility_snapshot_future")
            allowed = {kind: set(ids) for kind, ids in snapshot["eligible_ids"].items()}
            _require(all(i in allowed.get(k, ()) for k, i, _payload in rows), "eligibility_membership_conflict")
        req["canonical_session_id"], req["session_mapping_revision"] = _resolve_session(req["session_id"], _session_mappings(conn))
        existing = conn.execute("SELECT * FROM rec_requests WHERE request_id=? OR (session_id=? AND client_request_id=?)",
                                (req["request_id"], req["session_id"], req["client_request_id"])).fetchall()
        if existing:
            _require(len(existing) == 1 and dict(existing[0]) == req, "request_conflict")
            _require(_request_eligibility(conn, req["request_id"]) == roles, "eligibility_binding_conflict")
            old_items = conn.execute("SELECT kind,item_id,payload_json,event_id FROM rec_events WHERE request_id=? AND event_type='served' ORDER BY rowid",
                                     (req["request_id"],)).fetchall()
            _require([(r[0], r[1], r[2]) for r in old_items] == rows, "request_items_conflict")
            return {(r[0], r[1]): r[3] for r in old_items}
        content = conn.execute("SELECT * FROM rec_requests WHERE ranking_content_id=? LIMIT 1", (req["ranking_content_id"],)).fetchone()
        if content:
            _require(_request_eligibility(conn, content["request_id"]) == roles, "cached_content_conflict")
            binding_fields = set(_REQUEST_FIELDS) - {"request_id", "client_request_id", "created_at", "surface"}
            _require(all(content[k] == req[k] for k in binding_fields), "cached_content_conflict")
            stored = conn.execute("SELECT kind,item_id,payload_json FROM rec_events WHERE request_id=? AND event_type='served' ORDER BY rowid",
                                  (content["request_id"],)).fetchall()
            _require([tuple(r) for r in stored] == rows, "cached_content_conflict")
        if req["experiment_id"] is not None:
            prior = conn.execute("SELECT * FROM rec_requests WHERE experiment_id=? LIMIT 1", (req["experiment_id"],)).fetchone()
            _require(prior is None or all(prior[k] == req[k] for k in
                     ("config_hash", "arms_json", "ranker_revision", "eligibility_revision")), "experiment_conflict")
            _require(prior is None or _request_eligibility(conn, prior["request_id"]).get("generation") == roles.get("generation"),
                     "experiment_conflict")
        columns = ",".join(_REQUEST_FIELDS)
        conn.execute(f"INSERT INTO rec_requests({columns}) VALUES({','.join('?' for _ in _REQUEST_FIELDS)})",
                     tuple(req[k] for k in _REQUEST_FIELDS))
        if roles:
            conn.executemany("INSERT INTO rec_request_eligibility VALUES(?,?,?)",
                             [(req["request_id"], role, identity) for role, identity in roles.items()])
        refs = {}
        for kind, iid, payload in rows:
            event = dict(event_id=str(uuid.uuid4()), event_type="served", occurred_at=req["created_at"],
                         session_id=req["session_id"], kind=kind, item_id=iid, request_id=req["request_id"],
                         parent_id=None, corrects_id=None, source="engine",
                         source_event_id=_json([req["request_id"], kind, iid]), payload_json=payload,
                         canonical_session_id=req["canonical_session_id"], session_mapping_revision=req["session_mapping_revision"])
            refs[kind, iid] = _insert_event(conn, event)
        return refs


def _insert_event(conn, event):
    submitted = _json(event)
    receipt = conn.execute("SELECT * FROM rec_event_receipts WHERE source=? AND source_event_id=?",
                           (event["source"], event["source_event_id"])).fetchone()
    if receipt:
        _require(receipt["submitted_json"] == submitted, "event_conflict")
        return receipt["event_id"]
    old = conn.execute("SELECT * FROM rec_events WHERE event_id=? OR (source=? AND source_event_id=?)",
                       (event["event_id"], event["source"], event["source_event_id"])).fetchall()
    if old:
        _require(len(old) == 1 and all(old[0][k] == event[k] for k in _EVENT_FIELDS), "event_conflict")
        return old[0]["event_id"]
    if event["event_type"] == "viewed":
        old = conn.execute("SELECT event_id FROM rec_events WHERE event_type='viewed' AND parent_id=?",
                           (event["parent_id"],)).fetchone()
        if old:
            # Recreated observers cannot add another trial for the same delivery.
            conn.execute("INSERT INTO rec_event_receipts VALUES(?,?,?,?)",
                         (event["source"], event["source_event_id"], old[0], submitted))
            return old[0]
    if event["corrects_id"]:
        _require(not conn.execute("SELECT 1 FROM rec_events WHERE corrects_id=?", (event["corrects_id"],)).fetchone(),
                 "already_corrected")
    fields = ",".join(_EVENT_FIELDS)
    conn.execute(f"INSERT INTO rec_events({fields},received_at) VALUES({','.join('?' for _ in _EVENT_FIELDS)},?)",
                 tuple(event[k] for k in _EVENT_FIELDS) + (time.time(),))
    conn.execute("INSERT INTO rec_event_receipts VALUES(?,?,?,?)",
                 (event["source"], event["source_event_id"], event["event_id"], submitted))
    return event["event_id"]


def _event(conn, event, *, internal=False, kinds=DEFAULT_KINDS):
    _keys(event, {"event_type", "source", "source_event_id", "session_id", "kind", "item_id", "occurred_at", "payload"},
          {"event_id", "parent_id", "request_id", "corrects_id"})
    ev = dict(event)
    _require(ev["event_type"] in ("viewed", "outcome"), "invalid_event_type")
    for key in ("source", "source_event_id", "session_id"):
        _text(ev[key])
    ev["canonical_session_id"], ev["session_mapping_revision"] = _resolve_session(ev["session_id"], _session_mappings(conn))
    _require(internal or ev["source"] not in ("engine", "feedback", "sync_capture"), "reserved_source")
    _identity(ev["kind"], ev["item_id"], kinds)
    _number(ev["occurred_at"])
    _require(ev["occurred_at"] >= conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0], "before_cutover")
    ev.setdefault("event_id", str(uuid.uuid5(uuid.NAMESPACE_URL, _json([ev["source"], ev["source_event_id"]]))))
    _text(ev["event_id"])
    for key in ("parent_id", "request_id", "corrects_id"):
        ev.setdefault(key, None)
        if ev[key] is not None:
            _text(ev[key])
    parent = None
    if ev["parent_id"]:
        parent = conn.execute("SELECT * FROM rec_events WHERE event_id=?", (ev["parent_id"],)).fetchone()
        _require(parent is not None and parent["event_type"] in ("served", "viewed"), "invalid_parent")
        _require(_same_session(parent, ev) and all(parent[k] == ev[k] for k in ("kind", "item_id")), "parent_identity_mismatch")
        _require(ev["occurred_at"] >= parent["occurred_at"], "before_parent")
        _require(ev["request_id"] is None or ev["request_id"] == parent["request_id"], "parent_request_mismatch")
        ev["request_id"] = parent["request_id"]
    if ev["request_id"]:
        req = conn.execute("SELECT * FROM rec_requests WHERE request_id=?", (ev["request_id"],)).fetchone()
        _require(req is not None and _same_session(req, ev), "request_identity_mismatch")
    p = ev["payload"]
    if ev["event_type"] == "viewed":
        _require(parent is not None and parent["event_type"] == "served" and ev["corrects_id"] is None, "invalid_parent")
        _keys(p, {"visible_fraction", "dwell_ms", "foreground", "display_rank", "placement"})
        _number(p["visible_fraction"])
        _number(p["dwell_ms"])
        _require(.6 <= p["visible_fraction"] <= 1 and p["dwell_ms"] >= 1200 and p["foreground"] is True,
                 "unqualified_view")
        _integer(p["display_rank"])
        _text(p["placement"])
    else:
        _require(isinstance(p, Mapping), "invalid_outcome")
        signal = p.get("signal")
        fields = {"watch": {"watched_s_delta", "started_at"}, "rating": {"rating_before", "rating_after"},
                  "engagement": {"engagement_delta"}, "correction": set()}
        _require(isinstance(signal, str) and signal in fields, "invalid_signal")
        _keys(p, {"signal", "provenance"} | fields[signal])
        _require(p["provenance"] in ("confirmed_delta_v1", "unknown"), "invalid_provenance")
        if signal == "watch":
            _require(ev["kind"] == kinds[0], "watch_kind_invalid")
            _number(p["watched_s_delta"])
            _number(p["started_at"])
            _require(p["watched_s_delta"] > 0 and p["started_at"] <= ev["occurred_at"], "invalid_delta")
        elif signal == "rating":
            _rating(p["rating_before"])
            _rating(p["rating_after"])
            _require(p["rating_before"] != p["rating_after"], "unchanged_rating")
        elif signal == "engagement":
            _require(type(p["engagement_delta"]) is int and p["engagement_delta"] == 1, "invalid_delta")
        if signal == "correction":
            original = conn.execute("SELECT * FROM rec_events WHERE event_id=?", (ev["corrects_id"],)).fetchone()
            _require(original is not None and original["event_type"] == "outcome" and original["corrects_id"] is None,
                     "invalid_correction")
            _require(_same_session(original, ev) and all(original[k] == ev[k] for k in ("kind", "item_id")), "correction_identity_mismatch")
            _require(ev["occurred_at"] >= original["occurred_at"], "before_original")
            _require(ev["parent_id"] is None or ev["parent_id"] == original["parent_id"], "correction_parent_mismatch")
            ev["parent_id"], ev["request_id"] = original["parent_id"], original["request_id"]
            for key in ("session_id", "canonical_session_id", "session_mapping_revision"):
                ev[key] = original[key]
        else:
            _require(ev["corrects_id"] is None, "invalid_correction")
    ev["payload_json"] = _json(ev.pop("payload"))
    return ev


def record_event(db_path: str, *, event: Mapping[str, Any], kinds=DEFAULT_KINDS) -> str:
    with _connection(db_path, write=True) as conn:
        return _insert_event(conn, _event(conn, event, kinds=kinds))


def begin_sync_capture(db_path: str, *, capture_id: str, source_id: str,
                       source_revision: str, event_ids: Sequence[str]) -> dict[str, Any]:
    """Reserve source identities BEFORE invoking the host's sync. This is not commit proof.

    A process crash leaves claims reserved; retrying the HTTP mutation never makes
    their eventual effects new evidence. This command does not open the host
    database or call the host. Only a verified source adapter may use this
    interface. A prior pending/quarantined attempt is a source-wide gap;
    later unique IDs cannot turn uncertain cumulative history into fresh credit.
    """
    for value in (capture_id, source_id, source_revision):
        _text(value)
    _require(isinstance(event_ids, Sequence) and not isinstance(event_ids, (str, bytes)) and 0 < len(event_ids) <= 10000,
             "invalid_source_ids")
    ids = [_text(i) for i in event_ids]
    serialized = _json(ids)
    with _connection(db_path, write=True) as conn:
        old = conn.execute("SELECT * FROM rec_sync_captures WHERE capture_id=?", (capture_id,)).fetchone()
        if old:
            _require(old["source_id"] == source_id and old["source_revision"] == source_revision and old["event_ids_json"] == serialized,
                     "capture_conflict")
            result = conn.execute("SELECT result_json FROM rec_sync_results WHERE capture_id=?", (capture_id,)).fetchone()
            return json.loads(result[0]) if result else {"capture_id": capture_id, "status": "indeterminate", "reason": "capture_already_started"}
        now = time.time()
        cutover = conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0]
        _require(now >= cutover, "before_cutover")
        token = str(uuid.uuid4())
        prior_gap = bool(conn.execute("""SELECT 1 FROM rec_sync_captures c
            LEFT JOIN rec_sync_results r ON r.capture_id=c.capture_id
            WHERE c.source_id=? AND (r.status IS NULL OR r.status='quarantined') LIMIT 1""", (source_id,)).fetchone())
        conn.execute("INSERT INTO rec_sync_captures VALUES(?,?,?,?,?,?)", (capture_id, source_id, source_revision, serialized, token, now))
        replay = len(ids) != len(set(ids))
        for event_id in dict.fromkeys(ids):
            inserted = conn.execute("INSERT OR IGNORE INTO rec_sync_claims VALUES(?,?,?)", (source_id, event_id, capture_id))
            replay = replay or inserted.rowcount != 1
        if replay:
            result = {"capture_id": capture_id, "status": "quarantined", "reason": "source_event_replay", "event_ids": []}
            conn.execute("INSERT INTO rec_sync_results VALUES(?,?,?,?,?)", (capture_id, "quarantined", result["reason"], now, _json(result)))
            return result
        return {"capture_id": capture_id, "status": "planned", "owner_token": token, "cutover_ts": cutover,
                "received_at": now, "prior_gap": prior_gap}


def finish_sync_capture(db_path: str, *, capture_id: str, owner_token: str,
                        events: Sequence[Mapping[str, Any]] = (), reason: str | None = None,
                        kinds=DEFAULT_KINDS) -> dict[str, Any]:
    """Atomic capture result plus immutable deltas, called only after proven commit.

    The caller must hold the host's global sync lock across before/after reads
    and the upstream handler. Any uncertain commit/identity/interval yields reason,
    no events. An unlinked valid delta remains unlinked; this never fabricates views.
    """
    _text(capture_id)
    _text(owner_token)
    if reason is not None:
        _text(reason)
        _require(not events, "quarantine_with_events")
    with _connection(db_path, write=True) as conn:
        capture = conn.execute("SELECT * FROM rec_sync_captures WHERE capture_id=?", (capture_id,)).fetchone()
        _require(capture is not None and capture["owner_token"] == owner_token, "stale_capture_owner")
        prior = conn.execute("SELECT result_json FROM rec_sync_results WHERE capture_id=?", (capture_id,)).fetchone()
        if prior:
            return json.loads(prior[0])
        ids = []
        scopes = set()
        for event in events:
            _require(event.get("source") == "sync_capture" and event.get("event_type") == "outcome" and
                     event.get("payload", {}).get("signal") == "watch" and event.get("payload", {}).get("provenance") == "confirmed_delta_v1",
                     "invalid_capture_event")
            normalized = _event(conn, event, internal=True, kinds=kinds)
            scope = (normalized["canonical_session_id"], normalized["kind"], normalized["item_id"])
            _require(scope[0] is not None and scope not in scopes, "capture_scope_unproven")
            scopes.add(scope)
            start = event["payload"]["started_at"]
            cutover = conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0]
            _require(start >= cutover and event["occurred_at"] <= capture["received_at"], "capture_time_unproven")
            last = conn.execute("""SELECT max(occurred_at) FROM rec_events WHERE source='sync_capture'
                AND canonical_session_id=? AND kind=? AND item_id=?""", scope).fetchone()[0]
            _require(last is None or start >= last, "capture_interval_overlap")
            ids.append(_insert_event(conn, normalized))
        status = "quarantined" if reason else "captured"
        result = {"capture_id": capture_id, "status": status, "reason": reason, "event_ids": ids}
        conn.execute("INSERT INTO rec_sync_results VALUES(?,?,?,?,?)", (capture_id, status, reason, time.time(), _json(result)))
        return result


def attribute_outcomes(db_path: str, *, through_ts: float, window_s: float,
                       policy_revision: str) -> int:
    """Explicit evaluation cutoff, NOT stream completeness. Only explicit references
    finalize. No source currently supplies trusted view AND outcome completeness
    watermarks, so chronological fallback stays provisional regardless of poll times.
    Corrections inherit their original owner; they never select another exposure.
    """
    _number(through_ts)
    _number(window_s)
    _require(window_s > 0, "invalid_window")
    _text(policy_revision)
    with _connection(db_path, write=True) as conn:
        runs = conn.execute("SELECT * FROM rec_attribution_runs ORDER BY through_ts").fetchall()
        last = runs[-1] if runs else None
        if last:
            _require(last["window_s"] == window_s and last["policy_revision"] == policy_revision, "policy_conflict")
            _require(through_ts >= last["through_ts"], "cutoff_regression")
            if through_ts == last["through_ts"]:
                return 0
        rows = [dict(r) for r in conn.execute("SELECT rowid AS seq,* FROM rec_events WHERE occurred_at<=? AND received_at<=? ORDER BY occurred_at,rowid",
                                              (through_ts, through_ts))]
        by_id = {r["event_id"]: r for r in rows}
        _watch_bound_rows(conn, by_id, {}, through_ts)
        mappings = _session_mappings(conn, through_ts)
        claims = {r["outcome_id"]: r["viewed_id"] for r in conn.execute("SELECT * FROM rec_attributions")}
        added = 0
        for row in rows:
            if row["event_type"] != "outcome" or row["event_id"] in claims:
                continue
            chosen, _ = _outcome_owner(row, by_id, claims, mappings, window_s)
            if chosen is not None and chosen["occurred_at"] + window_s <= through_ts:
                conn.execute("INSERT INTO rec_attributions VALUES(?,?,?,?,?)",
                             (row["event_id"], chosen["event_id"], policy_revision, time.time(), through_ts))
                claims[row["event_id"]] = chosen["event_id"]
                added += 1
        seq = conn.execute("SELECT coalesce(max(rowid),0) FROM rec_events").fetchone()[0]
        conn.execute("INSERT INTO rec_attribution_runs VALUES(?,?,?,?,?)", (through_ts, window_s, policy_revision, time.time(), seq))
        return added


ATTRIBUTION_WINDOW_S = 3600.0
ATTRIBUTION_POLICY_REVISION = "production-explicit-w3600-v1"
ATTRIBUTION_MIN_ADVANCE_S = 60.0


def advance_attribution(db_path: str, *, now: float | None = None) -> int:
    """Advance the one pinned production attribution policy to a monotone cutoff.

    Called best-effort by the mutation boundaries that add confirmed outcomes
    (native watch sync, feedback). Attributions are derived state: a skipped or
    failed advance is recovered in full by the next one, so callers must never
    fail their own mutation on this, and read paths never call it. The rate
    limit keeps steady watch progress from flooding rec_attribution_runs; a
    clock that moved backwards is absorbed by the monotone cutoff.
    """
    now = time.time() if now is None else now
    _number(now)
    with _connection(db_path) as conn:
        last = conn.execute("SELECT max(through_ts) FROM rec_attribution_runs").fetchone()[0]
    if last is not None and now < last + ATTRIBUTION_MIN_ADVANCE_S:
        return 0
    return attribute_outcomes(db_path, through_ts=max(now, last or now),
                              window_s=ATTRIBUTION_WINDOW_S,
                              policy_revision=ATTRIBUTION_POLICY_REVISION)


def _outcome_owner(row, by_id, claims, mappings, window_s):
    p = row.get("payload") or json.loads(row["payload_json"])
    if p["provenance"] != "confirmed_delta_v1":
        return None, "unknown_source_provenance"
    if row["corrects_id"]:
        owner = by_id.get(claims.get(row["corrects_id"]))
        return owner, None if owner else "original_unresolved"
    if not _mapping_valid(row, mappings):
        return None, "session_mapping_changed" if row["canonical_session_id"] else "session_mapping_unresolved"
    if not row["parent_id"]:
        return None, "source_completeness_unavailable"
    parent = by_id.get(row["parent_id"])
    if parent is None:
        return None, "explicit_view_missing"
    if parent["event_type"] == "served":
        parent = next((v for v in by_id.values() if v["event_type"] == "viewed" and v["parent_id"] == parent["event_id"]), None)
    if parent is None or parent["event_type"] != "viewed":
        return None, "explicit_view_missing"
    served = by_id.get(parent["parent_id"])
    if not served or not _mapping_valid(parent, mappings) or not _mapping_valid(served, mappings):
        return None, "session_mapping_changed"
    if parent["canonical_session_id"] != row["canonical_session_id"] or any(parent[k] != row[k] for k in ("kind", "item_id")):
        return None, "parent_identity_mismatch"
    if not parent["occurred_at"] <= row["occurred_at"] <= parent["occurred_at"] + window_s:
        return None, "outside_attribution_window"
    if p["signal"] == "watch" and p["started_at"] < parent["occurred_at"]:
        return None, "pre_exposure_watch"
    return parent, None


def _unavailable(code):
    return {"status": "unavailable", "valid": False, "validity_reasons": [code],
            "promotion_enabled": False, "requests": [], "events": [], "attributions": [],
            "sessions": {}, "counts": {}, "watch_capture_supported": False,
            "promotion_reasons": ["watch_capture_unavailable"]}


def read_evidence(db_path: str, *, since_ts: float, through_ts: float,
                  experiment_id: str | None = None, include_eligibility_snapshots=False) -> dict[str, Any]:
    _number(since_ts)
    _number(through_ts)
    _require(since_ts <= through_ts, "invalid_window")
    _require(type(include_eligibility_snapshots) is bool, "invalid_snapshot_export")
    if experiment_id is not None:
        _text(experiment_id)
    try:
        with _connection(db_path) as conn:
            meta = dict(conn.execute("SELECT * FROM rec_metadata").fetchone())
            requests = {r["request_id"]: dict(r) for r in conn.execute("SELECT * FROM rec_requests WHERE created_at<=?", (through_ts,))}
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rec_request_eligibility'").fetchone():
                for binding in conn.execute("SELECT e.* FROM rec_request_eligibility e JOIN rec_requests r "
                                            "USING(request_id) WHERE r.created_at<=?", (through_ts,)):
                    request = requests[binding["request_id"]]
                    request.setdefault("eligibility_snapshots", {"generation": None, "delivery": None})[binding["role"]] = binding["snapshot_id"]
            events = [dict(r) for r in conn.execute("SELECT rowid AS seq,* FROM rec_events WHERE occurred_at<=? AND received_at<=? ORDER BY occurred_at,rowid",
                                                   (through_ts, through_ts))]
            runs = [dict(r) for r in conn.execute("SELECT * FROM rec_attribution_runs WHERE through_ts<=? ORDER BY through_ts", (through_ts,))]
            all_claims = [dict(r) for r in conn.execute("SELECT * FROM rec_attributions WHERE through_ts<=?", (through_ts,))]
            mappings = _session_mappings(conn, through_ts)
            _watch_bound_rows(conn, {r["event_id"]: r for r in events}, requests, through_ts)
            watch_receipts = {r[0]: r[1] for r in conn.execute("""SELECT status,count(*) FROM rec_watch_capture_imports
                WHERE imported_at<=? GROUP BY status""", (through_ts,))}
            captured_outcomes = conn.execute("""SELECT count(*) FROM rec_watch_steps s JOIN rec_watch_capture_imports i
                ON s.source_id=i.source_id AND s.capture_id=i.capture_id WHERE s.outcome_id IS NOT NULL AND i.imported_at<=?""",
                (through_ts,)).fetchone()[0]
            watch_quarantined_views = {r[0] for r in conn.execute("""SELECT json_extract(s.payload_json,'$.viewed_event_id')
                FROM rec_watch_steps s JOIN rec_watch_capture_imports i ON s.source_id=i.source_id AND s.capture_id=i.capture_id
                WHERE s.status='quarantined' AND i.imported_at<=?""", (through_ts,))}
            unresolved = conn.execute("""SELECT count(*) FROM feedback_operations o JOIN feedback_steps s
                ON s.step_id=(SELECT max(step_id) FROM feedback_steps WHERE operation_id=o.operation_id AND ts<=?)
                WHERE o.created_at<=? AND s.state NOT IN ('confirmed','conflict')""",
                (through_ts, through_ts)).fetchone()[0]
            sync_unresolved = conn.execute("""SELECT count(*) FROM rec_sync_captures c WHERE c.received_at<=?
                AND NOT EXISTS(SELECT 1 FROM rec_sync_results r WHERE r.capture_id=c.capture_id AND r.completed_at<=?)""",
                (through_ts, through_ts)).fetchone()[0]
            sync_results = {r[0]: r[1] for r in conn.execute("""SELECT status,count(*) FROM rec_sync_results
                WHERE completed_at>=? AND completed_at<=? GROUP BY status""", (since_ts, through_ts))}
            sync_gap = conn.execute("SELECT 1 FROM rec_sync_results WHERE status='quarantined' AND completed_at<=? LIMIT 1",
                                    (through_ts,)).fetchone() is not None
        for req in requests.values():
            req["config"] = json.loads(req.pop("config_json"))
            req["arms"] = json.loads(req.pop("arms_json"))
            req["identity_valid"] = _mapping_valid(req, mappings)
        for event in events:
            event["payload"] = json.loads(event.pop("payload_json"))
            event["identity_valid"] = _mapping_valid(event, mappings)
        by_id = {r["event_id"]: r for r in events}
        selected = {r["event_id"] for r in events if r["occurred_at"] >= since_ts and
                    (experiment_id is None or requests.get(r["request_id"], {}).get("experiment_id") == experiment_id)}
        selected_views = {eid for eid in selected if by_id[eid]["event_type"] == "viewed"}
        cohort_items = {(by_id[eid]["canonical_session_id"] or by_id[eid]["session_id"], by_id[eid]["kind"], by_id[eid]["item_id"]) for eid in selected_views}
        selected.update(r["event_id"] for r in events if r["event_type"] == "outcome" and r["occurred_at"] >= since_ts
                        and (r["canonical_session_id"] or r["session_id"], r["kind"], r["item_id"]) in cohort_items)
        claims = [r for r in all_claims if r["viewed_id"] in selected_views and r["outcome_id"] in by_id]
        selected.update(r["outcome_id"] for r in claims)
        pending = list(selected)
        while pending:
            row = by_id[pending.pop()]
            for ref in (row["parent_id"], row["corrects_id"]):
                if ref in by_id and ref not in selected:
                    selected.add(ref)
                    pending.append(ref)
        kept = [r for r in events if r["event_id"] in selected]
        request_ids = {r["request_id"] for r in kept}
        exported = {}
        if include_eligibility_snapshots:
            for rid in request_ids:
                for role, identity in requests.get(rid, {}).get("eligibility_snapshots", {}).items():
                    if identity is not None:
                        if identity not in exported:
                            exported[identity] = read_eligibility_snapshot(db_path, snapshot_id=identity)
                        body = exported[identity]
                        cutoff = requests[rid]["preference_cutoff"] if role == "generation" else requests[rid]["created_at"]
                        _require(body["metadata"]["created_at"] <= cutoff, "eligibility_snapshot_future")
        reasons = []
        if not selected_views:
            reasons.append("no_qualified_views")
        if not runs or runs[-1]["through_ts"] < through_ts:
            reasons.append("attribution_pending")
        if any(r["event_type"] == "outcome" and r["payload"]["provenance"] != "confirmed_delta_v1" for r in kept):
            reasons.append("unknown_source_provenance")
        if any(not r["identity_valid"] and r["canonical_session_id"] is not None for r in kept):
            reasons.append("session_mapping_changed")
        if any(r["canonical_session_id"] is None for r in kept):
            reasons.append("session_mapping_unresolved")
        claim_map = {r["outcome_id"]: r["viewed_id"] for r in all_claims}
        provisional = []
        if runs:
            for row in kept:
                if row["event_type"] != "outcome" or row["event_id"] in claim_map:
                    continue
                owner, reason = _outcome_owner(row, by_id, claim_map, mappings, runs[-1]["window_s"])
                reason = reason or "attribution_pending"
                provisional.append({"outcome_id": row["event_id"], "reason": reason})
                if reason not in ("outside_attribution_window", "pre_exposure_watch") and reason not in reasons:
                    reasons.append(reason)
        if unresolved:
            reasons.append("feedback_unresolved")
        if sync_unresolved:
            reasons.append("sync_capture_unresolved")
        if sync_gap:
            reasons.append("sync_capture_gap")
        if watch_quarantined_views & selected_views:
            reasons.append("watch_capture_quarantined")
        sessions = {}
        viewed_ids = [r["event_id"] for r in events if r["event_id"] in selected_views]
        for eid in viewed_ids:
            row = by_id[eid]
            sessions.setdefault(row["canonical_session_id"] or row["session_id"], []).append(eid)
        return {"status": "ok", "valid": not reasons, "validity_reasons": reasons,
                "promotion_enabled": False, "promotion_reasons": ["automatic_promotion_disabled", "session_evidence_policy_required"] +
                    ([] if captured_outcomes else ["watch_capture_unavailable"]),
                "watch_capture_supported": bool(captured_outcomes), "watch_capture_receipts": watch_receipts,
                "source_completeness": {"views": "unavailable", "outcomes": "unavailable"},
                "sync_capture": {"results": sync_results, "unresolved": sync_unresolved, "deployment_verified": False},
                "provisional": provisional,
                "metadata": meta, "since_ts": since_ts, "through_ts": through_ts,
                "requests": [r for k, r in requests.items() if k in request_ids],
                "events": kept, "attributions": claims, "sessions": sessions, "viewed_ids": viewed_ids,
                "attribution_run": runs[-1] if runs else None,
                **({"eligibility_snapshots": exported} if include_eligibility_snapshots else {})}
    except ContractError as exc:
        return _unavailable(str(exc))


def read_view_counts(db_path: str, *, since_ts: float, through_ts: float,
                     recommender: str | None = None) -> dict[str, Any]:
    """Counts distinct UTC days with qualified views, matching existing fatigue units."""
    _number(since_ts)
    _number(through_ts)
    _require(since_ts <= through_ts, "invalid_window")
    if recommender is not None:
        _text(recommender)
    try:
        with _connection(db_path) as conn:
            rows = conn.execute("""SELECT e.kind,e.item_id,count(DISTINCT CAST(e.occurred_at/86400 AS INTEGER)) AS n
                FROM rec_events e JOIN rec_requests r ON r.request_id=e.request_id
                WHERE e.event_type='viewed' AND e.occurred_at>=? AND e.occurred_at<=? AND e.received_at<=?
                  AND (? IS NULL OR r.recommender=?) GROUP BY e.kind,e.item_id""",
                (since_ts, through_ts, through_ts, recommender, recommender)).fetchall()
        return {"status": "ok", "counts": {(r[0], r[1]): r[2] for r in rows}}
    except ContractError as exc:
        return _unavailable(str(exc))


def read_capture_readiness(db_path: str) -> dict[str, Any]:
    """Cumulative proof of the watch capture path: committed receipts that produced
    confirmed outcomes, and how many of those outcomes attribution has finalized.
    Tracker configuration alone never counts; only ledger rows do."""
    try:
        with _connection(db_path) as conn:
            receipts = {r[0]: r[1] for r in conn.execute("SELECT status,count(*) FROM rec_watch_capture_imports GROUP BY status")}
            captured = conn.execute("SELECT count(*) FROM rec_watch_steps WHERE outcome_id IS NOT NULL").fetchone()[0]
            last_imported = conn.execute("SELECT max(imported_at) FROM rec_watch_capture_imports WHERE status='imported'").fetchone()[0]
            attributed = conn.execute("SELECT count(*) FROM rec_attributions").fetchone()[0]
            through = conn.execute("SELECT max(through_ts) FROM rec_attribution_runs").fetchone()[0]
        return {"status": "ok", "verified": captured > 0, "receipts": receipts, "captured_outcomes": captured,
                "last_imported_at": last_imported, "attributed_outcomes": attributed, "attribution_through_ts": through}
    except ContractError as exc:
        return {"status": "unavailable", "verified": False, "reason": str(exc)}


def summarize_trials(evidence: Mapping[str, Any], *, verdict: Callable,
                     trial_reward: Callable,
                     cumulative_at_cutoff: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Pure projection: shared cumulative verdict, ONLY fresh evidence as payment.

    cumulative_at_cutoff is {cutoff_ts, items:{(kind,id):{watched_s,duration_s,rating,
    engagement_count}}}, resolved by the owning engine at that exact cutoff. Missing/mismatched
    facts never fall back to a delta-based verdict or current profile. Corrections
    are folded before the ONE shared trial_reward call per exposure. Old explicit
    floors are excluded by passing only fresh rating/O inputs to that same function.
    """
    reasons = list(evidence["validity_reasons"])
    result = {"status": evidence["status"], "valid": False, "validity_reasons": reasons,
              "promotion_enabled": False, "watch_capture_supported": False,
              "promotion_reasons": list(evidence.get("promotion_reasons", ["watch_capture_unavailable"])),
              "trials": [], "sessions": {}}
    run = evidence.get("attribution_run")
    if evidence["status"] != "ok" or not run:
        return result
    cumulative = {}
    if cumulative_at_cutoff is not None:
        _keys(cumulative_at_cutoff, {"cutoff_ts", "items"})
        _number(cumulative_at_cutoff["cutoff_ts"])
        _require(cumulative_at_cutoff["cutoff_ts"] == evidence["through_ts"], "cumulative_cutoff_mismatch")
        _require(isinstance(cumulative_at_cutoff["items"], Mapping))
        cumulative = cumulative_at_cutoff["items"]
    events = {r["event_id"]: r for r in evidence["events"]}
    requests = {r["request_id"]: r for r in evidence["requests"]}
    grouped = {}
    for claim in evidence["attributions"]:
        grouped.setdefault(claim["viewed_id"], []).append(events[claim["outcome_id"]])
    selected_views = set(evidence["viewed_ids"])
    for event in evidence["events"]:
        if event["event_id"] not in selected_views:
            continue
        if event["occurred_at"] + run["window_s"] > run["through_ts"]:
            continue
        served = events[event["parent_id"]]
        req = requests[event["request_id"]]
        p = served["payload"]
        if p["arm"] not in ("base", "cand") or p["control"] or p["explore"] or req["experiment_id"] is None:
            continue
        cfg = req["config"]
        if not all(k in cfg for k in ("finished_ratio", "abandon_ratio")):
            reasons.append("verdict_config_missing")
            continue
        rows = sorted(grouped.get(event["event_id"], []), key=lambda r: (r["occurred_at"], r["seq"]))
        # An unassigned correction must also suppress confidence in its original
        # owner's effective reward; it never supplies a second paid signal.
        row_ids = {r["event_id"] for r in rows}
        pending_corrections = [r for r in evidence["events"] if r["corrects_id"] in row_ids and r not in rows]
        canceled = {r["corrects_id"] for r in rows if r["corrects_id"]}
        watched, rating, engagement_count = 0.0, None, 0
        for row in rows:
            if row["event_id"] in canceled or row["corrects_id"]:
                continue
            signal = row["payload"]
            if signal["signal"] == "watch":
                watched += signal["watched_s_delta"]
            elif signal["signal"] == "rating":
                rating = signal["rating_after"]
            elif signal["signal"] == "engagement":
                engagement_count += signal["engagement_delta"]
        kwargs = {k: cfg[k] for k in ("finished_ratio", "abandon_ratio", "rating_strength") if k in cfg}
        if "dislike_min_watch_seconds" in cfg:
            kwargs["dislike_min_watch"] = cfg["dislike_min_watch_seconds"]
        facts = cumulative.get((event["kind"], event["item_id"]))
        liked = disliked = reward = None
        trial_reasons = []
        if not event["identity_valid"] or not served["identity_valid"] or not req["identity_valid"] or any(not r["identity_valid"] for r in rows):
            trial_reasons.append("session_mapping_changed")
        if pending_corrections:
            trial_reasons.append("correction_pending")
        for pending in evidence["provisional"]:
            unresolved = events[pending["outcome_id"]]
            original = events.get(unresolved["corrects_id"], unresolved)
            if pending["reason"] in ("outside_attribution_window", "pre_exposure_watch"):
                continue
            if original["parent_id"] not in (None, event["event_id"], served["event_id"]):
                continue
            if (_same_session(original, event) and all(original[k] == event[k] for k in ("kind", "item_id"))
                    and event["occurred_at"] <= original["occurred_at"] <= event["occurred_at"] + run["window_s"]
                    and pending["reason"] not in trial_reasons):
                trial_reasons.append(pending["reason"])
        if facts is None:
            trial_reasons.append("cumulative_verdict_unavailable")
        else:
            _keys(facts, {"watched_s", "duration_s", "rating", "engagement_count"})
            _number(facts["watched_s"])
            _number(facts["duration_s"])
            _rating(facts["rating"])
            _integer(facts["engagement_count"])
            liked, disliked, _ = verdict(facts["watched_s"], facts["duration_s"], rating=facts["rating"], engagement_count=facts["engagement_count"], **kwargs)
        if not trial_reasons:
            reward = trial_reward(watched, rating=rating, engagement_count=engagement_count) if liked else 0.0
            _require(type(reward) in (int, float) and math.isfinite(reward) and 0 <= reward <= 1, "invalid_shared_reward")
        for reason in trial_reasons:
            if reason not in reasons:
                reasons.append(reason)
        trial = {"viewed_id": event["event_id"], "session_id": event["session_id"], "request_id": event["request_id"],
                 "experiment_id": req["experiment_id"], "kind": event["kind"], "item_id": event["item_id"],
                 "arm": p["arm"], "reward": reward, "watched_s_delta": watched,
                 "liked": liked, "disliked": disliked, "category": p["category"],
                 "canonical_session_id": event["canonical_session_id"], "validity_reasons": trial_reasons}
        result["trials"].append(trial)
        result["sessions"].setdefault(event["canonical_session_id"] or event["session_id"], []).append(trial)
    if not result["trials"]:
        reasons.append("no_eligible_trials")
    result["valid"] = not reasons
    return result


def _operation(operation, kinds):
    _keys(operation, {"operation_id", "kind", "item_id", "action", "session_id"},
          {"rating100", "undo_of", "request_id", "viewed_event_id"})
    op = dict(operation)
    _identity(op["kind"], op["item_id"], kinds)
    for key in ("operation_id", "session_id"):
        _text(op[key])
    _require(op["action"] in ("rating", "engagement", "undo"), "invalid_action")
    if op["action"] == "rating":
        _require("rating100" in op and "undo_of" not in op, "invalid_action_fields")
        _rating(op["rating100"])
    elif op["action"] == "undo":
        _require("undo_of" in op and "rating100" not in op, "invalid_action_fields")
        _text(op["undo_of"])
    else:
        _require("undo_of" not in op and "rating100" not in op, "invalid_action_fields")
    for k in ("request_id", "viewed_event_id"):
        op.setdefault(k, None)
        if op[k] is not None:
            _text(op[k])
    return op


def _current(value, status):
    _require(isinstance(value, Mapping) and value.get("status") == status, "authority_unavailable")
    _require("rating100" in value and "engagement_count" in value, "authority_unavailable")
    return {"rating100": _rating(value["rating100"]), "engagement_count": _integer(value["engagement_count"])}


def _step(conn, operation_id):
    row = conn.execute("SELECT * FROM feedback_steps WHERE operation_id=? ORDER BY step_id DESC LIMIT 1", (operation_id,)).fetchone()
    return {"state": row["state"], "facts": json.loads(row["facts_json"]), "owner_token": row["owner_token"]} if row else None


def _result(operation_id, step):
    return {"operation_id": operation_id, "status": step["state"], **step["facts"]}


def _owns(conn, op):
    return conn.execute("SELECT 1 FROM feedback_ownership WHERE kind=? AND item_id=? AND active_operation=? AND owner_token=?",
                        (op["kind"], op["item_id"], op["operation_id"], op["_owner_token"])).fetchone() is not None


def _stale(op):
    return {"operation_id": op["operation_id"], "status": "indeterminate", "reason": "stale_owner"}


def _append_step(conn, op, state, facts, *, rejected=False):
    if not rejected and not _owns(conn, op):
        return _stale(op)
    previous = _step(conn, op["operation_id"])
    _require(previous is None or previous["state"] not in ("confirmed", "conflict"), "terminal_operation")
    _require(not rejected or previous is None and state == "conflict" and facts.get("reason") == "item_owned", "invalid_rejection")
    conn.execute("INSERT INTO feedback_steps(operation_id,state,ts,facts_json,owner_token) VALUES(?,?,?,?,?)",
                 (op["operation_id"], state, time.time(), _json(facts), op["_owner_token"]))
    if not rejected and state in ("confirmed", "conflict"):
        changed = conn.execute("""UPDATE feedback_ownership SET active_operation=NULL,owner_token=NULL,
            latest_operation=CASE WHEN ?='confirmed' THEN ? ELSE latest_operation END
            WHERE kind=? AND item_id=? AND active_operation=? AND owner_token=?""",
            (state, op["operation_id"], op["kind"], op["item_id"], op["operation_id"], op["_owner_token"]))
        _require(changed.rowcount == 1, "ownership_lost")
    return {"operation_id": op["operation_id"], "status": state, **facts}


def _validate_operation_refs(conn, op):
    _require(time.time() >= conn.execute("SELECT cutover_ts FROM rec_metadata").fetchone()[0], "before_cutover")
    if op["request_id"]:
        row = conn.execute("SELECT * FROM rec_requests WHERE request_id=?", (op["request_id"],)).fetchone()
        canonical, _ = _resolve_session(op["session_id"], _session_mappings(conn))
        _require(row is not None and (row["session_id"] == op["session_id"] or canonical is not None and canonical == row["canonical_session_id"]), "request_identity_mismatch")
    if op["viewed_event_id"]:
        row = conn.execute("SELECT * FROM rec_events WHERE event_id=?", (op["viewed_event_id"],)).fetchone()
        _require(row is not None and row["event_type"] == "viewed", "invalid_view_reference")
        canonical, _ = _resolve_session(op["session_id"], _session_mappings(conn))
        _require(all(row[k] == op[k] for k in ("kind", "item_id")) and (row["session_id"] == op["session_id"] or
                 canonical is not None and canonical == row["canonical_session_id"]), "view_identity_mismatch")
        _require(op["request_id"] is None or op["request_id"] == row["request_id"], "view_request_mismatch")
        _require(time.time() >= row["occurred_at"], "before_view")


def _finish_feedback(conn, op, facts, after, *, provenance, kinds):
    if not _owns(conn, op):
        return _stale(op)
    signal = {"provenance": provenance}
    corrects_id = None
    if op["action"] == "undo":
        original = _step(conn, op["undo_of"])
        corrects_id = original["facts"].get("event_id")
        signal["signal"] = "correction"
    elif op["action"] == "rating":
        signal.update(signal="rating", rating_before=facts["before"]["rating100"], rating_after=after["rating100"])
    else:
        signal.update(signal="engagement", engagement_delta=1)
    event_id = None
    if op["action"] != "rating" or facts["before"]["rating100"] != after["rating100"]:
        if op["action"] != "undo" or corrects_id:
            original_event = conn.execute("SELECT * FROM rec_events WHERE event_id=?", (corrects_id,)).fetchone() if corrects_id else None
            ev = dict(event_type="outcome", source="feedback", source_event_id=op["operation_id"],
                      occurred_at=time.time(), session_id=original_event["session_id"] if original_event else op["session_id"],
                      kind=op["kind"], item_id=op["item_id"],
                      request_id=original_event["request_id"] if original_event else op["request_id"],
                      parent_id=original_event["parent_id"] if original_event else op["viewed_event_id"],
                      corrects_id=corrects_id, payload=signal)
            event_id = _insert_event(conn, _event(conn, ev, internal=True, kinds=kinds))
    return _append_step(conn, op, "confirmed", {**facts, "after": after, "event_id": event_id,
                                               "provenance": provenance, "dispatch_finished": True})


def perform_feedback(db_path: str, *, operation: Mapping[str, Any],
                     read_current: Callable, apply_change: Callable, kinds=DEFAULT_KINDS) -> dict[str, Any]:
    op = _operation(operation, kinds)
    key = (op["kind"], op["item_id"])
    op_id = op["operation_id"]
    serialized = _json(op)
    op["_owner_token"] = str(uuid.uuid4())
    with _connection(db_path, write=True) as conn:
        old = conn.execute("SELECT * FROM feedback_operations WHERE operation_id=?", (op_id,)).fetchone()
        if old:
            _require(old["operation_json"] == serialized, "operation_conflict")
            return _result(op_id, _step(conn, op_id))
        _validate_operation_refs(conn, op)
        conn.execute("INSERT OR IGNORE INTO feedback_ownership(kind,item_id) VALUES(?,?)", key)
        owner = conn.execute("SELECT * FROM feedback_ownership WHERE kind=? AND item_id=?", key).fetchone()
        if owner["active_operation"]:
            conn.execute("INSERT INTO feedback_operations VALUES(?,?,?,?,?,?,?)", (op_id, *key, serialized, os.getpid(), time.time(), op["_owner_token"]))
            return _append_step(conn, op, "conflict", {"reason": "item_owned", "dispatch_finished": True}, rejected=True)
        if op["action"] == "undo":
            original = conn.execute("SELECT * FROM feedback_operations WHERE operation_id=?", (op["undo_of"],)).fetchone()
            _require(original is not None and (original["kind"], original["item_id"]) == key, "invalid_undo")
            _require(json.loads(original["operation_json"])["action"] != "undo", "undo_of_undo")
            _require(_step(conn, op["undo_of"])["state"] == "confirmed", "unconfirmed_undo")
        conn.execute("INSERT INTO feedback_operations VALUES(?,?,?,?,?,?,?)", (op_id, *key, serialized, os.getpid(), time.time(), op["_owner_token"]))
        claimed = conn.execute("UPDATE feedback_ownership SET active_operation=?,owner_token=? WHERE kind=? AND item_id=? AND active_operation IS NULL",
                               (op_id, op["_owner_token"], *key))
        _require(claimed.rowcount == 1, "ownership_lost")
        _append_step(conn, op, "planned", {"dispatch_finished": False})
    try:
        before = _current(read_current(key), "ok")
    except Exception:
        with _connection(db_path, write=True) as conn:
            return _append_step(conn, op, "indeterminate", {"reason": "read_unavailable", "dispatch_finished": True})
    with _connection(db_path, write=True) as conn:
        owner = conn.execute("SELECT * FROM feedback_ownership WHERE kind=? AND item_id=?", key).fetchone()
        if not _owns(conn, op):
            return _stale(op)
        if op["action"] == "undo":
            original = _step(conn, op["undo_of"])["facts"]
            field = original["field"]
            if owner["latest_operation"] != op["undo_of"] or before[field] != original["after"][field]:
                return _append_step(conn, op, "conflict", {"reason": "undo_conflict", "dispatch_finished": True})
            expected = {**before, field: original["before"][field]}
        else:
            field = "rating100" if op["action"] == "rating" else "engagement_count"
            expected = {**before, field: op["rating100"] if field == "rating100" else before[field] + 1}
        if expected["engagement_count"] > 9223372036854775807:
            return _append_step(conn, op, "conflict", {"reason": "counter_limit", "dispatch_finished": True})
        change = {"action": "rating", "rating100": expected[field]} if field == "rating100" else {"action": "engagement", "delta": expected[field] - before[field]}
        facts = {"before": before, "expected": expected, "field": field, "change": change, "dispatch_finished": False}
        if before[field] == expected[field]:
            return _finish_feedback(conn, op, facts, before, provenance="confirmed_delta_v1", kinds=kinds)
        _append_step(conn, op, "sent", facts)
    try:
        response = apply_change(key, dict(change))
        conflict = isinstance(response, Mapping) and response.get("status") == "conflict"
        after = _current(response, "conflict" if conflict else "confirmed")
        confirmed = not conflict and after[field] == expected[field]
    except Exception:
        confirmed, conflict, after = False, False, None
    with _connection(db_path, write=True) as conn:
        if confirmed:
            return _finish_feedback(conn, op, facts, after, provenance="confirmed_delta_v1", kinds=kinds)
        if conflict:
            return _append_step(conn, op, "conflict", {**facts, "after": after, "reason": "authority_conflict", "dispatch_finished": True})
        return _append_step(conn, op, "indeterminate", {**facts, "reason": "write_unconfirmed", "dispatch_finished": True})


def reconcile_feedback(db_path: str, *, operation_id: str,
                       read_current: Callable, kinds=DEFAULT_KINDS) -> dict[str, Any]:
    """Read-only external reconciliation; never invokes an external mutation."""
    _text(operation_id)
    with _connection(db_path, write=True) as conn:
        row = conn.execute("SELECT * FROM feedback_operations WHERE operation_id=?", (operation_id,)).fetchone()
        _require(row is not None, "unknown_operation")
        op = json.loads(row["operation_json"])
        step = _step(conn, operation_id)
        if step["state"] in ("confirmed", "conflict"):
            return _result(operation_id, step)
        if not step["facts"].get("dispatch_finished"):
            running = True
            if os.name == "posix" and row["owner_pid"] != os.getpid():
                try:
                    os.kill(row["owner_pid"], 0)
                    running = True
                except ProcessLookupError:
                    running = False
                except (PermissionError, OSError):
                    running = True
            if running:
                return {"operation_id": operation_id, "status": "indeterminate", "reason": "dispatch_in_progress"}
        op["_owner_token"] = str(uuid.uuid4())
        claimed = conn.execute("""UPDATE feedback_ownership SET owner_token=?
            WHERE kind=? AND item_id=? AND active_operation=? AND owner_token=?""",
            (op["_owner_token"], op["kind"], op["item_id"], operation_id, step["owner_token"]))
        if claimed.rowcount != 1:
            return _stale(op)
        facts = step["facts"]
        if "expected" not in facts:
            return _append_step(conn, op, "conflict", {"reason": "not_dispatched", "dispatch_finished": True})
        _append_step(conn, op, "indeterminate", {**facts, "reason": "reconciling", "dispatch_finished": True})
    try:
        after = _current(read_current((op["kind"], op["item_id"])), "ok")
    except Exception:
        with _connection(db_path, write=True) as conn:
            return _append_step(conn, op, "indeterminate", {**facts, "reason": "read_unavailable", "dispatch_finished": True})
    with _connection(db_path, write=True) as conn:
        if not _owns(conn, op):
            return _stale(op)
        if facts["field"] == "engagement_count":
            return _append_step(conn, op, "indeterminate", {**facts, "reason": "counter_confirmation_ambiguous", "dispatch_finished": True})
        if after[facts["field"]] == facts["expected"][facts["field"]]:
            return _finish_feedback(conn, op, facts, after, provenance="unknown", kinds=kinds)
        return _append_step(conn, op, "indeterminate", {**facts, "reason": "manual_resolution_required", "dispatch_finished": True})

"""Explicit prospective capture from local staged facts and the shared ledger reader.

Staged catalog/features are current records, not historical snapshots. A collector
must supply complete records and source revisions; this module never imports the
database-backed application core or discovers a live database automatically.
"""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import types


def initial_facts(current, at):
    from experimental.evaluation.evaluate import digest, item_key, require, validate_initial_watch
    require(current and current.get("complete") is True, "complete staged catalog/features required")
    require(current.get("source_revisions"), "current source revision markers required")
    catalog, features = copy.deepcopy(current["catalog"]), copy.deepcopy(current["features"])
    events, states = [], {}
    for state in current.get("state", []):
        key = item_key(state)
        require(key not in states, "duplicate initial state")
        require(set(state["value"]) <= {"rating", "engagement_count", "play_count", "watch"},
                "cumulative watches require the explicit initial watch shape, not historical deltas")
        if "watch" in state["value"]:
            validate_initial_watch(state["value"]["watch"], at)
        id_ = "capture:" + digest([at, key])
        event = {"event_id": id_, "client_event_id": id_, "kind": key[0], "id": key[1],
                 "type": "initial_state", "session_id": None, "occurred_at": None,
                 "known_at": at, "value": {name: copy.deepcopy(value) for name, value in state["value"].items()
                     if name not in ("engagement_count", "play_count") or value is not None}}
        states[key] = event
        events.append(event)
    for row in catalog:
        row.update(known_at=at, available_at=at, removed_at=None)
    for row in features:
        row.update(known_at=at, effective_at=at, evidence_ids=[])
        if row["watched_tag_seconds"]:
            state = states.get(item_key(row))
            require(state is not None and "watch" in state["value"], "watched tags require same-item initial watch")
            row["evidence_ids"] = [state["event_id"]]
    return catalog, features, events


def current_snapshot(current, plan, *, source_root, code_revision, evaluation_at=None):
    """Freeze real current facts and designed probes, without creating sessions."""
    from experimental.evaluation import sweep
    from experimental.evaluation.evaluate import PURE_MODULES, RANKING_MODULE, SECTIONS, digest, load_pure_modules, require, timestamp, validate
    at = current["captured_at"]
    evaluated = evaluation_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    require(timestamp(at) < timestamp(evaluated), "evaluation must follow actual capture")
    root = Path(source_root).resolve(strict=True)
    modules = {name: path for name, path in PURE_MODULES.items() if (root / path).is_file()}
    require(RANKING_MODULE in modules, "explicit source root has no production ranking module")
    files = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in modules.values()}
    catalog, features, events = initial_facts(current, at)
    m = {name: copy.deepcopy(plan[name]) for name in ("seed", "policy", "ranker_config", "ranker_config_complete")}
    m["policy"].update(history="captured_initial_state", popularity="captured_play_counts")
    m.update(schema_version=1, mode="current_state", history_complete=False, freeze_at=at, cutoff=evaluated,
             training_end=None, development_end=None, sessions=[], code_revision=code_revision,
             diagnostic_scope="planned_current_state_no_temporal_labels",
             feature_revision=digest(features), event_revision=digest(events),
             source_revisions=copy.deepcopy(current["source_revisions"]),
             capture_status={"initial_state": "known_at_capture_not_historical_deltas", "future_outcomes": "unmeasured"},
             rankers={name: {"variant": name, "revision": code_revision, "modules": modules, "files": files}
                      for name in ("audit_old", "current", "admission_only")})
    m["source_revisions"]["staged_current_facts"] = digest(current)
    m["capture_metadata"] = copy.deepcopy(current.get("capture_metadata", {}))
    m["diagnostic_kinds"] = plan.get("diagnostic_kinds", ["image", "video"] if m["ranker_config"].get("include_images", False) else ["video"])
    require(isinstance(m["diagnostic_kinds"], list) and m["diagnostic_kinds"]
            and m["diagnostic_kinds"] == sorted(set(m["diagnostic_kinds"]))
            and set(m["diagnostic_kinds"]) <= {"video", "image"}, "invalid declared diagnostic output kinds")
    loaded = load_pure_modules(m["rankers"]["current"], source_root=root)
    require(m["ranker_config_complete"] is True
            and set(loaded[RANKING_MODULE].REQUIRED_CONFIG) <= set(m["ranker_config"]),
            "current diagnostics require the complete resolved native ranker configuration")
    base = {"id": "planned-base", "session_id": None, "split": "development", "cutoff": evaluated,
            "outcome_until": None, "history_kinds": ["image", "video"],
            "kinds": list(m["diagnostic_kinds"]),
            "exclude": [], "intent": {"tag_ids": [], "seed": None}, "stratum": "planned", "judgment": False}
    snapshot = {"manifest": m, "catalog": catalog, "features": features, "events": events, "contexts": [base]}
    probes, support = sweep.contexts(snapshot, split="development")
    unique, selected = set(), []
    for probe in probes:
        identity = digest({k: v for k, v in probe.items() if k not in ("id", "split", "stratum")})
        if identity in unique:
            continue
        unique.add(identity)
        probe.update(split="development" if len(selected) % 2 == 0 else "heldout", judgment=True)
        selected.append(probe)
        if len(selected) == 24:
            break
    require(selected, "no supported current-state probes")
    snapshot["contexts"] = selected
    m["diagnostic_design"] = {"judgment_contexts": len(selected), "judgment_shortfall": 24-len(selected),
                              "popularity_status": support["popularity_status"],
                              "strata_support": support["support"], "missing_strata": support["missing_strata"]}
    m["hashes"] = {name: digest(snapshot[name]) for name in SECTIONS}
    validate(snapshot)
    return snapshot


def read_ledger(source_root, sha256, database, *, through):
    from experimental.evaluation.evaluate import canonical, pure_execution, require
    path = Path(source_root) / "src/feedloop/ledger.py"
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha256, "shared ledger-reader source hash mismatch")
    names = {"__future__", "contextlib", "hashlib", "hmac", "json", "math", "os", "pathlib", "sqlite3", "time", "typing", "urllib.parse", "uuid", "feedloop.taste"}
    tree = ast.parse(data)
    # Deferred mutation-route imports are not needed by read_evidence. They stay
    # unexecuted and the runtime guard still rejects importing them if called.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            targets = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            require(all(t in names for t in targets) and not getattr(node,"level",0), "unsupported event-reader dependency")
    modules = {}
    for name in sorted(names | {"urllib.parse", "feedloop"}):
        original = importlib.import_module(name)
        proxy = types.ModuleType(name)
        proxy.__dict__.update({k:v for k,v in vars(original).items() if k != "__path__"})
        modules[name] = proxy
    reader = types.ModuleType("_e1_shared_events")
    modules[reader.__name__] = reader
    with pure_execution(modules):
        exec(compile(data,str(path),"exec"),reader.__dict__)
        target = reader.__dict__.get("read_evidence")
        require(type(target) is types.FunctionType, "shared read_evidence callable unavailable")
    with pure_execution(modules,read_only_database=database):
        result = target(str(Path(database).resolve()),since_ts=0,through_ts=through)
        require(type(result) is dict and result.get("status") == "ok", "shared evidence source unavailable")
        encoded = canonical(result)
    return json.loads(encoded) | {"reader_source_sha256":sha256}


def export_snapshot(plan, current, *, ledger=None, captured_at=None, frozen=None):
    """No unknown timestamp is backdated: current facts become known at capture.

    captured_at is injectable for synthetic integration tests. The CLI supplies
    the actual UTC clock. A later export retains frozen facts and appends only
    confirmed ledger outcomes. Corrections fail closed until the event owner
    supplies a history-preserving correction projection.
    """
    from experimental.evaluation.evaluate import digest, item_key, require, timestamp, validate, verify_prospective, SECTIONS
    at = captured_at or datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    timestamp(at)
    snapshot = copy.deepcopy(frozen if frozen is not None else plan)
    m = snapshot["manifest"]
    if frozen is None:
        catalog, features, events = initial_facts(current, at)
        m.update(mode="prospective",history_complete=False,freeze_at=at,cutoff=at)
        m["source_revisions"] = copy.deepcopy(current["source_revisions"])
        m["source_revisions"]["staged_current_facts"] = digest(current)
        snapshot.update(catalog=catalog, features=features, events=events)
    else:
        require(timestamp(at) >= timestamp(m["freeze_at"]), "capture precedes frozen plan")
        m["cutoff"] = at
    if ledger is not None:
        require(ledger.get("status") == "ok", "shared ledger read failed")
        reader_hash = ledger["reader_source_sha256"]
        if frozen is None:
            m["source_revisions"]["events_adapter"] = reader_hash
        else:
            require(m["source_revisions"].get("events_adapter") == reader_hash, "prospective event reader revision changed")
        m["capture_status"] = {k:copy.deepcopy(ledger.get(k)) for k in
            ("valid","validity_reasons","source_completeness","watch_capture_supported","promotion_enabled")}
        requests = {r["request_id"]:r for r in ledger["requests"]}
        rows = {r["event_id"]:r for r in ledger["events"]}
        claims = {r["outcome_id"]:r["viewed_id"] for r in ledger["attributions"]}
        existing = {e["event_id"] for e in snapshot["events"]}
        def iso(seconds):
            return datetime.fromtimestamp(seconds,timezone.utc).isoformat().replace("+00:00","Z")
        for row in sorted(rows.values(),key=lambda r:(r["occurred_at"],r["event_id"])):
            if row["event_id"] in existing or row["occurred_at"] < timestamp(m["freeze_at"]).timestamp():
                continue
            require(row.get("identity_valid") is True and row.get("canonical_session_id"), "event canonical identity unproven")
            p = row["payload"]
            event = {"event_id":row["event_id"],"client_event_id":digest([row["source"],row["source_event_id"]]),
                "kind":row["kind"],"id":row["item_id"],"session_id":row["canonical_session_id"],
                "occurred_at":iso(row["occurred_at"]),"known_at":iso(row["received_at"]),"source":row["source"]}
            if row["event_type"] in ("served","viewed"):
                req = requests[row["request_id"]]
                event.update(type="served" if row["event_type"] == "served" else "visible",value=1,
                    request_id=row["request_id"],surface=req["surface"],rank=p["source_rank"] if row["event_type"] == "served" else p["display_rank"],
                    ranker_revision=req["ranker_revision"],config_revision=req["config_hash"])
            else:
                require(not row.get("corrects_id"), "event correction projection unsupported; cannot rewrite frozen outcomes")
                require(p.get("provenance") == "confirmed_delta_v1", "unconfirmed outcome cannot become evaluation evidence")
                signal = p["signal"]
                require(signal in ("rating","engagement","watch"), "unsupported outcome signal")
                event.update(type={"rating":"rating","engagement":"engagement_delta","watch":"watch"}[signal],
                    value=p["rating_after"] if signal == "rating" else p["engagement_delta"] if signal == "engagement" else p["watched_s_delta"])
                if signal == "watch":
                    event.update(start_at=iso(p["started_at"]),end_at=event["occurred_at"])
                if row["event_id"] in claims:
                    event.update(exposure_id=claims[row["event_id"]],request_id=row["request_id"])
            snapshot["events"].append(event)
        m["event_revision"] = digest(ledger)
    m["hashes"] = {name:digest(snapshot[name]) for name in SECTIONS}
    validate(snapshot,planning=frozen is None)
    if frozen is not None:
        verify_prospective(snapshot,frozen)
    return snapshot

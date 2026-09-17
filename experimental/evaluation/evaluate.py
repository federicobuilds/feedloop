"""Frozen, local evaluation. CLI: export, validate, run, judge-export, judge-import.

The JSON bundle has manifest/catalog/features/events/contexts sections. Hashes
cover canonical, key-sorted JSON section bytes (UTF-8, no whitespace or NaN).
Export is explicitly invoked on local staged facts and an optional offline event
ledger. Missing production adapters are never replaced by synthetic scores.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import Counter
from contextlib import ExitStack, contextmanager
import copy
from datetime import datetime, timedelta
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import tracemalloc
import types
from unittest.mock import patch

import numpy as np

try:
    import resource
except ModuleNotFoundError:
    resource = None

from experimental.evaluation import metrics


ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ("catalog", "features", "events", "contexts")
VARIANTS = ("most_played", "random_tagmatched", "audit_old", "current", "admission_only")
KS = (10, 20, 50)
STAGES = ("profile", "candidates", "hydration", "scoring", "selection", "total")
SAFE_IMPORTS = {"__future__", "math", "collections", "typing", "dataclasses", "random", "numpy",
                "datetime", "time", "statistics", "itertools", "functools", "heapq", "threading"}
PURE_PACKAGE = "feedloop"
PURE_MODULES = {f"{PURE_PACKAGE}.{name}": f"src/feedloop/{name}.py"
                for name in ("ranking", "taste", "profiles")}
PURE_ALIASES = {name: f"src/feedloop/{name}.py" for name in ("taste", "profiles")}
RANKING_MODULE = PURE_PACKAGE + ".ranking"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def timestamp(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamps must be explicit UTC strings ending in Z")
    result = datetime.fromisoformat(value[:-1] + "+00:00")
    require(result.utcoffset() == timedelta(0), "timestamp must be UTC")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, minimum=0):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum


def item_key(row):
    require(row.get("kind") in ("video", "image") and type(row.get("id")) is int
            and row["id"] >= 0, "invalid (kind, integer id) key")
    return metrics.key(row)


def validate_initial_watch(watch, known_at):
    require(isinstance(watch, dict) and set(watch) == {"watched_s", "last_at", "visit_days"},
            "initial watch requires watched_s, last_at and visit_days")
    require(finite(watch["watched_s"]), "invalid initial watch seconds")
    last = timestamp(watch["last_at"])
    require(last <= timestamp(known_at), "initial watch last_at is after known_at")
    days = watch["visit_days"]
    require(isinstance(days, list) and all(type(day) is int for day in days)
            and days == sorted(set(days)) and all(day <= math.floor(last.timestamp() / 86400) for day in days)
            and (watch["watched_s"] == 0 or days), "invalid initial watch UTC visit_days")


def read_json(path):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, "duplicate JSON property: " + name)
            result[name] = value
        return result
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique,
                         parse_constant=lambda value: require(False, "nonfinite JSON: " + value))


def session_partition(session, manifest):
    start, end = timestamp(session["start_at"]), timestamp(session["end_at"])
    padded_end = end + timedelta(seconds=manifest["policy"]["purge_seconds"])
    boundaries = [timestamp(manifest[n]) for n in ("training_end", "development_end")]
    if any(start < boundary <= padded_end for boundary in boundaries):
        return "purged"
    return "train" if start < boundaries[0] else "development" if start < boundaries[1] else "heldout"


def validate(snapshot, *, planning=False):
    require(set(snapshot) == {"manifest", *SECTIONS}, "unexpected or missing snapshot section")
    m = snapshot["manifest"]
    require(m["schema_version"] == 1 and m["mode"] in ("historical", "prospective", "current_state"), "unsupported snapshot version/mode")
    diagnostic = m["mode"] == "current_state"
    require(type(m["history_complete"]) is bool, "history_complete must be explicit")
    require(m["history_complete"] or m["mode"] in ("prospective", "current_state"), "missing history requires prospective mode")
    cutoff = timestamp(m["cutoff"])
    if diagnostic:
        require(not planning and not m["history_complete"] and m["sessions"] == []
                and m["training_end"] is None and m["development_end"] is None
                and m.get("diagnostic_scope") == "planned_current_state_no_temporal_labels",
                "current diagnostics have no observed sessions or chronological splits")
        train_end = None
    else:
        train_end, dev_end = timestamp(m["training_end"]), timestamp(m["development_end"])
        require(train_end < dev_end and (planning or dev_end <= cutoff), "chronological split boundaries required")
    freeze = timestamp(m["freeze_at"]) if m["mode"] in ("prospective", "current_state") else None
    require(not planning or freeze == cutoff, "prospective planning cutoff must equal freeze_at")
    require((freeze < cutoff) if diagnostic else (freeze is None or freeze <= train_end),
            "prospective freeze must precede evaluation")
    require(type(m["seed"]) is int, "integer seed required")
    require(re.fullmatch(r"[0-9a-f]{40}", m["code_revision"]) is not None, "full code revision required")
    require(bool(m["feature_revision"]) and bool(m["event_revision"]), "feature/event revisions required")
    require(m["source_revisions"] and all(isinstance(v, str) and v for v in m["source_revisions"].values()), "source revisions required")
    expected = {name: digest(snapshot[name]) for name in SECTIONS}
    require(m["hashes"] == expected, "section hash mismatch; expected " + json.dumps(expected, sort_keys=True))
    policy = m["policy"]
    for field in ("cooldown_seconds", "purge_seconds", "attribution_window_seconds", "watch_positive_seconds", "repeat_delay_days"):
        require(finite(policy[field]), "invalid policy: " + field)
    require(policy["watch_positive_seconds"] > 0, "positive watch threshold required")
    require(policy["repeat_delay_days"] > 0, "repeat_delay_days must be positive")
    require(policy["purge_seconds"] >= policy["attribution_window_seconds"], "purge shorter than attribution window")
    require((policy["history"], policy["popularity"]) ==
            (("captured_initial_state", "captured_play_counts") if diagnostic else ("frozen_train", "play_events")),
            "unsupported history/popularity policy")
    require(policy["ks"] == list(KS), "required k values are 10,20,50")
    weights = policy["coverage_weights"]
    require(set(weights) == {"video", "image"} and all(finite(v) for v in weights.values())
            and abs(sum(weights.values()) - 1) < 1e-12, "invalid coverage weights")
    require(isinstance(m["ranker_config"], dict) and type(m["ranker_config_complete"]) is bool, "resolved ranker configuration required")
    require(set(m["rankers"]) == {"audit_old", "current", "admission_only"}, "explicit ranker availability required")
    for variant, pin in m["rankers"].items():
        if pin is None:
            continue
        require(pin["variant"] == variant and re.fullmatch(r"[0-9a-f]{40}", pin["revision"]), "invalid ranker revision/variant pin")
        validate_source_manifest(pin)

    sessions = {}
    for s in m["sessions"]:
        require(s["id"] not in sessions and s["id"], "duplicate/empty canonical session")
        require(timestamp(s["start_at"]) <= timestamp(s["end_at"])
                and (planning or timestamp(s["end_at"]) <= cutoff), "invalid session interval")
        require(s["split"] == session_partition(s, m), "session crosses split/purge boundary or has wrong split")
        sessions[s["id"]] = s

    catalog = {}
    for row in snapshot["catalog"]:
        require(set(row) == {"kind", "id", "available_at", "known_at", "removed_at", "eligible", "duration_s", "tag_ids",
                             "feature_presence", "duplicate_verified", "duplicate_group"}, "unexpected catalog fields; no cached ratings/counts allowed")
        key = item_key(row)
        require(key not in catalog, "duplicate catalog key")
        require(timestamp(row["available_at"]) <= timestamp(row["known_at"]) <= cutoff, "invalid catalog availability")
        if diagnostic:
            require(timestamp(row["known_at"]) == timestamp(row["available_at"]) == freeze and row["removed_at"] is None,
                    "current catalog facts must be first known at capture, not backdated")
        require(freeze is None or timestamp(row["known_at"]) <= freeze, "prospective catalog changed after freeze")
        require(row["removed_at"] is None or timestamp(row["available_at"]) <= timestamp(row["removed_at"]) <= cutoff, "invalid removal interval")
        require(type(row["eligible"]) is bool and finite(row["duration_s"]), "invalid eligibility/duration")
        require(type(row["duplicate_verified"]) is bool and (row["duplicate_group"] is None or
                (isinstance(row["duplicate_group"], str) and row["duplicate_group"] and row["duplicate_verified"])), "unverified duplicate group")
        require(isinstance(row["tag_ids"], list) and all(type(t) is int for t in row["tag_ids"]), "invalid catalog tags")
        require(isinstance(row["feature_presence"], list), "feature presence required")
        catalog[key] = row
    require(bool(catalog), "empty catalog is not an evaluation snapshot")

    feature_keys = set()
    feature_times = set()
    dimensions = {}
    event_ids = {e["event_id"] for e in snapshot["events"]}
    for f in snapshot["features"]:
        require(set(f) == {"kind", "id", "revision", "model_revision", "effective_at", "known_at", "complete", "tag_seconds",
                           "tag_categories", "watched_tag_seconds", "evidence_ids", "identity_ids", "vectors"}, "unexpected feature fields")
        key = item_key(f)
        identity = (key, f["revision"])
        require(key in catalog and identity not in feature_keys, "unknown item/duplicate feature revision")
        feature_keys.add(identity)
        require((key, timestamp(f["known_at"])) not in feature_times, "ambiguous simultaneous feature revisions")
        feature_times.add((key, timestamp(f["known_at"])))
        require(timestamp(f["effective_at"]) <= timestamp(f["known_at"]) <= cutoff, "invalid feature availability")
        if diagnostic:
            require(timestamp(f["known_at"]) == timestamp(f["effective_at"]) == freeze,
                    "current feature facts must be first known at capture, not backdated")
        require(freeze is None or timestamp(f["known_at"]) <= freeze, "prospective features changed after freeze")
        require(f["model_revision"] and f["revision"] and f["complete"] is True, "complete revisioned features required")
        require(all(finite(v) for v in f["tag_seconds"].values()), "invalid tag durations")
        require(set(f["evidence_ids"]) <= event_ids, "unknown feature evidence dependency")
        require(isinstance(f["watched_tag_seconds"], dict) and all(finite(v) for v in f["watched_tag_seconds"].values()), "invalid watched tag durations")
        require(not f["watched_tag_seconds"] or f["evidence_ids"], "watched features require evidence provenance")
        require(isinstance(f["identity_ids"], list), "identity links required, may be empty")
        for space, vector in f["vectors"].items():
            require(isinstance(vector, list) and vector and all(finite(v, -float("inf")) for v in vector)
                    and sum(v * v for v in vector) > 0, "invalid vector")
            require(space not in dimensions or dimensions[space] == len(vector), "mixed vector dimensions in one space")
            dimensions[space] = len(vector)

    events = {}
    client_ids = set()
    initial_keys = set()
    for e in snapshot["events"]:
        require(set(e) <= {"event_id", "client_event_id", "kind", "id", "session_id", "occurred_at", "known_at", "type", "value",
                           "start_at", "end_at", "request_id", "exposure_id", "surface", "rank", "ranker_revision", "config_revision",
                           "visit_id", "segment_id", "source", "rating_before"}, "unexpected evidence fields")
        require(e["event_id"] and e["event_id"] not in events and e["client_event_id"] and e["client_event_id"] not in client_ids, "duplicate event identity")
        require(item_key(e) in catalog, "unknown event item")
        known = timestamp(e["known_at"])
        require(known <= cutoff, "event recorded after snapshot cutoff")
        if e["type"] == "initial_state":
            require(freeze is not None and e["occurred_at"] is None and known == freeze and e["session_id"] is None,
                    "untimestamped state is prospective-only at freeze")
            require(isinstance(e["value"], dict) and set(e["value"]) <= {"rating", "engagement_count", "play_count", "watch"}, "unsupported initial state")
            require(item_key(e) not in initial_keys, "duplicate initial state for item")
            initial_keys.add(item_key(e))
            for name, value in e["value"].items():
                if name == "watch":
                    require(e["kind"] == "video", "initial watch is primary-kind only")
                    validate_initial_watch(value, e["known_at"])
                    continue
                require(value is None and name == "rating" or finite(value), "invalid initial state value")
                require(name != "rating" or value is None or value <= 100, "initial rating outside 0..100")
                require(name == "rating" or type(value) is int, "initial counters must be integers")
        else:
            require(not diagnostic, "current diagnostics accept initial state only, not temporal events")
            require(e["type"] in {"watch", "play", "image_view", "rating", "o_delta", "served", "visible"}, "unknown event type")
            at = timestamp(e["occurred_at"])
            require(at <= known, "event known before occurrence")
            s = sessions.get(e["session_id"])
            require(s is not None and timestamp(s["start_at"]) <= at <= timestamp(s["end_at"]), "event outside canonical session")
            require(e["value"] is None and e["type"] == "rating" or finite(e["value"], -float("inf")), "invalid event value")
            if e["type"] == "rating":
                require(e["value"] is None or 0 <= e["value"] <= 100, "rating outside 0..100")
            elif e["type"] == "o_delta":
                require(type(e["value"]) is int, "O change must be an integer delta")
            else:
                require(e["value"] >= 0, "negative behavioral delta")
            if e["type"] == "play":
                require(e["value"] == 1, "one deduplicated play event per play")
            if e["type"] == "watch":
                require(e["kind"] == "video" and timestamp(s["start_at"]) <= timestamp(e["start_at"]) <= at
                        and timestamp(e["end_at"]) == at, "watch interval must end at event occurrence")
                require(e["value"] <= (at - timestamp(e["start_at"])).total_seconds() + 1e-9, "watch delta exceeds its interval")
            if e["type"] in ("visible", "served"):
                require(e["request_id"] and e["surface"] and type(e["rank"]) is int and e["rank"] >= 0
                        and e["ranker_revision"] and e["config_revision"], "exposure/serve identity incomplete")
        events[e["event_id"]] = e
        client_ids.add(e["client_event_id"])
    for e in events.values():
        exposure = e.get("exposure_id")
        if exposure is not None:
            parent = events.get(exposure)
            require(parent is not None and parent["type"] == "visible" and item_key(parent) == item_key(e)
                    and parent["session_id"] == e["session_id"] and parent["request_id"] == e.get("request_id"), "invalid outcome attribution")
            delta = (timestamp(e["occurred_at"]) - timestamp(parent["occurred_at"])).total_seconds()
            require(0 < delta <= policy["attribution_window_seconds"], "outcome is not post-exposure within window")
            require(e["type"] != "watch" or timestamp(e["start_at"]) >= timestamp(parent["occurred_at"]), "pre-exposure watch cannot earn new reward")
    for f in snapshot["features"]:
        require(all(timestamp(events[e]["known_at"]) <= timestamp(f["known_at"]) for e in f["evidence_ids"]), "feature uses future evidence")
        initial = [events[e] for e in f["evidence_ids"] if events[e]["type"] == "initial_state"]
        if initial:
            require(f["watched_tag_seconds"] and all(item_key(e) == item_key(f) and "watch" in e["value"] for e in initial),
                    "watched features require same-item initial watch provenance")

    context_ids = set()
    judgments = Counter()
    for c in snapshot["contexts"]:
        require(set(c) == {"id", "session_id", "split", "cutoff", "outcome_until", "history_kinds", "kinds", "exclude", "intent", "stratum", "judgment"}, "unexpected context fields")
        require(c["id"] not in context_ids and c["id"], "duplicate context")
        context_ids.add(c["id"])
        if diagnostic:
            require(c["session_id"] is None and c["outcome_until"] is None
                    and c["split"] in ("development", "heldout") and timestamp(c["cutoff"]) == cutoff,
                    "diagnostic probes are planned, not observed session windows")
        else:
            s = sessions.get(c["session_id"])
            require(s is not None and c["split"] == s["split"] and c["split"] in ("development", "heldout", "purged"), "context split mismatch")
            require(timestamp(s["start_at"]) <= timestamp(c["cutoff"]) < timestamp(c["outcome_until"]) <= timestamp(s["end_at"]), "context window crosses session")
        require(c["history_kinds"] == sorted(set(c["history_kinds"])) and set(c["history_kinds"]) <= {"video", "image"}, "invalid history kinds")
        require(set(c["kinds"]) <= {"video", "image"} and c["kinds"], "invalid output kinds")
        require(all(set(i) == {"kind", "id"} and item_key(i) in catalog for i in c["exclude"]), "unknown excluded item")
        require(set(c["intent"]) == {"tag_ids", "seed"} and isinstance(c["intent"]["tag_ids"], list)
                and all(type(t) is int for t in c["intent"]["tag_ids"]), "invalid intent fields/tags")
        if c["intent"]["seed"] is not None:
            require(set(c["intent"]["seed"]) == {"kind", "id"} and item_key(c["intent"]["seed"]) in catalog, "unknown intent seed")
            seed = catalog[item_key(c["intent"]["seed"])]
            require(timestamp(seed["known_at"]) < timestamp(c["cutoff"]), "future intent seed")
        require(c["stratum"] and type(c["judgment"]) is bool, "context stratum/judgment flag required")
        if c["judgment"]:
            judgments[c["split"]] += 1
    if diagnostic:
        require(all(judgments[s] <= 12 for s in ("development", "heldout")), "at most 12 diagnostic judgments per split")
    else:
        require(judgments == {"development": 12, "heldout": 12}, "freeze exactly 12 development and 12 heldout judgment contexts")
    return {"valid": True, "phase": "current_state_diagnostic" if diagnostic else "prospective_plan" if planning else "observed_snapshot",
            "snapshot_hash": digest(snapshot), "section_hashes": expected,
            "purged_sessions": sorted(s["id"] for s in sessions.values() if s["split"] == "purged")}


def verify_prospective(snapshot, frozen):
    """Compare observed outcomes with an independently retained pre-outcome plan.

    The local collector owns making/retaining that plan at T0. Hashes establish
    immutable inputs, not proof of when a person actually created the file.
    """
    validate(frozen, planning=True)
    validate(snapshot)
    require(snapshot["manifest"]["mode"] == "prospective", "prospective anchor requires prospective mode")
    mutable = {"cutoff", "hashes", "event_revision", "capture_status"}
    require({k:v for k,v in frozen["manifest"].items() if k not in mutable} ==
            {k:v for k,v in snapshot["manifest"].items() if k not in mutable}, "prospective plan/config/source revisions changed")
    for section in ("catalog", "features", "contexts"):
        require(digest(snapshot[section]) == digest(frozen[section]), "prospective frozen section changed: " + section)
    observed = {e["event_id"]: e for e in snapshot["events"]}
    original = {e["event_id"]: e for e in frozen["events"]}
    require(all(observed.get(id_) == event for id_,event in original.items()), "prospective pre-freeze evidence changed")
    freeze = timestamp(frozen["manifest"]["freeze_at"])
    for id_,event in observed.items():
        if id_ not in original:
            require(event["type"] != "initial_state" and timestamp(event["occurred_at"]) >= freeze
                    and timestamp(event["known_at"]) >= freeze, "prospective addition is backdated")
    return digest(frozen)


def inputs_at(snapshot, context):
    """Never hand labels, manifest, future revisions or heldout events to rank()."""
    m = snapshot["manifest"]
    cutoff = timestamp(context["cutoff"])
    training_end = timestamp(m["freeze_at"] if m["mode"] == "current_state" else m["training_end"])
    sessions = {s["id"]: s for s in m["sessions"]}
    evidence = []
    session_visible = set()
    for e in snapshot["events"]:
        if timestamp(e["known_at"]) >= cutoff:
            continue
        if e["type"] == "initial_state":
            usable = timestamp(e["known_at"]) <= training_end
        else:
            s = sessions[e["session_id"]]
            usable = (s["split"] == "train" and timestamp(e["occurred_at"]) < training_end
                      and timestamp(e["known_at"]) < training_end)
            if e["session_id"] == context["session_id"] and e["type"] == "visible" and timestamp(e["occurred_at"]) < cutoff:
                session_visible.add(item_key(e))
        if usable and e["kind"] in context["history_kinds"]:
            evidence.append(copy.deepcopy(e))
    evidence.sort(key=lambda e: (timestamp(e["known_at"]), e["event_id"]))
    evidence_ids = {e["event_id"] for e in evidence}
    catalog = [copy.deepcopy(r) for r in snapshot["catalog"]
               if timestamp(r["known_at"]) < cutoff and timestamp(r["available_at"]) < cutoff
               and (r["removed_at"] is None or timestamp(r["removed_at"]) > cutoff)]
    catalog_keys = {item_key(r) for r in catalog}
    # A later removal date is future metadata, even when it does not filter.
    for r in catalog:
        r["removed_at"] = None
    features = {}
    initial_ids = {e["event_id"] for e in snapshot["events"] if e["type"] == "initial_state"}
    for f in sorted(snapshot["features"], key=lambda f: (timestamp(f["known_at"]), f["revision"])):
        key = item_key(f)
        if key in catalog_keys and timestamp(f["known_at"]) < cutoff and timestamp(f["effective_at"]) < cutoff:
            if not set(f["evidence_ids"]) <= evidence_ids:
                if not set(f["evidence_ids"]) <= initial_ids:
                    continue
                # Mask taste provenance, not the item's independent content features.
                features[key] = dict(copy.deepcopy(f), watched_tag_seconds={}, evidence_ids=[])
            else:
                features[key] = copy.deepcopy(f)
    for r in catalog:
        r["feature_presence"] = sorted(features.get(item_key(r), {}).get("vectors", {}))
    exclusions = {item_key(i): "context_exclusion" for i in context["exclude"]}
    exclusions.update({i: "visible_in_session" for i in session_visible})
    seed = context["intent"]["seed"]
    if seed is not None:
        seed_key = item_key(seed)
        exclusions[seed_key] = "intent_seed"
        seed_row = next((r for r in catalog if item_key(r) == seed_key), None)
        if seed_row and seed_row["duplicate_verified"] and seed_row["duplicate_group"] is not None:
            for row in catalog:
                if (row["kind"] == seed["kind"] and row["duplicate_verified"]
                        and row["duplicate_group"] == seed_row["duplicate_group"] and item_key(row) != seed_key):
                    exclusions[item_key(row)] = "seed_duplicate"
    cooldown = m["policy"]["cooldown_seconds"]
    for e in evidence:
        if e["type"] in ("watch", "play") and (cutoff - timestamp(e["occurred_at"])).total_seconds() < cooldown:
            exclusions[item_key(e)] = "shared_cooldown"
    for r in catalog:
        if not r["eligible"] or r["kind"] not in context["kinds"]:
            exclusions[item_key(r)] = "catalog_or_kind_policy"
    eligible = sorted(catalog_keys - exclusions.keys())
    counts = Counter()
    for e in evidence:
        if e["type"] == "initial_state":
            if "play_count" in e["value"]:
                counts[item_key(e)] = e["value"]["play_count"]
        elif e["type"] == "play":
            counts[item_key(e)] += 1
    safe_context = {name: copy.deepcopy(context[name]) for name in ("id", "session_id", "intent", "kinds", "cutoff")}
    safe_context.update(now=context["cutoff"], limit=max(KS), offset=0,
                        eligible_ids=[{"kind": kind, "id": id_} for kind, id_ in eligible])
    inputs = {"catalog": sorted(catalog, key=item_key), "features": [features[i] for i in sorted(features)], "evidence": evidence}
    return inputs, safe_context, counts, exclusions


def labels_at(snapshot, context, eligible):
    labels = {"implicit": {}, "explicit": {}}
    watches = Counter()
    o_changes = Counter()
    outside = set()
    events = (e for e in snapshot["events"] if e["type"] != "initial_state")
    for e in sorted(events, key=lambda e: (timestamp(e["occurred_at"]), e["event_id"])):
        if e["session_id"] != context["session_id"]:
            continue
        if not timestamp(context["cutoff"]) <= timestamp(e["occurred_at"]) <= timestamp(context["outcome_until"]):
            continue
        if e["type"] == "watch" and timestamp(e["start_at"]) < timestamp(context["cutoff"]):
            continue
        key = item_key(e)
        if key not in eligible:
            if e["type"] in ("watch", "rating", "o_delta"):
                outside.add(key)
            continue
        if e["type"] == "watch":
            watches[key] += e["value"]
        elif e["type"] == "rating":
            if e["value"] is None:
                labels["explicit"].pop(key, None)
            else:
                labels["explicit"][key] = e["value"] / 20 if e["value"] >= 80 else 0
        elif e["type"] == "o_delta":
            o_changes[key] += e["value"]
    for key, delta in o_changes.items():
        if delta > 0:
            labels["explicit"][key] = 5
    labels["implicit"] = {key: 1 for key, seconds in watches.items()
                          if seconds >= snapshot["manifest"]["policy"]["watch_positive_seconds"]}
    return labels, len(outside)


def seeded(seed, context_id, purpose):
    return int(digest([seed, context_id, purpose])[:16], 16)


def baseline(name, inputs, context, counts, seed):
    started = time.perf_counter()
    eligible = [item_key(i) for i in context["eligible_ids"]]
    catalog = {item_key(i): i for i in inputs["catalog"]}
    if name == "most_played":
        order = sorted(eligible, key=lambda i: (-counts.get(i, 0), i))
        degenerate = not any(counts.get(i, 0) for i in eligible)
    elif name == "random_tagmatched":
        tags = set(context["intent"]["tag_ids"])
        order = sorted(i for i in eligible if tags.intersection(catalog[i]["tag_ids"]))
        random.Random(seed).shuffle(order)
        degenerate = not order
    else:
        raise ValueError("unknown baseline")
    return {"items": [{"kind": k, "id": i, "score": counts.get((k, i), 0) if name == "most_played" else 0,
                       "explanation": {"baseline": name}} for k, i in order[:max(KS)]],
            "source_counts": {"eligible": len(eligible), "retrieved": len(order)},
            "route_inventory": {"scope":"frozen_context","complete":True,"eligible_ids":context["eligible_ids"],
                "supported_routes":[name],"routes":{name:{"enabled":True,"complete":True,"stage":"pre_budget",
                "eligible_ids":[{"kind":k,"id":i} for k,i in order]}}},
            "exclusions": [], "timings": {"total": time.perf_counter() - started}, "degenerate": degenerate,
            "degenerate_by_kind": {kind: not any(counts.get(i, 0) for i in eligible if i[0] == kind)
                                   if name == "most_played" else not any(i[0] == kind for i in order)
                                   for kind in context["kinds"]}}


def validate_source_manifest(pin):
    """Keep files[path]=sha256; modules adds explicit import-name ownership.

    Every source needs its canonical qualified name. Optional bare aliases must
    also be declared and resolve to the same module object, not a second exec.
    No namespace initializer, arbitrary path, or wildcard is accepted.
    """
    modules, files = pin.get("modules"), pin.get("files")
    require(isinstance(modules, dict) and isinstance(files, dict), "explicit modules and files source manifest required")
    require(modules.get(RANKING_MODULE) == PURE_MODULES[RANKING_MODULE], "production entry point must be source pinned")
    allowed = PURE_MODULES | PURE_ALIASES
    for name, path in modules.items():
        require(name in allowed and path == allowed[name], "unknown pure module/path: " + str(name))
    require(set(files) == set(modules.values()), "unmapped or unpinned source dependency")
    for path, sha in files.items():
        require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), "invalid source file hash")
        canonical_name = next(name for name, owned_path in PURE_MODULES.items() if owned_path == path)
        require(modules.get(canonical_name) == path, "dependency requires its canonical qualified module name")


@contextmanager
def pure_execution(modules=None, *, read_only_database=None):
    """Tripwires for accidental application imports/I/O, not a security sandbox."""
    modules = modules or {}
    attempted = []
    original_import = builtins.__import__
    readonly_connect = None
    if read_only_database is not None:
        import sqlite3
        connect = sqlite3.connect
        path = Path(read_only_database).resolve()
        expected_uri = path.as_uri() + "?mode=ro"
        def readonly_connect(connection_uri, **kwargs):
            require(connection_uri == expected_uri and kwargs.get("uri") is True, "capture reader attempted another database")
            require(not Path(str(path) + "-wal").exists(), "capture requires a consistent offline database without WAL sidecar")
            conn = connect(connection_uri + "&immutable=1", uri=True, timeout=5, isolation_level=None)
            conn.execute("PRAGMA query_only=ON")
            def authorize(action, first, second, database, trigger):
                if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_TRANSACTION):
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_PRAGMA and first in ("query_only", "foreign_keys") and second in (None, "ON"):
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            conn.set_authorizer(authorize)
            return conn
    def forbidden(*args, **kwargs):
        attempted.append(True)
        raise ValueError("production ranking attempted external I/O or thread creation")
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = importlib.util.resolve_name("." * level + name, (globals or {}).get("__package__", "")) if level else name
        if absolute in modules:
            module = modules[absolute]
            if hasattr(module, "__path__"):
                # A package shell grants access to declared children only.
                if not fromlist or any(absolute + "." + child not in modules or
                                       hasattr(modules[absolute + "." + child], "__path__") for child in fromlist):
                    forbidden()
            if fromlist:
                return module
            return modules[absolute.split(".")[0]]
        if level or absolute.split(".")[0] not in SAFE_IMPORTS:
            forbidden()
        return original_import(name, globals, locals, fromlist, level)
    previous = {name: sys.modules.get(name) for name in modules}
    try:
        sys.modules.update(modules)
        with ExitStack() as stack:
            for target in ("builtins.open", "io.open", "os.open", "os.system", "threading.Thread.start",
                           "socket.socket", "sqlite3.connect", "subprocess.Popen"):
                stack.enter_context(patch(target, side_effect=readonly_connect if target == "sqlite3.connect" and readonly_connect else forbidden))
            stack.enter_context(patch("builtins.__import__", side_effect=guarded_import))
            yield
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    require(not attempted, "production ranking swallowed an I/O or thread violation")


def load_pure_modules(pin, *, source_root=None):
    """Verify the complete source closure before executing any source bytes.

    All package parents are empty in-memory modules with empty search paths.
    Only buffered, hash-verified source runs, including dependencies; Python's
    filesystem importer never sees an application package or __init__.py.
    """
    validate_source_manifest(pin)
    source_root = ROOT if source_root is None else Path(source_root).resolve()
    sources = {}
    for filename, sha in sorted(pin["files"].items()):
        path = source_root / filename
        require(path.resolve().is_relative_to(source_root.resolve()), "pure source path escapes worktree")
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == sha, "production source hash mismatch: " + filename)
        sources[filename] = data.decode("utf-8-sig")

    modules = {}
    for name, filename in PURE_MODULES.items():
        if name not in pin["modules"]:
            continue
        parts = name.split(".")
        for length in range(1, len(parts)):
            parent = ".".join(parts[:length])
            if parent not in modules:
                package = types.ModuleType(parent)
                package.__package__ = parent
                package.__path__ = []
                package.__spec__ = importlib.machinery.ModuleSpec(parent, loader=None, is_package=True)
                modules[parent] = package
        spec = importlib.util.spec_from_file_location(name, source_root / filename)
        modules[name] = importlib.util.module_from_spec(spec)
    for name, module in list(modules.items()):
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    for alias, filename in pin["modules"].items():
        if alias in PURE_ALIASES:
            canonical_name = next(name for name, path in PURE_MODULES.items() if path == filename)
            modules[alias] = modules[canonical_name]

    dependencies = {}
    for name, filename in pin["modules"].items():
        if name in PURE_ALIASES:
            continue
        dependencies[name] = set()
        tree = ast.parse(sources[filename], filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), modules[name].__package__) if node.level else node.module
                if base in modules and hasattr(modules[base], "__path__"):
                    imports = [base + "." + alias.name for alias in node.names]
                else:
                    imports = [base]
            else:
                continue
            for target in imports:
                if target in pin["modules"]:
                    dependencies[name].add(modules[target].__name__)
                else:
                    require(target and target.split(".")[0] in SAFE_IMPORTS,
                            "unknown or unpinned pure dependency: " + str(target))

    order, visiting = [], set()
    def visit(name):
        require(name not in visiting, "cyclic pure source dependency")
        if name in order:
            return
        visiting.add(name)
        for dependency in sorted(dependencies[name]):
            visit(dependency)
        visiting.remove(name)
        order.append(name)
    for name in sorted(dependencies):
        visit(name)

    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        for name in sorted(SAFE_IMPORTS):
            importlib.import_module(name)
        with pure_execution(modules):
            for name in order:
                filename = pin["modules"][name]
                exec(compile(sources[filename], filename, "exec"), modules[name].__dict__)
            boundaries = {}
            filename = pin["modules"][RANKING_MODULE]
            for function in ast.parse(sources[filename]).body:
                if not isinstance(function,ast.FunctionDef):
                    continue
                for node in ast.walk(function):
                    for target in node.targets if isinstance(node,ast.Assign) else ():
                        if (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
                                and target.value.id == "timings" and isinstance(target.slice,ast.Constant)
                                and target.slice.value in STAGES):
                            boundaries[(filename,function.name,node.lineno)] = target.slice.value
            modules[RANKING_MODULE].__dict__["_e1_stage_boundaries"] = boundaries
    finally:
        sys.dont_write_bytecode = old_bytecode
    return modules


def load_ranker(manifest, variant, *, source_root=None):
    """Only the owner's real callable produces production ranking results."""
    pin = manifest["rankers"][variant]
    if pin is None:
        return None, "no source-pinned production variant supplied"
    if not manifest["ranker_config_complete"]:
        return None, "fully resolved production configuration not supplied"
    try:
        modules = load_pure_modules(pin) if source_root is None else load_pure_modules(pin, source_root=source_root)
    except FileNotFoundError as exc:
        return None, "production dependency unavailable: " + str(exc.filename)
    module = modules[RANKING_MODULE]
    with pure_execution(modules):
        # Read inert storage, not source-defined descriptors or __getattr__.
        require(type(module) is types.ModuleType, "production metadata requires a plain module")
        namespace = types.ModuleType.__getattribute__(module, "__dict__")
        supported = namespace.get("SUPPORTED_VARIANTS", ())
        require(type(supported) in (tuple, list), "SUPPORTED_VARIANTS must be a built-in tuple/list")
        require(all(type(name) is str and name in VARIANTS[2:] for name in supported),
                "SUPPORTED_VARIANTS contains an invalid name")
        require(len(supported) == len(set(supported)), "SUPPORTED_VARIANTS contains duplicate names")
        if variant not in supported:
            return None, "production callable does not declare " + variant + "; supported=" + repr(tuple(supported))
        target = namespace.get("rank")
        require(type(target) is types.FunctionType, "production rank callable must be a plain function")
    def rank(*args, **kwargs):
        with pure_execution(modules):
            return target(*args, **kwargs)
    rank.stage_boundaries = namespace.get("_e1_stage_boundaries",{})
    eligibility = namespace.get("shared_hard_eligibility")
    if type(eligibility) is types.FunctionType:
        def shared_hard_eligibility(*args, **kwargs):
            with pure_execution(modules):
                return eligibility(*args, **kwargs)
        rank.shared_hard_eligibility = shared_hard_eligibility
    return rank, None


def measure_call(call, boundaries):
    """Observe actual source timing boundaries, without changing its algorithm.

    Traced allocation peaks are Python/NumPy-tracked bytes, not total native RSS.
    Profiling overhead is included. PC cold/warm serving benchmarks remain a
    separate run on the actual machine and handler.
    """
    old_trace = sys.gettrace()
    starts, stages = {}, {}
    greatest = 0
    functions = {(file,name) for file,name,_ in boundaries}
    def observe(frame,event,arg):
        nonlocal greatest
        code = frame.f_code
        if (code.co_filename,code.co_name) not in functions:
            return None
        if event == "call":
            current,_ = tracemalloc.get_traced_memory()
            starts[id(frame)] = [time.perf_counter(),current,time.perf_counter(),current]
            tracemalloc.reset_peak()
        stage = boundaries.get((code.co_filename,code.co_name,frame.f_lineno)) if event == "line" else None
        if stage is not None and id(frame) in starts:
            tick,base,total_tick,total_base = starts[id(frame)]
            current,peak = tracemalloc.get_traced_memory()
            greatest = max(greatest,peak)
            start_time,start_bytes = (total_tick,total_base) if stage == "total" else (tick,base)
            stages[stage] = {"observed_seconds":time.perf_counter()-start_time,
                "traced_net_bytes":current-start_bytes,
                "traced_peak_above_entry_bytes":max(0,(greatest if stage == "total" else peak)-start_bytes)}
            starts[id(frame)][:2] = [time.perf_counter(),current]
            tracemalloc.reset_peak()
        if event == "return":
            starts.pop(id(frame),None)
        return observe
    tracemalloc.start()
    started = time.perf_counter()
    try:
        if old_trace is None and boundaries:
            sys.settrace(observe)
        result = call()
        elapsed = time.perf_counter()-started
        _,peak = tracemalloc.get_traced_memory()
    finally:
        sys.settrace(old_trace)
        tracemalloc.stop()
    return result,elapsed,max(peak,greatest),stages


def check_result(result, eligible):
    require(isinstance(result, dict) and all(k in result for k in ("items", "source_counts", "exclusions", "timings")), "rank result contract incomplete")
    order = [item_key(i) for i in result["items"]]
    require(len(order) == len(set(order)) and set(order) <= set(eligible), "ranker/baseline returned duplicate or ineligible items")
    require(all(finite(i["score"], -float("inf")) and isinstance(i["explanation"], dict) for i in result["items"]), "invalid score/explanation")
    require(isinstance(result["source_counts"], dict) and all(type(v) is int and v >= 0 for v in result["source_counts"].values()), "invalid source counts")
    require(isinstance(result["exclusions"], list) and isinstance(result["timings"], dict)
            and set(result["timings"]) <= set(STAGES) and all(finite(v) for v in result["timings"].values()), "invalid exclusions/timings")
    return order


def source_survival(result, k):
    inventory = result.get("route_inventory") or {}
    routes = inventory.get("routes", {})
    admitted = {item_key(i) for i in (result.get("admission_trace") or {}).get("admitted_ids", [])}
    output = result["items"][:k]
    out = {}
    for name, route in routes.items():
        raw = {item_key(i) for i in route["eligible_ids"]}
        out[name] = {"pre_budget": len(raw), "admitted": result["source_counts"].get(name),
                     "returned": sum(name in i["explanation"].get("sources", ())
                                     or i["explanation"].get("baseline") == name for i in output)}
        if name in ("most_played", "random_tagmatched"):
            out[name]["admitted"] = len(raw)
    if result.get("admission_trace") is not None:
        tags = {item_key(i) for i in routes.get("tags", {}).get("eligible_ids", [])}
        embeddings = {item_key(i) for name in ("visual", "voice", "sound")
                      for i in routes.get(name, {}).get("eligible_ids", [])}
        only = embeddings - tags
        admitted_only = only & admitted
        out["embedding_only"] = {"pre_budget": len(only), "admitted": len(admitted_only),
                                 "returned": sum(item_key(i) in admitted_only for i in output)}
    return out


def run(snapshot, *, split="development", variants=VARIANTS, source_root=None,
        frozen_from=None, admission_comparison=False, sweep_policy=None):
    from experimental.evaluation import sweep
    validation = validate(snapshot)
    require(split in ("development", "heldout", "all"), "invalid split")
    require(variants and len(set(variants)) == len(variants) and set(variants) <= set(VARIANTS), "invalid comparison variants")
    m = snapshot["manifest"]
    diagnostic = m["mode"] == "current_state"
    eligibility_ranker = None
    if diagnostic:
        eligibility_ranker, reason = load_ranker(m, "current", source_root=source_root)
        require(eligibility_ranker is not None and hasattr(eligibility_ranker, "shared_hard_eligibility"),
                "current diagnostics require source-pinned shared_hard_eligibility; integrate the ranking owner adapter")
    anchor = None
    if m["mode"] == "prospective":
        require(frozen_from is not None, "prospective evaluation requires independently retained --frozen-from plan")
        anchor = verify_prospective(snapshot, frozen_from)
    admission_old = "audit_old" if "audit_old" in variants else "current"
    if admission_comparison:
        require({admission_old, "admission_only"} <= set(variants), "admission contract mismatch: request old and admission_only")
        for name in (admission_old, "admission_only"):
            _, reason = load_ranker(m, name, source_root=source_root)
            require(reason is None, "admission contract mismatch: " + name + ": " + str(reason))
        first,second = (m["rankers"][name] for name in (admission_old, "admission_only"))
        require(first["files"] == second["files"] and first["modules"] == second["modules"],
                "admission contract mismatch: variants must share pinned scorer/profile/selector sources")
    contexts = sorted((c for c in snapshot["contexts"] if c["split"] != "purged"
                       and (split == "all" or c["split"] == split)), key=lambda c: (timestamp(c["cutoff"]), c["id"]))
    sweep_plan = None
    if sweep_policy is not None:
        contexts,sweep_plan = sweep.contexts(snapshot,split=split,policy=sweep_policy)
    report = {"schema_version": 1, "snapshot_hash": validation["snapshot_hash"], "split": split,
              "code_revision": m["code_revision"], "ranker_pins": m["rankers"], "seed": m["seed"],
              "ranker_config_hash": digest(m["ranker_config"]), "prospective_anchor_hash": anchor,
              "capture_status":m.get("capture_status"),
              "admission_comparison": {"requested": admission_comparison, "isolation_verified": False,
                                       "status": "pending_production_variant_and_parity_evidence"},
              "policy": m["policy"], "purged_sessions": validation["purged_sessions"], "variants": {},
              "quality_lift": None, "judgments": "unmeasured", "accuracy_scope": "unmeasured_current_state_no_temporal_labels" if diagnostic else "observed_labels_not_causal_lift",
              "sweep": {"contexts": len(contexts), "target_contexts": 128,
                        "shortfall": max(0, 128 - len(contexts)), "strata": dict(sorted(Counter(c["stratum"] for c in contexts).items()))}}
    report["evaluator_source_hashes"] = {name: hashlib.sha256((ROOT / "experimental" / "evaluation" / name).read_bytes()).hexdigest()
                                         for name in ("evaluate.py", "metrics.py", "capture.py", "sweep.py")}
    if sweep_plan is not None:
        report["sweep"] = sweep_plan
    measurements = {"schema_version": 1, "snapshot_hash": validation["snapshot_hash"],
                    "scope": "offline_rank_call_not_serving_or_database_latency",
                    "wall_includes": "allocation tracing and production I/O tripwire overhead",
                    "python_version": sys.version, "numpy_version": np.__version__, "platform": sys.platform,
                     "process_peak_rss_status": "measured" if resource is not None else "unavailable_on_platform", "samples": []}
    hard_contexts = {}
    for variant in variants:
        if diagnostic and variant == "most_played":
            known_counts = {item_key(e) for e in snapshot["events"] if "play_count" in e["value"]}
            output_kinds = {kind for c in contexts for kind in c["kinds"]}
            missing = {r["kind"] for r in snapshot["catalog"] if r["eligible"] and r["kind"] in output_kinds and item_key(r) not in known_counts}
            if missing:
                report["variants"][variant] = {"status": "unavailable", "reason": "authoritative_play_counts_missing",
                                                "missing_kinds": sorted(missing)}
                continue
        ranker, reason = (None, None) if variant in VARIANTS[:2] else load_ranker(m, variant, source_root=source_root)
        if reason:
            if source_root is not None:
                raise ValueError("production contract mismatch: " + variant + ": " + reason)
            report["variants"][variant] = {"status": "unavailable", "reason": reason}
            continue
        rows, aggregate = [], {str(k): [] for k in KS}
        stopped = None
        for c in contexts:
            prepare_started = time.perf_counter()
            inputs, context, counts, exclusions = inputs_at(snapshot, c)
            if diagnostic:
                if c["id"] not in hard_contexts:
                    full, full_context, full_counts, _ = inputs_at(snapshot, dict(c, history_kinds=["image", "video"]))
                    decision = eligibility_ranker.shared_hard_eligibility(full, context=full_context, config=m["ranker_config"])
                    require(isinstance(decision, list) and all(type(key) is str and re.fullmatch(r"(?:video|image):[1-9][0-9]*", key) for key in decision),
                            "shared hard eligibility contract mismatch")
                    allowed = [(key.split(":")[0], int(key.split(":")[1])) for key in decision]
                    allowed_set = set(allowed)
                    require(len(allowed) == len(allowed_set) and allowed_set <= {item_key(i) for i in full_context["eligible_ids"]},
                            "shared eligibility added unknown or excluded items")
                    decision = {"eligible_ids": [{"kind": kind, "id": id_} for kind, id_ in allowed],
                                "exclusions": [{"kind": kind, "id": id_, "reason": "shared_hard_policy"}
                                    for kind, id_ in sorted({item_key(i) for i in full_context["eligible_ids"]} - allowed_set)]}
                    hard_contexts[c["id"]] = (copy.deepcopy(decision), full_counts)
                decision, counts = hard_contexts[c["id"]]
                context["eligible_ids"] = copy.deepcopy(decision["eligible_ids"])
                for excluded in decision["exclusions"]:
                    exclusions[item_key(excluded)] = excluded["reason"]
            prepare_seconds = time.perf_counter()-prepare_started
            eligible = [item_key(i) for i in context["eligible_ids"]]
            eligible_set = set(eligible)
            seed = seeded(m["seed"], c["id"], variant if variant in VARIANTS[:2] else "ranking")
            before = digest([inputs, context, m["ranker_config"]])
            def call():
                if ranker is None:
                    return baseline(variant, inputs, context, counts, seed)
                return ranker(inputs, context=context, config=m["ranker_config"], seed=seed, variant=variant)
            stable = None
            # Warm repetitions check actual output, not just deterministic seeding.
            for temperature in ("first_call", "warm", "warm"):
                repeated,elapsed,peak,stage_memory = measure_call(call,ranker.stage_boundaries if ranker is not None else {})
                order = check_result(repeated, eligible)
                if variant == "most_played":
                    expected = sorted(eligible, key=lambda i: (-counts.get(i, 0), i))[:max(KS)]
                    require(order == expected, "most-played baseline policy violated")
                elif variant == "random_tagmatched":
                    tags = set(context["intent"]["tag_ids"])
                    pool = {item_key(i) for i in inputs["catalog"] if item_key(i) in eligible_set and tags.intersection(i["tag_ids"])}
                    require(set(order) <= pool and len(order) == min(max(KS), len(pool)), "tag-matched baseline policy violated")
                current = {k: v for k, v in repeated.items() if k != "timings"}
                require((stable is None or digest(current) == digest(stable))
                        and digest([inputs, context, m["ranker_config"]]) == before, "nonrepeatable or mutating ranker")
                stable = current
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
                measurements["samples"].append({"variant": variant, "context_id": c["id"], "temperature": temperature,
                                                "total_seconds": elapsed, "stage_seconds": repeated["timings"],
                                                "stage_measurements":stage_memory,"adapter_prepare_seconds":prepare_seconds,
                                                "prepared_request_seconds":prepare_seconds+elapsed if temperature == "first_call" else None,
                                                "traced_peak_bytes": peak, "process_peak_rss_bytes": rss if rss is None or sys.platform == "darwin" else rss * 1024})
            if ranker is not None:
                # Equivalent UTC spelling must not change a production result.
                equivalent_inputs, equivalent_context = copy.deepcopy(inputs), copy.deepcopy(context)
                for section in equivalent_inputs.values():
                    for row in section:
                        for field in ("known_at", "occurred_at", "effective_at", "available_at", "start_at", "end_at"):
                            if row.get(field) is not None:
                                row[field] = timestamp(row[field]).isoformat(timespec="microseconds").replace("+00:00", "Z")
                        if row.get("type") == "initial_state" and "watch" in row["value"]:
                            watch = row["value"]["watch"]
                            watch["last_at"] = timestamp(watch["last_at"]).isoformat(timespec="microseconds").replace("+00:00", "Z")
                for field in ("cutoff", "now"):
                    equivalent_context[field] = timestamp(context[field]).isoformat(timespec="microseconds").replace("+00:00", "Z")
                equivalent_before = digest([equivalent_inputs, equivalent_context])
                probe = ranker(equivalent_inputs, context=equivalent_context, config=m["ranker_config"], seed=seed, variant=variant)
                check_result(probe, eligible)
                require(digest({k:v for k,v in probe.items() if k != "timings"}) == digest(stable),
                        "production temporal contract mismatch: equivalent UTC instants change ranking")
                require(digest([equivalent_inputs, equivalent_context]) == equivalent_before
                        and digest([inputs, context, m["ranker_config"]]) == before, "production temporal probe mutated inputs")
            labels, outside = labels_at(snapshot, c, set(eligible))
            if sweep_plan is not None or diagnostic:
                labels,outside = {"implicit":{},"explicit":{}},0
            catalog = {item_key(r): r for r in inputs["catalog"]}
            features = {item_key(f): f for f in inputs["features"]}
            row = {"context_id": c["id"], "session_id": c["session_id"], "split": c["split"], "stratum": c["stratum"],
                   "input_hash": before, "ranking_seed": seed,
                   "temporal_invariance_checked": ranker is not None,
                   "structural_reachability":sweep.reachability(stable.get("route_inventory"),eligible),
                   "admission_trace":stable.get("admission_trace"),
                   "items": stable["items"], "source_counts": stable["source_counts"], "exclusions": stable["exclusions"],
                   "shared_exclusions": [{"kind": i[0], "id": i[1], "reason": reason} for i, reason in sorted(exclusions.items())],
                   "degenerate": stable.get("degenerate", False), "degenerate_by_kind": stable.get("degenerate_by_kind", {}),
                   "outcome_items_outside_eligible": outside, "metrics": {}}
            for k in KS:
                accuracy = {channel: metrics.accuracy(order, grades, k) for channel, grades in labels.items()}
                row["metrics"][str(k)] = {"accuracy": accuracy,
                    "source_survival": source_survival(stable, k),
                    "list": metrics.list_metrics(order[:k], eligible, counts, features, catalog, m["policy"]["diversity_space"],
                        counts_complete=not diagnostic or all(i in counts for i in eligible))}
                if diagnostic:
                    measured = row["metrics"][str(k)]["list"]
                    measured["novelty_smoothing"] = ("add_one_eligible_captured_play_counts" if measured["popularity_status"] == "measured"
                                                     else "unavailable_unknown_counts")
                    popularity = measured["popularity"]
                    popularity["captured_play_count_share"] = popularity.pop("training_interaction_share")
                aggregate[str(k)].append({"eligible": eligible, "order": order[:k]})
            rows.append(row)
            if sweep_plan is not None:
                stopped = sweep.stop_reason(aggregate[str(max(KS))],sweep_plan,paired=admission_comparison)
                if stopped:
                    break
        summary = {}
        for k in KS:
            acc = {}
            for channel in ("implicit", "explicit"):
                acc[channel] = {metric: metrics.session_mean([
                    {"session_id": r["session_id"], metric: r["metrics"][str(k)]["accuracy"][channel][metric]} for r in rows], metric)
                    for metric in ("recall", "ndcg", "map")}
            curve = [metrics.coverage(aggregate[str(k)][:i], m["policy"]["coverage_weights"]) for i in range(1, len(rows) + 1)]
            for index, point in enumerate(curve):
                point["context_count"] = index + 1
                point["new_recommended"] = {kind: point[kind]["recommended"] - (curve[index - 1][kind]["recommended"] if index else 0)
                                            for kind in ("video", "image")}
            inventories = [r["structural_reachability"] for r in rows]
            complete_routes = bool(rows) and all(r["status"] == "measured" for r in inventories)
            inventory_total = sum(r.get("eligible",0) for r in inventories)
            unreachable = sum(len(r.get("unreachable_ids",[])) for r in inventories)
            summary[str(k)] = {"accuracy": acc, "coverage": metrics.coverage(aggregate[str(k)], m["policy"]["coverage_weights"]),
                               "coverage_curve": curve,
                               "structural_unreachability":unreachable/inventory_total if complete_routes and inventory_total else None,
                               "structural_scope":"eligible-item-weighted frozen contexts, not global catalog exclusion",
                               "structural_status":"measured" if complete_routes else "unmeasured_without_complete_route_inventory"}
        report["variants"][variant] = {"status": "measured", "contexts": rows, "summary": summary,
            "sweep_stop_reason":stopped,"executed_contexts":len(rows)}
    report["requested_variants_complete"] = all(v["status"] == "measured" for v in report["variants"].values())
    report["offline_variants_complete"] = bool(contexts) and all(report["variants"].get(v, {}).get("status") == "measured" for v in VARIANTS)
    report["complete"] = False
    report["completion_gates"] = {"human_judgments": "unmeasured", "serving_parity": "unverified",
                                    "admission_isolation": "unverified", "structural_route_inventory": "not_supplied"}
    production_rows = [row for name in VARIANTS[2:] for row in report["variants"].get(name, {}).get("contexts", [])]
    if production_rows and all(row["structural_reachability"]["status"] == "measured" for row in production_rows):
        report["completion_gates"]["structural_route_inventory"] = "measured_for_executed_production_contexts"
    if admission_comparison:
        first,second = (report["variants"][name]["contexts"] for name in (admission_old, "admission_only"))
        require([(r["context_id"],r["input_hash"],r["ranking_seed"]) for r in first] ==
                [(r["context_id"],r["input_hash"],r["ranking_seed"]) for r in second],
                "admission contract mismatch: inputs/configuration/seed differ")
        report["admission_comparison"].update(sweep.admission_parity(first,second,require_support=not diagnostic))
        new_at_ten = []
        for old, corrected in zip(first, second):
            new = {item_key(i) for i in corrected["admission_trace"]["admitted_ids"]} - {item_key(i) for i in old["admission_trace"]["admitted_ids"]}
            new_at_ten.append(sum(item_key(i) in new for i in corrected["items"][:10]))
        report["admission_comparison"].update(newly_admitted_top10_contexts=sum(n > 0 for n in new_at_ten),
                                             newly_admitted_top10_items=sum(new_at_ten))
        report["completion_gates"]["admission_isolation"] = report["admission_comparison"]["status"]
    measurements["percentiles"] = {}
    for variant in variants:
        samples = [s for s in measurements["samples"] if s["variant"] == variant and s["temperature"] == "warm"]
        measurements["percentiles"][variant] = {}
        for stage in STAGES:
            values = [s["total_seconds"] if stage == "total" else s["stage_seconds"].get(stage) for s in samples]
            values = [v for v in values if v is not None]
            measurements["percentiles"][variant][stage] = {"samples": len(values),
                "p50_seconds": float(np.percentile(values, 50)) if values else None,
                "p95_seconds": float(np.percentile(values, 95)) if values else None}
    measurements["memory_scope"] = "traced stage/net/peak allocations at actual source boundaries; RSS is process lifetime high-water"
    measurements["request_scope"] = "frozen-input preparation and actual pure ranking adapter; HTTP/vendor I/O is not measured"
    measurements["cold_status"] = "unmeasured_requires_fresh_process_per_variant"
    return report, measurements


def aggregate_report(snapshot, report, measurements):
    """Allowlist only aggregate scalars; no IDs, traces, queries or source URLs."""
    from evaluation.sweep import STRATA
    require(report["snapshot_hash"] == measurements["snapshot_hash"] == digest(snapshot), "aggregate input mismatch")
    require(report["split"] in ("development", "heldout", "all"), "invalid aggregate split")
    def numbers(row, fields):
        result = {}
        for field in fields:
            value = row.get(field)
            require(value is None or finite(value, -float("inf")), "invalid aggregate scalar")
            result[field] = value
        return result
    def coverage(row):
        return {kind: numbers(row.get(kind, {}), ("eligible", "recommended", "coverage", "noncoverage", "gini"))
                for kind in ("video", "image")} | numbers(row, ("combined_coverage",))
    def average(values):
        values = [v for v in values if v is not None]
        require(all(finite(v, -float("inf")) for v in values), "invalid measured aggregate")
        return {"mean": sum(values)/len(values) if values else None, "measured_contexts": len(values)}
    m = snapshot["manifest"]
    out = {"schema_version": 1, "scope": "allowlisted_offline_aggregates", "snapshot_hash": report["snapshot_hash"],
           "code_revision": m["code_revision"], "source_revisions_hash": digest(m["source_revisions"]),
           "ranker_pins_hash": digest(m["rankers"]), "ranker_config_hash": digest(m["ranker_config"]),
           "split": report["split"], "current_state_only": m["mode"] == "current_state",
           "quality_lift": None, "human_judgments": "unmeasured", "search_quality": "not_evaluated",
           "catalog": {}, "variants": {}, "timings": {},
           "accuracy_scope": "no_temporal_labels" if m["mode"] == "current_state" else "observed_not_causal"}
    out["kind_weights"] = numbers(m["policy"]["coverage_weights"], ("video", "image"))
    metadata = m.get("capture_metadata", {})
    out["capture"] = numbers(metadata, ("elapsed_seconds", "source_equal_passes"))
    out["capture"]["orphan_rows_omitted"] = numbers(metadata.get("orphan_rows_omitted", {}),
        ("tag_aggregates", "video_embeddings", "image_embeddings", "identity_links", "watch_segments", "unsupported_embedding_models"))
    out["capture"]["producer_provenance"] = "not_verified_by_byte_capture"
    features = {item_key(f): f for f in snapshot["features"]}
    for kind in ("video", "image"):
        rows = [r for r in snapshot["catalog"] if r["kind"] == kind]
        fs = [features.get(item_key(r), {}) for r in rows]
        out["catalog"][kind] = {"items": len(rows), "eligible": sum(r["eligible"] for r in rows),
            "zero_or_unavailable_duration": sum(r["duration_s"] == 0 for r in rows),
            "missing_tags": sum(not f.get("tag_seconds") for f in fs),
            "missing_vectors": sum(not f.get("vectors") for f in fs),
            "missing_vector_spaces": {s: sum(s not in f.get("vectors", {}) for f in fs)
                                      for s in ("visual", "semvisual", "audioembed", "audiomix")},
            "identity_linked": sum(bool(f.get("identity_ids")) for f in fs),
            "duplicate_verified": sum(r["duplicate_verified"] for r in rows),
            "duplicate_grouped": sum(r["duplicate_group"] is not None for r in rows)}
        out["catalog"][kind]["known_play_counts"] = sum(e["type"] == "initial_state" and e["kind"] == kind
            and "play_count" in e["value"] for e in snapshot["events"])
    design = report["sweep"]
    out["sweep"] = numbers(design, ("target_contexts", "shortfall", "available_contexts", "minimum_contexts"))
    out["sweep"]["support"] = {s: int(design.get("support", design.get("strata", {})).get(s, 0)) for s in STRATA}
    out["sweep"]["missing_strata"] = [s for s in STRATA if not out["sweep"]["support"][s]]
    out["sweep"]["popularity_status"] = ("unavailable_unknown_counts" if design.get("popularity_status",
        m.get("diagnostic_design", {}).get("popularity_status")) == "unavailable_unknown_counts" else "measured")
    out["admission_comparison"] = numbers(report["admission_comparison"],
        ("unchanged_controls", "changed_admission_contexts", "newly_admitted_top10_contexts", "newly_admitted_top10_items"))
    out["admission_comparison"]["isolation_verified"] = report["admission_comparison"]["isolation_verified"] is True
    sources = ("tags", "visual", "voice", "sound", "images", "fallback", "explore", "control",
               "embedding_only", "most_played", "random_tagmatched")
    for variant in VARIANTS:
        result = report["variants"].get(variant)
        if result is None:
            continue
        if result["status"] != "measured":
            out["variants"][variant] = {"status": "unavailable",
                "play_counts_missing": result.get("reason") == "authoritative_play_counts_missing"}
            continue
        rows = result["contexts"]
        summary = {"status": "measured", "executed_contexts": len(rows), "ks": {},
                   "degenerate_contexts": sum(r["degenerate"] for r in rows)}
        for k in map(str, KS):
            source_rows = [r["metrics"][k] for r in rows]
            lists = [r["list"] for r in source_rows]
            original = result["summary"][k]
            summary["ks"][k] = {
                "unknown_popularity_contexts": sum(r.get("popularity_status") == "unavailable_unknown_counts" for r in lists),
                "coverage": coverage(original["coverage"]),
                "coverage_curve": [coverage(p) | numbers(p, ("context_count",)) for p in original["coverage_curve"]],
                "structural_unreachability": numbers(original, ("structural_unreachability",))["structural_unreachability"],
                "accuracy": {channel: {name: numbers(original["accuracy"][channel][name],
                    ("value", "measured_sessions", "measured_contexts")) for name in ("recall", "ndcg", "map")}
                    for channel in ("implicit", "explicit")},
                "list_metrics": {name: average(r.get(name) for r in lists) for name in
                    ("novelty_bits", "diversity", "valid_pairs", "missing_pairs", "total_pairs", "duplicate_items",
                     "duplicate_pairs", "duplicate_known_pairs", "duplicate_unknown_items", "duplicate_pair_rate")},
                "popularity": {name: average(r["popularity"].get(name) for r in lists) for name in
                    ("exposure_share", "exposure_share_min", "exposure_share_max",
                     "captured_play_count_share" if m["mode"] == "current_state" else "training_interaction_share")},
                "source_survival": {source: {stage: average(r["source_survival"].get(source, {}).get(stage)
                    for r in source_rows) for stage in ("pre_budget", "admitted", "returned")} for source in sources}}
        out["variants"][variant] = summary
        samples = [s for s in measurements["samples"] if s["variant"] == variant]
        out["timings"][variant] = {stage: numbers(measurements["percentiles"][variant][stage],
            ("samples", "p50_seconds", "p95_seconds")) for stage in STAGES}
        out["timings"][variant]["preparation"] = average(s["adapter_prepare_seconds"] for s in samples if s["temperature"] == "first_call")
        out["timings"][variant]["traced_peak_bytes"] = max((s["traced_peak_bytes"] for s in samples), default=None)
        out["timings"][variant]["process_peak_rss_bytes"] = max((s["process_peak_rss_bytes"] for s in samples if s["process_peak_rss_bytes"] is not None), default=None)
    out["timing_scope"] = "offline_python_rank_and_input_preparation_not_HTTP_database_or_cold_start"
    return out


def blank_judgments(form):
    blank = copy.deepcopy(form)
    for task in blank["tasks"]:
        task["judged_at"] = None
        task["preference"] = None
        for side in ("left", "right"):
            for item in task[side]["items"]:
                item["relevance"] = None
            for name in ("diversity", "non_obviousness", "session_fit"):
                task[side][name] = None
    return blank


def judge_export(snapshot, report, *, left, right, split):
    """not_before is a scheduling floor, not evidence of a human repeat delay.

    Actual repeat acceptance uses the supplied primary judged_at plus the frozen
    delay. The exporter cannot know when the primary will be answered.
    """
    validate(snapshot)
    require(split in ("development", "heldout") and left != right, "choose distinct systems and one judgment split")
    require(report["snapshot_hash"] == digest(snapshot), "report/snapshot mismatch")
    systems = report["variants"]
    require(all(s in systems and systems[s]["status"] == "measured" for s in (left, right)), "cannot judge an unavailable ranker")
    contexts = sorted((c for c in snapshot["contexts"] if c["judgment"] and c["split"] == split), key=lambda c: c["id"])
    lists = {s: {r["context_id"]: r["items"] for r in systems[s]["contexts"]} for s in (left, right)}
    tasks, answers = [], {}
    item_map, tag_map, context_map = {}, {}, {}
    def item_token(item):
        if item is None:
            return None
        token = digest([digest(snapshot), "blind_item", item_key(item)])[:20]
        item_map[token] = {"kind": item["kind"], "id": item["id"]}
        return token
    m = snapshot["manifest"]
    for index, c in enumerate(contexts + contexts[:3]):
        repeat = index >= len(contexts)
        task_id = digest([digest(snapshot), split, c["id"], repeat, left, right])[:20]
        pair = [left, right]
        random.Random(seeded(m["seed"], task_id, "blind_sides")).shuffle(pair)
        delay = timedelta(days=m["policy"]["repeat_delay_days"] if repeat else 0)
        context_token = digest([digest(snapshot), "blind_context", c["id"]])[:20]
        context_map[context_token] = {"context_id": c["id"], "intent": c["intent"]}
        tag_tokens = []
        for tag in c["intent"]["tag_ids"]:
            token = digest([digest(snapshot), "blind_tag", tag])[:20]
            tag_map[token] = tag
            tag_tokens.append(token)
        task = {"task_id": task_id, "context_id": context_token,
                "intent": {"tag_tokens": tag_tokens, "seed_token": item_token(c["intent"]["seed"])},
                "not_before": (timestamp(m["cutoff"]) + delay).isoformat().replace("+00:00", "Z"),
                "judged_at": None, "preference": None}
        for side, system in zip(("left", "right"), pair):
            require(c["id"] in lists[system], "report does not contain judgment context")
            task[side] = {"items": [{"kind": i["kind"], "item_token": item_token(i), "relevance": None} for i in lists[system][c["id"]][:10]],
                          "diversity": None, "non_obviousness": None, "session_fit": None}
        tasks.append(task)
        answers[task_id] = {"left": pair[0], "right": pair[1], "repeat": repeat, "context_id": c["id"]}
    form = {"schema_version": 1, "snapshot_hash": digest(snapshot), "split": split,
            "repeat_delay_days": m["policy"]["repeat_delay_days"], "repeat_anchor": "primary.judged_at",
            "instructions": "Relevance 1-5; list diversity/non_obviousness/session_fit 1-5; preference left/right/tie. Blanks are unmeasured. Repeat delay is measured from the provided primary judged_at, not snapshot/export time.", "tasks": tasks}
    key = {"schema_version": 1, "snapshot_hash": digest(snapshot), "form_hash": digest(form),
           "repeat_delay_days": m["policy"]["repeat_delay_days"], "assignments": answers,
           "items": item_map, "tags": tag_map, "contexts": context_map}
    return form, key


def judge_import(form, key):
    require(finite(key["repeat_delay_days"]) and key["repeat_delay_days"] > 0, "repeat_delay_days must be positive")
    require(finite(form.get("repeat_delay_days")) and form["repeat_delay_days"] > 0, "form repeat_delay_days must be positive")
    require(form["repeat_delay_days"] == key["repeat_delay_days"] and form.get("repeat_anchor") == "primary.judged_at",
            "repeat policy does not match frozen form")
    require(form["snapshot_hash"] == key["snapshot_hash"] and digest(blank_judgments(form)) == key["form_hash"], "blind fixture/key mismatch")
    require(len({t["task_id"] for t in form["tasks"]}) == len(form["tasks"]), "duplicate judgment task")
    primaries = {}
    records, consistent = [], []
    blanks = partial = 0
    for task in form["tasks"]:
        assignment = key["assignments"][task["task_id"]]
        cells = [task["preference"]]
        for side in ("left", "right"):
            cells += [i["relevance"] for i in task[side]["items"]]
            cells += [task[side][n] for n in ("diversity", "non_obviousness", "session_fit")]
        if all(v is None for v in cells):
            require(task["judged_at"] is None, "empty judgment has a completion timestamp")
            blanks += 1
            continue
        require(task["judged_at"] is not None and timestamp(task["judged_at"]) >= timestamp(task["not_before"]), "judgment submitted before allowed date")
        require(task["preference"] in (None, "left", "right", "tie"), "invalid pairwise preference")
        require(all(v is None or type(v) is int and 1 <= v <= 5 for v in cells[1:]), "ratings must be blank or integers 1-5")
        partial += int(any(v is None for v in cells))
        winner = assignment.get(task["preference"], task["preference"])
        record = {"task_id": task["task_id"], "context_id": task["context_id"], "repeat": assignment["repeat"],
                  "winner": winner, "systems": {assignment[side]: task[side] for side in ("left", "right")}}
        if assignment["repeat"]:
            primary = primaries.get(task["context_id"])
            require(primary is not None and timestamp(task["judged_at"]) >= timestamp(primary["judged_at"]) + timedelta(days=key["repeat_delay_days"]), "repeat requires a delayed completed primary")
            if winner is not None and primary["winner"] is not None:
                consistent.append(winner == primary["winner"])
        else:
            primaries[task["context_id"]] = {"judged_at": task["judged_at"], "winner": winner}
        records.append(record)
    return {"snapshot_hash": form["snapshot_hash"], "split": form["split"],
            "status": "unmeasured" if not records else "partial" if blanks or partial else "measured",
            "blank_tasks": blanks, "partial_tasks": partial, "records": records, "repeat_pairs": len(consistent),
            "repeat_preference_agreement": sum(consistent) / len(consistent) if consistent else None,
            "repeat_timing_evidence": "provided_timestamps_only",
            "quality_lift": None, "population_claim": None}


def judge_html(form, key, presentation):
    """Private, static local presenter. Only the blinded form reaches the browser."""
    from urllib.parse import urlsplit
    require(digest(blank_judgments(form)) == key["form_hash"], "presenter form/key mismatch")
    items, tags, repeats, primaries = {}, {}, {}, {}
    for token, item in key["items"].items():
        row = presentation["items"].get(f"{item['kind']}:{item['id']}")
        require(isinstance(row, dict) and isinstance(row.get("label"), str), "presenter needs a local item label and URL")
        require(urlsplit(row.get("url", "")).scheme in ("http", "https", "file"), "invalid presenter item URL")
        items[token] = {"label": row["label"], "url": row["url"]}
    for token, tag in key["tags"].items():
        require(isinstance(presentation.get("tags", {}).get(str(tag)), str), "presenter needs actual tag names")
        tags[token] = presentation["tags"][str(tag)]
    for task in form["tasks"]:
        assignment = key["assignments"][task["task_id"]]
        if assignment["repeat"]:
            repeats[task["task_id"]] = primaries[assignment["context_id"]]
        else:
            primaries[assignment["context_id"]] = task["task_id"]
    data = canonical({"form": form, "form_hash": key["form_hash"], "items": items, "tags": tags, "repeat_of": repeats}).decode().replace("<", "\\u003c")
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local blind judging</title><style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:auto;padding:20px;background:#f6f4ed;color:#222}
header{position:sticky;top:0;background:#f6f4ed;padding:12px 0;border-bottom:2px solid #222;z-index:1}
section{margin:24px 0;padding:16px;background:white;border:1px solid #ccc}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}
label{display:block;margin:8px 0}select,button{font:inherit;padding:6px}li{margin:12px 0}a{color:#154d85}
@media(max-width:600px){.pair{grid-template-columns:1fr}}.status{font-weight:bold}small{display:block}
</style><header><h1>Blind judging</h1><p>Open each item locally. Leave unknown judgments blank. Ratings: 1 (low) to 5 (high).</p>
<button id="save">Download judgment JSON</button><span id="message" role="status"></span></header><main id="tasks"></main>
<script type="application/json" id="data">""" + data + """</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const storage='blind-judging:'+data.form_hash;
let form=data.form;
try { const saved=JSON.parse(localStorage.getItem(storage)); if(saved&&saved.snapshot_hash===form.snapshot_hash&&saved.split===form.split) form=saved; } catch (_) {}
function node(tag,text){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;return el;}
function persist(){try{localStorage.setItem(storage,JSON.stringify(form));}catch(_){} }
function due(task){let at=Date.parse(task.not_before);const parent=data.repeat_of[task.task_id];if(parent){const p=form.tasks.find(t=>t.task_id===parent);if(!p||!p.judged_at)return Infinity;at=Math.max(at,Date.parse(p.judged_at)+form.repeat_delay_days*86400000);}return at;}
function hasValues(task){return task.preference!==null||['left','right'].some(side=>task[side].items.some(i=>i.relevance!==null)||['diversity','non_obviousness','session_fit'].some(k=>task[side][k]!==null));}
function choice(task,obj,key,label,values){const wrap=node('label',label+' '),select=node('select');select.append(node('option','Unmeasured'));select.options[0].value='';for(const value of values){const option=node('option',String(value));option.value=String(value);select.append(option);}select.value=obj[key]===null?'':String(obj[key]);select.addEventListener('change',()=>{if(Date.now()<due(task)){render();return;}obj[key]=select.value===''?null:key==='preference'?select.value:Number(select.value);task.judged_at=hasValues(task)?new Date().toISOString():null;persist();refreshLocks();});wrap.append(select);return wrap;}
function link(token){const item=data.items[token],a=node('a',item.label);a.href=item.url;a.target='_blank';a.rel='noopener noreferrer';return a;}
function refreshLocks(){for(const section of document.querySelectorAll('section')){const task=form.tasks.find(t=>t.task_id===section.dataset.task);const at=due(task),locked=Date.now()<at;section.querySelector('.content').hidden=locked;section.querySelectorAll('select').forEach(s=>s.disabled=locked);section.querySelector('.status').textContent=locked?(Number.isFinite(at)?'Available '+new Date(at).toLocaleString():'Waiting for an earlier task and the declared delay'):'Ready';}}
function render(){const root=document.getElementById('tasks');root.replaceChildren();form.tasks.forEach((task,index)=>{const section=node('section');section.dataset.task=task.task_id;section.append(node('h2','Context '+(index+1)),Object.assign(node('p'),{className:'status'}));const content=Object.assign(node('div'),{className:'content'});const intent=node('p','Intent: '+task.intent.tag_tokens.map(t=>data.tags[t]).join(', '));if(task.intent.seed_token){intent.append(document.createTextNode(' | Seed: '),link(task.intent.seed_token));}content.append(intent);const pair=Object.assign(node('div'),{className:'pair'});for(const side of ['left','right']){const block=node('div');block.append(node('h3',side==='left'?'Left list':'Right list'));const list=node('ol');for(const item of task[side].items){const li=node('li');li.append(link(item.item_token),choice(task,item,'relevance','Relevance',[1,2,3,4,5]));list.append(li);}block.append(list);for(const key of ['diversity','non_obviousness','session_fit'])block.append(choice(task,task[side],key,key.replaceAll('_',' '),[1,2,3,4,5]));pair.append(block);}content.append(pair,choice(task,task,'preference','Preference',['left','right','tie']));section.append(content);root.append(section);});refreshLocks();}
document.getElementById('save').addEventListener('click',()=>{persist();const blob=new Blob([JSON.stringify(form,null,2)+'\\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download='judgments-'+form.split+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('message').textContent=' Downloaded. Blanks remain unmeasured.';});
render();setInterval(refreshLocks,10000);
</script></html>"""


@contextmanager
def new_output(path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        yield stream


def emit(value, path=None):
    text = canonical(value).decode("utf-8") + "\n"
    if path:
        with new_output(path) as stream:
            stream.write(text)
    else:
        print(text, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "validate", "run", "diagnose", "prepare-current", "judge-export"):
        command = commands.add_parser(name)
        command.add_argument("snapshot", type=Path)
        command.add_argument("--output", type=Path)
        if name in ("run", "diagnose"):
            command.add_argument("--split", choices=("development", "heldout", "all"), default="development")
            command.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
            command.add_argument("--measurements", type=Path)
            command.add_argument("--source-root", type=Path)
            command.add_argument("--frozen-from", type=Path)
            command.add_argument("--admission-comparison", action="store_true")
            command.add_argument("--sweep", action="store_true")
            command.add_argument("--sweep-policy", type=Path)
            if name == "diagnose":
                command.add_argument("--private-output", type=Path, required=True, help="New private full ranking report")
        elif name == "prepare-current":
            command.add_argument("--plan", type=Path, required=True, help="JSON containing seed, policy and fully resolved ranker_config")
            command.add_argument("--source-root", type=Path, required=True)
            command.add_argument("--code-revision", required=True, help="Accepted full source commit; file bytes are pinned independently")
        elif name == "export":
            command.add_argument("--current-facts",type=Path)
            command.add_argument("--frozen-from",type=Path)
            command.add_argument("--events-db",type=Path)
            command.add_argument("--events-source-root",type=Path)
            command.add_argument("--events-sha256")
        elif name == "validate":
            command.add_argument("--freeze-output", type=Path)
            command.add_argument("--frozen-from", type=Path)
        elif name == "judge-export":
            command.add_argument("--report", type=Path, required=True)
            command.add_argument("--key-output", type=Path, required=True)
            command.add_argument("--split", choices=("development", "heldout"), required=True)
            command.add_argument("--left", required=True, choices=VARIANTS)
            command.add_argument("--right", required=True, choices=VARIANTS)
            command.add_argument("--html-output", type=Path, help="Optional new private local browser presenter")
            command.add_argument("--presentation", type=Path, help="Private item links and tag names, or collector facts containing presentation")
    command = commands.add_parser("judge-import")
    command.add_argument("form", type=Path)
    command.add_argument("--key", type=Path, required=True)
    command.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "judge-import":
            result = judge_import(read_json(args.form), read_json(args.key))
        else:
            snapshot = read_json(args.snapshot)
            if args.command == "prepare-current":
                from evaluation.capture import current_snapshot
                require(args.output is not None, "prepare-current requires an explicit new snapshot output")
                result = current_snapshot(snapshot, read_json(args.plan), source_root=args.source_root, code_revision=args.code_revision)
            elif args.command == "export":
                from datetime import timezone
                from evaluation.capture import export_snapshot, read_ledger
                require(args.output is not None,"export requires an explicit output path")
                require(bool(args.current_facts) != bool(args.frozen_from),"choose current-facts or frozen-from")
                require(bool(args.events_db) == bool(args.events_source_root) == bool(args.events_sha256),"event database requires explicit shared source root and hash")
                at = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
                ledger = read_ledger(args.events_source_root,args.events_sha256,args.events_db,through=timestamp(at).timestamp()) if args.events_db else None
                result = export_snapshot(snapshot,read_json(args.current_facts) if args.current_facts else None,
                    frozen=read_json(args.frozen_from) if args.frozen_from else None,ledger=ledger,captured_at=at)
            elif args.command == "validate":
                require(not (args.freeze_output and args.frozen_from), "choose freeze-output or frozen-from")
                result = validate(snapshot, planning=bool(args.freeze_output))
                if args.freeze_output:
                    emit(snapshot, args.freeze_output)
                if args.frozen_from:
                    result["prospective_anchor_hash"] = verify_prospective(snapshot, read_json(args.frozen_from))
            elif args.command in ("run", "diagnose"):
                from evaluation.sweep import DEFAULT_POLICY
                diagnostic = args.command == "diagnose"
                if diagnostic:
                    require(snapshot["manifest"]["mode"] == "current_state", "diagnose requires a prepared current-state snapshot")
                    require(args.output is not None and args.source_root is not None,
                            "diagnose requires explicit aggregate output and source root")
                    paths = [args.output, args.private_output] + ([args.measurements] if args.measurements else [])
                    require(len({p.resolve() for p in paths}) == len(paths) and all(not p.exists() and p.parent.is_dir() for p in paths),
                            "diagnostic outputs must be distinct new files in existing private directories")
                result, measurements = run(snapshot, split=args.split, variants=args.variants, source_root=args.source_root,
                                            frozen_from=read_json(args.frozen_from) if args.frozen_from else None,
                                            admission_comparison=args.admission_comparison or diagnostic,
                                           sweep_policy=read_json(args.sweep_policy) if args.sweep_policy else
                                               DEFAULT_POLICY if args.sweep else None)
                if args.measurements:
                    emit(measurements, args.measurements)
                if diagnostic:
                    emit(result, args.private_output)
                    result = aggregate_report(snapshot, result, measurements)
            else:
                require(args.output and args.output.resolve() != args.key_output.resolve(), "blind form and key require separate files")
                result, key = judge_export(snapshot, read_json(args.report), left=args.left, right=args.right, split=args.split)
                if args.html_output:
                    require(args.presentation is not None, "HTML presenter requires private real item links and tag names")
                    presentation = read_json(args.presentation)
                    html = judge_html(result, key, presentation.get("presentation", presentation))
                    paths = [args.output, args.key_output, args.html_output]
                    require(len({p.resolve() for p in paths}) == len(paths) and all(not p.exists() for p in paths),
                            "blind form, key and HTML must be distinct new files")
                    with new_output(args.html_output) as stream:
                        stream.write(html)
                emit(key, args.key_output)
        emit(result, args.output)
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

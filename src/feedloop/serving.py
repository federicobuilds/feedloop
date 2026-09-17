"""The explicit serving contract: request validation, the feed envelope,
cursor-stable continuation over a frozen generation, delivery recording from
immutable provenance, measured views, and the scorecard.

Only ``serve_feed`` records ``served``; ``view_request`` records a measured,
qualified view; pure ranking and envelope building record nothing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from typing import Any, Callable, Dict, Mapping

from feedloop import ledger
from feedloop import tuning
from feedloop.taste import DEFAULT_KINDS, eligible_ranked_items

FEED_REQUEST_FIELDS = {"limit", "images", "intent", "surface", "offset", "cursor", "session_id", "client_request_id",
                       "request_id", "eligibility", "filter_identity"}
SURFACES = ("home", "feed", "shuffle")
VIEW_SURFACES = ("home", "feed", "shuffle", "search")
VISIBILITY_POLICY = "foreground-60pct-1200ms-v1"
IMPRESSION_WINDOW_DAYS = 14.0
CURSOR_TTL_S = 3600.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _identity(value):
    return (isinstance(value, str) and 1 <= len(value) <= 128 and value.isascii() and value[0].isalnum()
            and all(c.isalnum() or c in "._:-" for c in value))


def feed_request(payload, *, kinds=DEFAULT_KINDS):
    """Validate the explicit delivery request; returns the normalized payload or an error envelope."""
    if not isinstance(payload, dict) or not set(payload) <= FEED_REQUEST_FIELDS:
        return {"status": "error", "error_code": "invalid_feed_request", "items": []}
    limit, offset = payload.get("limit", 24), payload.get("offset", 0)
    surface, images = payload.get("surface", "feed"), payload.get("images", False)
    if (type(limit) is not int or not 1 <= limit <= 60 or type(offset) is not int or offset < 0
            or surface not in SURFACES or type(images) is not bool):
        return {"status": "error", "error_code": "invalid_feed_request", "items": []}
    for key in ("session_id", "client_request_id", "request_id"):
        if not _identity(payload.get(key)):
            return {"status": "error", "error_code": "invalid_feed_request", "items": []}
    if offset and not isinstance(payload.get("cursor"), dict):
        return {"status": "error", "error_code": "invalid_cursor", "items": []}
    return {**payload, "limit": limit, "offset": offset, "surface": surface, "images": images}


def feed_intent(raw=None, *, kinds=DEFAULT_KINDS) -> Dict[str, Any]:
    """Validate temporary intent without storing or deriving preference from it."""
    primary = kinds[0]
    seed_field = f"seed_{primary}_id"
    data = {} if raw is None or raw == "" else json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict) or not set(data) <= {"revision", seed_field, "tag_ids", "excluded_items"}:
        raise ValueError("invalid intent")
    revision, seed = data.get("revision", 0), data.get(seed_field)
    tags, excluded = data.get("tag_ids", []), data.get("excluded_items", [])
    if type(revision) is not int or revision < 0 or (seed is not None and (type(seed) is not int or seed <= 0)):
        raise ValueError("invalid intent")
    if not isinstance(tags, list) or len(tags) > 100 or any(type(tag) is not int or tag <= 0 for tag in tags):
        raise ValueError("invalid intent")
    if not isinstance(excluded, list) or len(excluded) > 2000 or any(
            not isinstance(item, dict) or item.get("kind") not in kinds or type(item.get("id")) is not int or item["id"] <= 0
            for item in excluded):
        raise ValueError("invalid intent")
    return {"revision": revision, seed_field: seed, "tag_ids": sorted(set(tags)),
            "excluded_items": [{"kind": k, "id": i} for k, i in sorted({(item["kind"], item["id"]) for item in excluded})]}


def intent_fields(intent, *, kinds=DEFAULT_KINDS):
    """The ranking-context intent JSON for a validated feed intent."""
    primary = kinds[0]
    seed = intent[f"seed_{primary}_id"]
    fields = {f"seed_{primary}_ids": [seed] if seed is not None else [], "tag_ids": intent["tag_ids"]}
    for kind in kinds:
        fields[f"exclude_{kind}_ids"] = [item["id"] for item in intent["excluded_items"] if item["kind"] == kind]
    return fields


def feed_eligibility(raw, *, kinds=DEFAULT_KINDS):
    """Inline membership (bounded id lists) or one immutable snapshot reference."""
    eligibility = {} if raw is None or raw == "" else json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(eligibility, dict):
        raise ValueError("invalid eligibility")
    if "snapshot_id" in eligibility:
        identity = eligibility["snapshot_id"]
        if (set(eligibility) != {"snapshot_id"} or not isinstance(identity, str) or not identity.startswith("elig:v1:")
                or len(identity) != 72 or any(c not in "0123456789abcdef" for c in identity[8:])):
            raise ValueError("invalid eligibility")
        return dict(eligibility)
    if not set(eligibility) <= {kind + "_ids" for kind in kinds}:
        raise ValueError("invalid eligibility")
    for ids in eligibility.values():
        if ids is not None and (not isinstance(ids, list) or len(ids) > 2048 or any(type(i) is not int or i <= 0 for i in ids)):
            raise ValueError("invalid eligibility")
    return {kind: sorted(set(ids)) if ids is not None else None for kind, ids in eligibility.items()}


def feed_time(value, duration):
    """A seek requires a known finite duration and a finite position within it."""
    if isinstance(value, bool) or isinstance(duration, bool) or not isinstance(value, (int, float)) or not isinstance(duration, (int, float)):
        return None
    return value if math.isfinite(value) and math.isfinite(duration) and 0 <= value < duration else None


def feed_reason(explanation, names: Mapping[int, str]) -> str:
    if not explanation:
        return ""
    if explanation.get("control"):
        return "placebo pick"
    if explanation.get("explore"):
        return "outside your usual"
    bits = []
    top = sorted(explanation.get("tag_contributions") or [], key=lambda r: -abs(r.get("contribution", 0.0)))
    if top:
        name = names.get(top[0].get("tag_id"))
        if name:
            bits.append(name.lower())
    am = explanation.get("affinity_multiplier")
    if am and abs(am - 1.0) >= 0.01:
        pct = round((am - 1.0) * 100)
        bits.append(f"contributors {'+' if pct > 0 else ''}{pct}%")
    return " · ".join(bits)


def tag_names_for(explanation, names: Mapping[int, str], limit=3):
    """Known names only; unknown ids stay absent, never invented."""
    top = sorted(explanation.get("tag_contributions") or [], key=lambda r: -abs(r.get("contribution", 0.0)))
    tags = [names[c["tag_id"]].lower() for c in top[:limit] if names.get(c.get("tag_id"))]
    all_names = {str(c["tag_id"]): names[c["tag_id"]] for c in top if names.get(c.get("tag_id"))}
    return tags, all_names


def ranking_response(items, total, has_more, offset, *, next_offset=None, kinds=DEFAULT_KINDS):
    """Continuation envelope from generating facts, never a newly minted delivery id."""
    provenance = copy.deepcopy(items[0].get("provenance")) if items else None
    next_offset = offset + len(items) if next_offset is None else next_offset
    return {"items": items, "total": total, "has_more": has_more, "next_offset": next_offset if has_more else None,
            "next_cursor": {"generation_id": provenance["ranking_generation_id"], "offset": next_offset,
                            "after": f'{items[-1]["kind"]}:{items[-1]["id"]}'} if has_more and provenance else None,
            "ranking": provenance}


def continue_cursor(snapshot, cursor, *, offset, limit, cursor_context, reset_generation, now, current, kinds=DEFAULT_KINDS):
    """Skip-aware continuation over the frozen generation.

    ``current`` supplies today's constraints: ``present`` keys, ``groups`` (verified
    duplicate groups), ``allowed`` per kind, and ``excluded`` keys. Newly ineligible
    items are skipped without rebuilding or reordering; the offset advances by
    original positions, not returned count."""
    if (snapshot is None or snapshot["context"] != cursor_context or snapshot["reset_generation"] != reset_generation
            or now - snapshot["created_at"] > CURSOR_TTL_S or snapshot["generation_id"] != cursor["generation_id"]):
        raise ValueError("stale_ranking_cursor")
    rows = snapshot["items"]
    if not 0 < offset <= len(rows):
        raise ValueError("stale_ranking_cursor")
    key = lambda item: (item["kind"], int(item["id"]))
    if ":".join(map(str, key(rows[offset - 1]))) != cursor["after"]:
        raise ValueError("stale_ranking_cursor")
    excluded = set(current["excluded"]) | {key(item) for item in rows if key(item) not in current["present"]}
    previous = [key(item) for item in rows[:offset]] + list(current.get("seeds", ()))
    remaining = [key(item) for item in rows[offset:]]
    allowed = {kind: None if ids is None else frozenset(ids) for kind, ids in current["allowed"].items()}
    for groups in (snapshot["catalog"]["groups"], current["groups"]):
        remaining = eligible_ranked_items(remaining, allowed, excluded, groups, already_selected=previous)
    available = set(remaining)
    positions = [index for index in range(offset, len(rows)) if key(rows[index]) in available]
    selected = positions[:limit or 20]
    page = copy.deepcopy([rows[index] for index in selected])
    return ranking_response(page, offset + len(positions), len(positions) > len(selected), offset,
                            next_offset=selected[-1] + 1 if selected else len(rows), kinds=kinds)


def first_page(snapshot, *, limit, current, kinds=DEFAULT_KINDS):
    """The first page of a frozen generation served again, under today's constraints."""
    rows = snapshot["items"]
    key = lambda item: (item["kind"], int(item["id"]))
    excluded = set(current["excluded"]) | {key(item) for item in rows if key(item) not in current["present"]}
    allowed = {kind: None if ids is None else frozenset(ids) for kind, ids in current["allowed"].items()}
    remaining = [key(item) for item in rows]
    for groups in (snapshot["catalog"]["groups"], current["groups"]):
        remaining = eligible_ranked_items(remaining, allowed, excluded, groups, already_selected=list(current.get("seeds", ())))
    available = set(remaining)
    positions = [index for index in range(len(rows)) if key(rows[index]) in available]
    selected = positions[:limit or 20]
    page = copy.deepcopy([rows[index] for index in selected])
    return ranking_response(page, len(positions), len(positions) > len(selected), 0,
                            next_offset=selected[-1] + 1 if selected else len(rows), kinds=kinds)


def build_feed(ranked, *, names, offset, kinds=DEFAULT_KINDS):
    """Envelope over a ranking response: display fields, explanations, pagination checks."""
    d = ranked
    if not isinstance(d, dict) or ((d.get("error") or d.get("error_code") or d.get("errors")) and d.get("status") != "partial") \
            or not isinstance(d.get("items"), list):
        return {"schema_version": 1, "items": [], "profile": {}, "status": d.get("status", "error") if isinstance(d, dict) else "error",
                "error_code": d.get("error_code", "ranking_response_invalid") if isinstance(d, dict) else "ranking_response_invalid",
                "components": d.get("components", {}) if isinstance(d, dict) else {}}
    items = []
    primary = kinds[0]
    for sc in d["items"]:
        m = sc.get("explanation") or {}
        shared = {"request_id": None, "served_item_id": None, "viewed_event_id": None, "source_rank": sc.get("source_rank", m.get("position"))}
        if sc["kind"] != primary:
            items.append({**sc, **shared, "kind": sc["kind"], "id": int(sc["id"]), "score": sc.get("score"),
                          "title": sc.get("title") or f"Item {sc['id']}", "reason": sc.get("reason") or "Serving explanation unavailable",
                          "duration": 0, "best_t": None, "category": sc["kind"], "explore": bool(m.get("explore")),
                          "control": bool(m.get("control")), "rating100": sc.get("rating100")})
            continue
        dur = float(sc.get("duration_s") or 0.0)
        best_t = feed_time(sc.get("best_t"), dur)
        tags, tag_names = tag_names_for(m, names)
        items.append({**sc, **shared, "kind": sc["kind"], "id": int(sc["id"]), "score": sc.get("score"),
                      "title": sc.get("title") or f"Item {sc['id']}", "reason": sc.get("reason") or feed_reason(m, names),
                      "tags": tags, "tag_names": tag_names, "duration": round(dur, 1), "best_t": best_t,
                      "category": sc.get("category") or "other", "explore": bool(m.get("explore")),
                      "control": bool(m.get("control")), "rating100": sc.get("rating100")})
    components = dict(d.get("components") or {})
    has_more = d.get("has_more", False)
    next_offset, next_cursor = d.get("next_offset"), d.get("next_cursor")
    if has_more and (type(next_offset) is not int or next_offset < offset + len(items) or not items
                     or not isinstance(next_cursor, dict) or next_cursor.get("offset") != next_offset):
        return {"items": [], "status": "error", "error_code": "invalid_pagination"}
    return {"schema_version": 1, "request_id": None, "ranking_revision": (d.get("ranking") or {}).get("ranking_revision"),
            "feature_revision": None, "intent_revision": None,
            "status": "partial" if any((v.get("status") if isinstance(v, dict) else v) in ("error", "unavailable", "partial", "no-feature")
                                       for v in components.values()) else d.get("status", "ok" if items else "empty"),
            "components": components, "pagination": {"offset": offset, "has_more": bool(has_more),
                                                       "next_offset": next_offset if has_more else None,
                                                       "next_cursor": next_cursor if has_more else None},
            "items": items, "total": d.get("total"), "profile": ((d["items"] or [{}])[0].get("explanation") or {}).get("profile", {})}


def serve_feed(result, payload, *, ledger_path, resolve_eligibility, clock=time.time, kinds=DEFAULT_KINDS, recommender="feed"):
    """Record a delivery from the generation's immutable provenance, never from current tuner settings."""
    reference_backed = False
    primary = kinds[0]
    try:
        eligibility = feed_eligibility(payload.get("eligibility"), kinds=kinds)
        reference_backed = "snapshot_id" in eligibility
        items = result["items"]
        provenance = items[0].get("provenance")
        if not isinstance(provenance, dict) or provenance.get("revision_status") != "stable":
            raise ValueError("ranking_provenance_unavailable")
        fingerprint = digest(provenance)
        if any(digest(item.get("provenance")) != fingerprint for item in items):
            raise ValueError("mixed_ranking_provenance")
        intent = feed_intent(payload.get("intent"), kinds=kinds)
        expected_intent = intent_fields(intent, kinds=kinds)
        expected_eligible = dict(eligibility) if reference_backed else {
            kind: sorted(set(eligibility[kind + "_ids"])) if eligibility.get(kind + "_ids") is not None else None for kind in kinds}
        membership = resolve_eligibility(eligibility["snapshot_id"]) if reference_backed else expected_eligible
        membership = {kind: set(ids) if ids is not None else None for kind, ids in membership.items()}
        cursor = payload.get("cursor")
        if cursor is not None and (not isinstance(cursor, dict) or set(cursor) != {"generation_id", "offset", "after"}
                                   or type(cursor["offset"]) is not int or cursor["offset"] != payload.get("offset", 0)
                                   or not isinstance(cursor["after"], str) or len(cursor["after"]) > 64):
            raise ValueError("invalid_cursor")
        if cursor and cursor["generation_id"] != provenance["ranking_generation_id"]:
            raise ValueError("ranking_generation_changed")
        filter_identity = payload.get("filter_identity") or "intent:" + str(intent["revision"])
        provenance_intent = {k: list(v) for k, v in provenance["intent"].items()}
        if (any(provenance_intent[field] != expected_intent[field] for field in (f"seed_{primary}_ids", "tag_ids"))
                or provenance["config"]["include_images"] != payload.get("images", False)
                or cursor is None and (provenance_intent != expected_intent or provenance["eligible_ids"] != expected_eligible
                                       or provenance["filter_identity"] != filter_identity)):
            raise ValueError("ranking_context_conflict")
        delivery_context = {"intent": expected_intent, "eligible_ids": expected_eligible, "filter_identity": filter_identity}
        revisions, experiment = provenance["revisions"], provenance["experiment"]
        arms = {}
        if experiment is not None:
            if experiment["knob"] not in provenance["config"]:
                raise ValueError("experiment_config_unavailable")
            arms = {"base": {**provenance["config"], experiment["knob"]: experiment["base"]},
                    "cand": {**provenance["config"], experiment["knob"]: experiment["candidate"]}}
        served = []
        for position, item in enumerate(items):
            allowed = membership[item["kind"]]
            if (allowed is not None and item["id"] not in allowed) or item["id"] in expected_intent["exclude_" + item["kind"] + "_ids"]:
                raise ValueError("ranking_context_conflict")
            trace = item.get("explanation") or {}
            served.append({"kind": item["kind"], "id": item["id"],
                           "source_rank": trace.get("source_rank", trace.get("position", payload.get("offset", 0) + position)),
                           "score": item["score"], "arm": trace.get("arm", ""),
                           "control": bool(trace.get("control", False)), "explore": bool(trace.get("explore", False)),
                           "category": item.get("category") or "unspecified", "duration_s": float(item.get("duration_s") or 0.0),
                           "trace": {"explanation": trace, "ranking_provenance": provenance, "delivery_context": delivery_context}})
        context_id = digest([delivery_context, payload.get("limit", 24), payload.get("offset", 0), cursor])
        request = {"schema_version": 1, "request_id": payload["request_id"], "client_request_id": payload["client_request_id"],
                   "session_id": payload["session_id"], "created_at": clock(), "surface": payload.get("surface", "feed"),
                   "recommender": recommender, "context_id": context_id,
                   "ranking_content_id": digest([provenance["ranking_generation_id"], payload["session_id"], context_id, served]),
                   "config": provenance["config"], "config_hash": provenance["config_hash"],
                   "ranker_revision": provenance["ranking_revision"],
                   "feature_revision": digest({k: revisions.get(k) for k in ("features", "tag_projection", "catalog_fingerprints")}),
                   "preference_revision": digest({k: revisions.get(k) for k in ("watch", "item_preferences", "secondary_preferences", "views")}),
                   "preference_cutoff": provenance["captured_at"], "seed": provenance["seed"],
                   "experiment_id": experiment["id"] if experiment else None, "arms": arms,
                   "intent_revision": digest([provenance["intent"], provenance["filter_identity"]]),
                   "eligibility_revision": digest(provenance["eligible_ids"])}
        bindings = {"generation": provenance["eligible_ids"].get("snapshot_id") if isinstance(provenance["eligible_ids"], dict) else None,
                    "delivery": expected_eligible.get("snapshot_id")}
        try:
            refs = ledger.record_served(ledger_path, request=request, items=served, eligibility_snapshots=bindings, kinds=kinds)
        except ledger.ContractError as exc:
            if str(exc) != "request_conflict":
                raise
            evidence = ledger.read_evidence(ledger_path, since_ts=0, through_ts=clock())
            original = next((row for row in evidence.get("requests", []) if row["request_id"] == request["request_id"]), None)
            if (not original or original["session_id"] != request["session_id"]
                    or original["client_request_id"] != request["client_request_id"]):
                raise
            request["created_at"] = original["created_at"]
            refs = ledger.record_served(ledger_path, request=request, items=served, eligibility_snapshots=bindings, kinds=kinds)
        for item, recorded in zip(items, served):
            item["request_id"] = request["request_id"]
            item["served_item_id"] = refs[item["kind"], item["id"]]
            item["source_rank"] = recorded["source_rank"]
            item.pop("viewed_event_id", None)
        components = dict(result.get("components") or {})
        components["delivery"] = {"status": "ok"}
        return {**result, "request_id": request["request_id"], "client_request_id": request["client_request_id"],
                "session_id": request["session_id"], "ranking_content_id": request["ranking_content_id"],
                "ranking_revision": request["ranker_revision"], "feature_revision": request["feature_revision"],
                "preference_revision": request["preference_revision"], "intent_revision": request["intent_revision"],
                "components": components,
                "status": "partial" if any((v.get("status") if isinstance(v, dict) else v) != "ok" for v in components.values()) else "ok"}
    except (ledger.ContractError, ValueError, KeyError, TypeError, IndexError) as exc:
        code = str(exc) if isinstance(exc, ledger.ContractError) or str(exc) in (
            "ranking_context_conflict", "mixed_ranking_provenance", "invalid_cursor", "ranking_generation_changed") else "ranking_provenance_unavailable"
        conflict = code in ("request_conflict", "request_items_conflict", "cached_content_conflict", "experiment_conflict",
                            "ranking_context_conflict", "mixed_ranking_provenance", "invalid_cursor", "ranking_generation_changed")
        for item in result.get("items", []):
            item["request_id"] = None
            item["served_item_id"] = None
            item.pop("viewed_event_id", None)
        return {**result, "request_id": None, "status": "error" if conflict else "partial", "error_code": code,
                "items": [] if conflict or reference_backed else result.get("items", []),
                "components": {**result.get("components", {}), "delivery": {"status": "unavailable", "error_code": code}}}


def view_request(payload, *, ledger_path, clock=time.time, kinds=DEFAULT_KINDS):
    """Translate a measured presentation into the ledger's validated viewed event."""
    try:
        required = {"client_event_id", "session_id", "request_id", "served_item_id", "surface", "position", "dwell_ms",
                    "visible_fraction", "visibility_policy", "kind", "item_id", "occurred_at"}
        if not isinstance(payload, dict) or set(payload) != required or payload["visibility_policy"] != VISIBILITY_POLICY:
            raise ValueError("invalid view")
        if payload["surface"] not in VIEW_SURFACES or payload["occurred_at"] > clock() + 5:
            raise ValueError("invalid view")
        event_id = ledger.record_event(ledger_path, event={
            "event_type": "viewed", "source": "browser", "source_event_id": payload["client_event_id"],
            "session_id": payload["session_id"], "kind": payload["kind"], "item_id": payload["item_id"],
            "occurred_at": payload["occurred_at"], "request_id": payload["request_id"], "parent_id": payload["served_item_id"],
            "payload": {"visible_fraction": payload["visible_fraction"], "dwell_ms": payload["dwell_ms"], "foreground": True,
                        "display_rank": payload["position"], "placement": payload["surface"]}}, kinds=kinds)
        return {"status": "confirmed", "event_id": event_id}
    except ledger.ContractError as exc:
        return {"status": "unavailable" if str(exc) in ("store_unavailable", "schema_unavailable") else "error", "error_code": str(exc)}
    except (ValueError, TypeError):
        return {"status": "error", "error_code": "invalid_view"}


def build_fatigue(ledger_path, *, now, titles: Callable[[Any], str] | None = None, recommender="feed"):
    """Qualified-view counts as distinct UTC days, from the ledger only."""
    try:
        snapshot = ledger.read_view_counts(ledger_path, since_ts=now - IMPRESSION_WINDOW_DAYS * 86400.0, through_ts=now, recommender=recommender)
        if snapshot.get("status") != "ok":
            return {"status": "unavailable", "items": [], "error_code": "view_store_unavailable"}
        rows = sorted(snapshot["counts"].items(), key=lambda row: (-row[1], row[0]))[:60]
    except Exception:
        return {"status": "unavailable", "items": [], "error_code": "view_store_unavailable"}
    return {"status": "ok" if rows else "empty", "source": "qualified_views", "unit": "distinct_utc_days",
            "window_days": IMPRESSION_WINDOW_DAYS,
            "items": [{"kind": kind, "id": item_id, "days_shown": days,
                       "title": titles((kind, item_id)) if titles else f"{kind} {item_id}"} for (kind, item_id), days in rows]}


def build_scorecard(tuner: tuning.Tuner, *, ledger_path, clock=time.time):
    """Display actual knobs and the ledger; never infer attributed outcomes from history.

    The trial summary uses the completed attribution boundary and the tuner's own
    evidence reader, summarizer and ``cumulative_facts``; a populated reward
    summary is not permission to promote."""
    with tuner.lock:
        state, tuned = tuner.snapshot()
        active = state[0] if state else None
        entry = tuner.registry_entry(active)
        arms, ledger_rows = tuner.display()
    current = arms.get(active, (None, None, None))
    now = clock()
    since = float(current[2] or 0)
    evidence = tuner.read_evidence(ledger_path, since_ts=since, through_ts=now)
    completed = evidence.get("attribution_run") if evidence.get("status") == "ok" else None
    cumulative = None
    if completed and since <= float(completed["through_ts"]) <= now:
        through = float(completed["through_ts"])
        evidence = tuner.read_evidence(ledger_path, since_ts=since, through_ts=through)
        viewed = set(evidence.get("viewed_ids") or [])
        items = {(row["kind"], row["item_id"]) for row in evidence.get("events", []) if row["event_id"] in viewed}
        cumulative = tuner.cumulative_facts(items, evidence["through_ts"])
    summary = tuner.summarize_trials(evidence, verdict=tuner.verdict, trial_reward=tuner.reward, cumulative_at_cutoff=cumulative)
    promotion = (evidence.get("valid") is True and evidence.get("promotion_enabled") is True
                 and summary.get("valid") is True and summary.get("promotion_enabled") is True)
    # 2026-09-16: per-trial facts are shown whenever the evidence read succeeded; the
    # validity reasons and gates above stay separate and promotion_eligible stays False.
    arm_summary = {}
    if summary.get("status") == "ok":
        for arm in ("base", "cand"):
            trials = [row for row in summary["trials"] if row["arm"] == arm and row["reward"] is not None]
            arm_summary[arm] = {"trials": len(trials), "successes": sum(bool(row["liked"]) for row in trials),
                                "mean_reward": sum(row["reward"] for row in trials) / len(trials) if trials else None,
                                "like_rate": sum(bool(row["liked"]) for row in trials) / len(trials) if trials else None}
    return {
        "gate": {"min_trials_per_arm": tuning.TUNER_MIN_TRIALS, "ripen_hours": tuning.TUNER_RIPEN_S / 3600.0},
        "capture": ledger.read_capture_readiness(ledger_path),
        "automation": {"enabled": bool(tuner.automatic), "interval_hours": tuning.TUNER_EVAL_EVERY_S / 3600.0,
                       "min_trials_per_arm": tuning.TUNER_MIN_TRIALS, "min_sessions_per_arm": tuning.TUNER_MIN_SESSIONS,
                       "ripen_hours": tuning.TUNER_RIPEN_S / 3600.0},
        "evidence": {"valid": evidence.get("valid") is True and summary.get("valid") is True,
                     "status": evidence.get("status", "unavailable"), "through_ts": evidence.get("through_ts"),
                     "validity_reasons": list(dict.fromkeys(evidence.get("validity_reasons", []) + summary.get("validity_reasons", []))),
                     "promotion_reasons": list(dict.fromkeys(evidence.get("promotion_reasons", ["attributed_evidence_unavailable"])
                                                             + summary.get("promotion_reasons", [])))},
        "trials": summary.get("trials", []),
        "tuner": {"knob": active, "rotation": [e["knob"] for e in tuner.registry], "stalls": state[1] if state else None,
                  "tuned_values": tuned, "default": tuning.knob_default(active) if entry else None,
                  "step": entry["step"] if entry else None, "base": current[0], "candidate": current[1],
                  "experiment_started": current[2], "promotion_eligible": promotion, "arms": arm_summary,
                  "gate_progress": None, "ledger": ledger_rows,
                  "knobs": [{"knob": row["knob"], "default": tuning.knob_default(row["knob"]),
                             "settled": tuned.get(row["knob"], tuning.knob_default(row["knob"])),
                             "base": arms.get(row["knob"], (None, None, None))[0],
                             "candidate": arms.get(row["knob"], (None, None, None))[1]} for row in tuner.registry]},
        "exposure": {"status": "unavailable"}, "organic": {"status": "unavailable"},
    }


__all__ = ["FEED_REQUEST_FIELDS", "SURFACES", "VIEW_SURFACES", "VISIBILITY_POLICY", "IMPRESSION_WINDOW_DAYS", "CURSOR_TTL_S",
           "digest", "feed_request", "feed_intent", "intent_fields", "feed_eligibility", "feed_time", "feed_reason",
           "tag_names_for", "ranking_response", "continue_cursor", "first_page", "build_feed", "serve_feed", "view_request",
           "build_fatigue", "build_scorecard"]

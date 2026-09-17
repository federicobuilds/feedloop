"""The facade: one Engine over the six slots, composing ranking, profiles, the
ledger, attribution, tuning, serving and discovery. It imports nothing but this
package, stdlib and NumPy, and never executes a host.

Stores are explicit: ``initialize_stores`` creates the event ledger and the
tuner store; the constructor, imports and reads create nothing.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import random
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from feedloop import catalog as catalog_module
from feedloop import discovery, ledger, ranking, serving, tuning
from feedloop.slots import DEFAULT_SPACE_ROLES
from feedloop.taste import DEFAULT_KINDS

RANKING_REVISION = "feedloop_feed/v1"
VIEW_WINDOW_DAYS = 14
DEFAULT_CONFIG = dict(
    half_life_days=21.0, min_watch_seconds=20.0, finished_ratio=0.45, abandon_ratio=0.15,
    dislike_min_watch_seconds=60.0, short_watch_ratio=0.5, history_limit=600, rating_strength=1.0,
    dislike_strength=1.0, profile_tags=40, candidate_pool=500, bodyparts_weight=0.3, max_tag_share=0.35,
    length_floor_seconds=120.0, diversity=0.35, calibration=0.25, cooldown_days=45.0, recovery_days=120.0,
    impression_discount=0.95, image_events_enabled=True, include_images=False, images_share=0.2,
    explore_slots=2, control_rate=1.0,
)
DEFAULT_ATTRIBUTION = {"window_s": ledger.ATTRIBUTION_WINDOW_S, "policy_revision": ledger.ATTRIBUTION_POLICY_REVISION,
                       "min_advance_s": ledger.ATTRIBUTION_MIN_ADVANCE_S}
_OMITTED = object()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()).hexdigest()


def initialize_stores(*, ledger_path, tuner_path, cutover_ts, clock=time.time, registry=tuning.TUNER_REGISTRY):
    """Explicit local store creation: the event ledger with its cutover, and the tuner's
    first experiment. The only place either file is created."""
    ledger.initialize_event_store(str(ledger_path), cutover_ts=cutover_ts)
    tuner = tuning.Tuner(str(tuner_path), ledger_path=str(ledger_path), cumulative_facts=lambda items, cutoff: {"cutoff_ts": cutoff, "items": {}},
                         registry=registry, clock=clock)
    tuner.initialize()


class Engine:
    def __init__(self, *, catalog, signals, spaces, encoder=None, links=None, annotator=None, ledger_path, tuner_path,
                 kinds=DEFAULT_KINDS, tag_namespace="", config=None, space_roles=DEFAULT_SPACE_ROLES,
                 read_current: Callable | None = None, apply_change: Callable | None = None,
                 attribution: Mapping[str, Any] | None = None, automatic_tuning=True, clock: Callable[[], float] = time.time):
        if (read_current is None) != (apply_change is None):
            raise ValueError("feedback callbacks are supplied together")
        self.catalog, self.signals, self.spaces = catalog, signals, spaces
        self.encoder, self.links, self.annotator = encoder, links, annotator
        self.ledger_path, self.tuner_path = str(ledger_path), str(tuner_path)
        self.kinds = tuple(kinds)
        self.tag_namespace = tag_namespace
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.roles = dict(space_roles)
        self.read_current, self.apply_change = read_current, apply_change
        self.attribution = {**DEFAULT_ATTRIBUTION, **(attribution or {})}
        self.clock = clock
        self.tuner = tuning.Tuner(self.tuner_path, ledger_path=self.ledger_path, cumulative_facts=self.cumulative_facts,
                                  clock=clock, automatic=automatic_tuning)
        self.sources = discovery.Sources(catalog=catalog, spaces=spaces, signals=signals, encoder=encoder, links=links,
                                         kinds=self.kinds, roles=self.roles, clock=clock)
        self.lock = threading.RLock()
        self.reset_generation = 0
        self._cursors: dict[str, dict] = {}

    # ----------------------------------------------------------------- facts
    @property
    def primary(self):
        return self.kinds[0]

    def _tag_name(self, name):
        return name[:-len(self.tag_namespace)] if self.tag_namespace and name.endswith(self.tag_namespace) else name

    def tag_names(self):
        return {int(t): self._tag_name(str(n)) for t, n in self.catalog.tag_names().items()}

    def resolve_eligibility(self, snapshot_id):
        return ledger.read_eligibility_snapshot(self.ledger_path, snapshot_id=snapshot_id)["eligible_ids"]

    def cumulative_facts(self, items, cutoff_ts):
        """Current Catalog/Signals values labelled with the exact evidence cutoff."""
        keys = sorted(set(items))
        rows = self.signals.read(keys)["rows"] if keys else {}
        catalog_rows = self.sources.rows(keys)
        facts = {}
        for key in keys:
            row, item = rows.get(key) or {}, catalog_rows.get(key) or {}
            watch = row.get("watch") or {}
            facts[key] = {"watched_s": float(watch.get("watched_s", 0.0)) if key[0] == self.primary else 0.0,
                          "duration_s": float(item.get("duration_s") or 0.0) if key[0] == self.primary else 0.0,
                          "rating": row.get("rating"), "engagement_count": int(row.get("engagement_count") or 0)}
        return {"cutoff_ts": cutoff_ts, "items": facts}

    def _evidence(self, rows, observed_at):
        known = _iso(observed_at)
        events = []
        for key, row in sorted(rows.items()):
            value = {"rating": row.get("rating"), "engagement_count": int(row.get("engagement_count") or 0)}
            watch = row.get("watch")
            if watch and key[0] == self.primary:
                value["watch"] = {"watched_s": float(watch["watched_s"]), "last_at": _iso(watch["last_at"]),
                                  "visit_days": sorted(set(int(d) for d in watch.get("visit_days") or ()))}
            events.append({"event_id": f"initial:{key[0]}:{key[1]}", "type": "initial_state", "kind": key[0], "id": key[1],
                           "occurred_at": None, "known_at": known, "value": value})
        return events

    def _views(self, now):
        """Qualified ledger views inside the fatigue window, with their real instants, plus the
        distinct-day counts whose digest is the view revision. Disabled when the discount is off."""
        if self.config["impression_discount"] >= 0.999:
            return {"status": "disabled", "events": [], "revision": _digest([])}
        since = max(0.0, now - VIEW_WINDOW_DAYS * 86400.0)
        counts = ledger.read_view_counts(self.ledger_path, since_ts=since, through_ts=now)
        if counts["status"] != "ok":
            return {"status": counts["status"], "events": [], "revision": None}
        evidence = ledger.read_evidence(self.ledger_path, since_ts=since, through_ts=now)
        if evidence.get("status") != "ok":
            return {"status": evidence.get("status", "unavailable"), "events": [], "revision": None}
        events = [{"event_id": f"visible:{row['event_id']}", "type": "visible", "kind": row["kind"], "id": row["item_id"],
                   "occurred_at": _iso(row["occurred_at"]), "known_at": _iso(max(row["received_at"], row["occurred_at"])), "value": None}
                  for row in evidence["events"] if row["event_type"] == "viewed" and row["kind"] in self.kinds]
        revision = _digest(sorted((f"{kind}:{item_id}", n) for (kind, item_id), n in counts["counts"].items()))
        return {"status": "ok", "events": events, "revision": revision}

    def _committed_revisions(self, signals, views):
        """The generation tokens a build must see unchanged before it may publish."""
        spaces = {space: self.spaces.revision(space) for space in sorted(set(self.spaces.spaces()) & set(self.roles.values()))}
        return {"features": _digest(sorted(spaces.items())) if all(v is not None for v in spaces.values()) else None,
                "signals": _digest([signals["observed_at"], sorted((f"{k[0]}:{k[1]}", v) for k, v in signals["rows"].items())]),
                "views": views["revision"]}

    def _ranking_inputs(self, keys, pin):
        rows = self.sources.rows(keys)
        features = self.catalog.features(keys)
        vectors = {}
        model_revisions = {}
        available = set(self.spaces.spaces())
        for role, space in self.roles.items():
            if space not in available:
                continue
            model_revisions[space] = self.spaces.revision(space)
            for key, vector in self.sources.means(space, keys).items():
                vectors.setdefault(key, {})[space] = vector
        links = self.sources.trusted_links(keys)
        catalog_rows = [{"kind": k, "id": i, "duration_s": float(rows[(k, i)].get("duration_s") or 0.0), "eligible": True,
                         "duplicate_group": pin["groups"].get((k, i)), "duplicate_verified": (k, i) in pin["groups"]}
                        for k, i in keys if (k, i) in rows]
        feature_rows = []
        for key in keys:
            f = features.get(key) or {}
            feature_rows.append({"kind": key[0], "id": key[1], "tag_seconds": dict(f.get("tag_seconds") or {}),
                                 "watched_tag_seconds": f.get("watched_tag_seconds"), "tag_categories": dict(f.get("tag_categories") or {}),
                                 "vectors": vectors.get(key, {}), "identity_ids": sorted(links.get(key, ())),
                                 "revision": _digest(sorted((s, r) for s, r in model_revisions.items()), ), "model_revision": None})
        return rows, catalog_rows, feature_rows, model_revisions

    # ------------------------------------------------------------------ feed
    def feed(self, request):
        payload = serving.feed_request(request, kinds=self.kinds)
        if payload.get("status") == "error":
            return payload
        try:
            intent = serving.feed_intent(payload.get("intent"), kinds=self.kinds)
        except (ValueError, TypeError):
            return {"schema_version": 1, "items": [], "status": "error", "error_code": "invalid_intent"}
        try:
            eligibility = serving.feed_eligibility(payload.get("eligibility"), kinds=self.kinds)
        except (ValueError, TypeError):
            return {"schema_version": 1, "items": [], "status": "error", "error_code": "invalid_eligibility"}
        filter_identity = payload.get("filter_identity")
        if filter_identity is not None and not serving._identity(filter_identity):
            return {"status": "error", "error_code": "invalid_filter_identity", "items": []}
        config_fields = {
            "intent": json.dumps(serving.intent_fields(intent, kinds=self.kinds), sort_keys=True, separators=(",", ":")),
            "eligibility": json.dumps(eligibility, sort_keys=True, separators=(",", ":")),
            "filter_identity": filter_identity or "intent:" + str(intent["revision"]),
            "session_id": payload["session_id"], "client_request_id": payload["client_request_id"], "request_id": payload["request_id"],
            "cursor": json.dumps(payload["cursor"], sort_keys=True, separators=(",", ":")) if payload.get("cursor") is not None else "",
        }
        try:
            ranked = self._rank(config_fields, limit=payload["limit"], offset=payload["offset"], include_secondary=payload["images"])
        except ValueError as error:
            code = str(error) if str(error) in ("stale_ranking_cursor", "cursor_offset_mismatch", "invalid cursor") else "ranking_contract_unavailable"
            return {"items": [], "status": "error", "error_code": code}
        except Exception as exc:
            # name the failure by type; the message only when it is one of our own
            # snake_case codes, never a driver string, path, query or traceback
            message = str(exc)
            code = message if re.fullmatch(r"[a-z][a-z0-9_]{2,64}", message) else None
            detail = type(exc).__name__ + (": " + code if code else "")
            return {"items": [], "profile": {}, "status": "unavailable", "error": "recommender unavailable", "error_detail": detail}
        result = serving.build_feed(ranked, names=self.tag_names(), offset=payload["offset"], kinds=self.kinds)
        if result.get("items") and result.get("status") in ("ok", "partial"):
            result = serving.serve_feed(result, payload, ledger_path=self.ledger_path, resolve_eligibility=self.resolve_eligibility,
                                        clock=self.clock, kinds=self.kinds)
            self.tuner.maybe_tick()
        return result

    def _rank(self, cfg, *, limit, offset, include_secondary):
        now = self.clock()
        context = ranking.parse_context(cfg, decode=json.loads, resolve_eligibility=self.resolve_eligibility, kinds=self.kinds)
        resolved = self.tuner.resolve_knobs({k: self.config[k] for k in tuning.KNOB_DEFAULTS if k in self.config and k not in DEFAULT_CONFIG})
        config = {**self.config, **resolved["values"], "include_images": bool(include_secondary),
                  "vector_spaces": dict(self.roles), "selection_contract": "rank_page/v1"}
        experiment = resolved["experiment"]
        requested_kinds = self.kinds if include_secondary else (self.primary,)
        cursor = context["cursor"]
        if cursor is not None and cursor["offset"] != offset:
            raise ValueError("cursor_offset_mismatch")
        all_keys = [(r["kind"], r["id"]) for r in catalog_module.read_catalog(self.catalog, kinds=self.kinds)]
        pin = catalog_module.fingerprint_snapshot(self.catalog, all_keys, kinds=self.kinds)
        keys = sorted(pin["present"])
        signals = self.signals.read()
        observed_at = float(signals["observed_at"])
        if observed_at >= now:
            raise ValueError("signals_observed_after_now")
        views = self._views(now)
        # commit the generation tokens before any feature hydration: a revision that moves while
        # vectors load is caught by the post-build comparison, never published under the new token
        committed = self._committed_revisions(signals, views)
        rows, catalog_rows, feature_rows, model_revisions = self._ranking_inputs(keys, pin)
        inputs = {"catalog": catalog_rows, "features": feature_rows,
                  "evidence": self._evidence(signals["rows"], observed_at) + views["events"]}
        content_context = {**context, "now": _iso(now), "cutoff": _iso(now), "kinds": requested_kinds, "page_size": limit}
        # the frozen generation is bound to its session, page size, kinds, resolved knobs and
        # generating tag/seed intent; current membership and exclusions are rechecked per page
        cursor_context = (_digest(sorted(config.items())), limit, include_secondary, context["session_id"],
                          tuple(tuple(context["intent"][name]) for name in ("tag_ids", f"seed_{self.primary}_ids")))
        content_key = (cursor_context, ranking.context_key(context))
        with self.lock:
            if cursor is not None:
                snapshot = self._cursors.get(cursor["generation_id"])
            else:
                # a repeated first page (a retry, a second tab of the same session) reuses the
                # frozen generation while it is fresh and its catalog and reset state are unchanged
                snapshot = next((entry for entry in self._cursors.values() if entry["content"] == content_key
                                 and entry["reset_generation"] == self.reset_generation and entry["catalog"] == pin
                                 and entry["revisions"] == committed
                                 and now - entry["created_at"] <= serving.CURSOR_TTL_S), None)
        if cursor is not None or snapshot is not None:
            eligible = set(ranking.shared_hard_eligibility(inputs, context={**content_context, "limit": 0, "offset": 0},
                                                            config={**config, "experiment": None}, kinds=self.kinds))
            excluded = {key for key in keys if f"{key[0]}:{key[1]}" not in eligible}
            current = {"present": pin["present"], "groups": pin["groups"], "allowed": context["eligible_ids"], "excluded": excluded,
                       "seeds": [(self.primary, sid) for sid in context["intent"][f"seed_{self.primary}_ids"]]}
            if cursor is not None:
                return serving.continue_cursor(snapshot, cursor, offset=offset, limit=limit, cursor_context=cursor_context,
                                               reset_generation=self.reset_generation, now=now, current=current, kinds=self.kinds)
            return serving.first_page(snapshot, limit=limit, current=current, kinds=self.kinds)
        seed = random.getrandbits(63)
        generating_config = dict(config)
        generating_config["experiment"] = None
        config_hash = _digest(generating_config)
        if experiment is not None:
            experiment = dict(experiment)
            experiment["id"] = _digest(["feedloop_feed_provenance_v1", config_hash, ranking.context_key(context), experiment])
            generating_config["experiment"] = {k: experiment[k] for k in ("id", "knob", "base", "candidate")}
            config_hash = _digest(generating_config)
        revisions = {"features": committed["features"], "tag_projection": "current",
                     "catalog_fingerprints": pin["revision"], "watch": committed["signals"], "item_preferences": committed["signals"],
                     "secondary_preferences": committed["signals"], "views": committed["views"]}
        provenance = json.loads(json.dumps({
            "ranking_revision": RANKING_REVISION, "config": generating_config, "config_hash": config_hash, "seed": seed,
            "revisions": revisions, "captured_at": now,
            "intent": {k: list(v) for k, v in context["intent"].items()},
            "eligible_ids": context["eligibility_spec"] if "snapshot_id" in context["eligibility_spec"] else
            {k: (None if v is None else list(v)) for k, v in context["eligible_ids"].items()},
            "filter_identity": context["filter_identity"], "experiment": experiment}, sort_keys=True, allow_nan=False))
        rank_config = {**config, "experiment": generating_config["experiment"] and {**experiment}}
        ranked = ranking.rank(inputs, context={**content_context, "limit": len(keys), "offset": 0}, config=rank_config, seed=seed, kinds=self.kinds)
        # the dominant category the source recorded per served item: the profile weights that
        # actually ranked this generation over the item's own tag seconds and categories
        weights = dict(ranked["admission_trace"]["invariants"]["profile"]["weights"])
        categories, vectors = {}, {}
        for feature in feature_rows:
            categories.update({int(t): str(c).lower() for t, c in feature["tag_categories"].items()})
            vectors[(feature["kind"], feature["id"])] = {int(t): float(v) for t, v in feature["tag_seconds"].items() if v > 0}
        items = []
        for position, row in enumerate(ranked["items"]):
            key = (row["kind"], row["id"])
            source = rows.get(key) or {}
            explanation = dict(row["explanation"])
            explanation["source_rank"] = explanation.get("position", position)
            explanation["position"] = position
            category = (ranking.dominant_category(vectors.get(key, {}), categories, weights, config["bodyparts_weight"])
                        if key[0] == self.primary else key[0])
            explanation["dominant_category"] = category
            items.append({"kind": key[0], "id": key[1], "score": row["score"], "explanation": explanation,
                          "title": source.get("title"), "media_url": source.get("media_url"),
                          "duration_s": float(source.get("duration_s") or 0.0), "category": category, "source_rank": position})
        # completed-generation equality: every token read before the build must read the same after it
        completed = self._committed_revisions(self.signals.read(), self._views(now))
        stable = (catalog_module.fingerprint_snapshot(self.catalog, pin["keys"], kinds=self.kinds) == pin
                  and self.tuner.stable(resolved) and committed["features"] is not None and views["status"] in ("ok", "disabled")
                  and completed == committed)
        provenance["revision_status"] = "stable" if stable else "unavailable_or_changed"
        provenance["ranking_generation_id"] = _digest([provenance, [(i["kind"], i["id"], i["score"], i["explanation"]) for i in items]])
        for item in items:
            item["provenance"] = copy.deepcopy(provenance)
        if stable:
            with self.lock:
                if len(self._cursors) >= 32:
                    oldest = min(self._cursors, key=lambda g: self._cursors[g]["created_at"])
                    del self._cursors[oldest]
                self._cursors[provenance["ranking_generation_id"]] = {
                    "items": copy.deepcopy(items), "catalog": pin, "context": cursor_context, "content": content_key, "created_at": now,
                    "revisions": committed,
                    "reset_generation": self.reset_generation, "generation_id": provenance["ranking_generation_id"]}
        page, total, has_more = ranking.page_items(items, offset=offset, limit=limit)
        response = serving.ranking_response(list(page), total, has_more, offset, kinds=self.kinds)
        response["source_counts"], response["fallback"] = ranked["source_counts"], ranked["fallback"]
        return response

    # ------------------------------------------------------------- discovery
    def search(self, query, mode="look", *, context=None, offset=0, limit=20):
        return discovery.search(self.sources, query, mode, context=context, offset=offset, limit=limit,
                                resolve_eligibility=self.resolve_eligibility)

    def similar(self, item_key, *, context=None, offset=0, limit=20, config=None):
        kind, item_id = item_key
        if kind != self.primary:
            raise ValueError("similar seeds are primary items")
        return discovery.similar(self.sources, context=context, seed_ids=[item_id], config=config, offset=offset, limit=limit,
                                 resolve_eligibility=self.resolve_eligibility)

    # -------------------------------------------------------------- recording
    def record(self, events: Sequence[Mapping[str, Any]]):
        receipts = []
        for event in events:
            kind = event.get("type")
            if kind == "served":
                receipts.append(ledger.record_served(self.ledger_path, request=event["request"], items=event["items"],
                                                     eligibility_snapshots=event.get("eligibility_snapshots"), kinds=self.kinds))
            elif kind == "viewed":
                receipts.append({"event_id": ledger.record_event(self.ledger_path, event=event["event"], kinds=self.kinds)})
            elif kind == "watch_capture":
                receipts.append(ledger.import_watch_capture(self.ledger_path, batch=event["batch"], kinds=self.kinds))
            elif kind == "feedback":
                receipts.append(self._feedback(event["operation"]))
            else:
                raise ValueError("unknown record type")
        return receipts

    def view(self, payload):
        return serving.view_request(payload, ledger_path=self.ledger_path, clock=self.clock, kinds=self.kinds)

    def _feedback(self, operation):
        if self.read_current is None:
            return {"status": "indeterminate", "error_code": "feedback_unsupported", "operation_id": operation.get("operation_id")}
        try:
            result = ledger.perform_feedback(self.ledger_path, operation=operation, read_current=self.read_current,
                                             apply_change=self.apply_change, kinds=self.kinds)
        except ledger.ContractError as exc:
            return {"status": "indeterminate", "error_code": str(exc), "operation_id": operation.get("operation_id")}
        if result.get("status") == "confirmed":
            self.reset_caches()
            try:
                self.advance_attribution()
            except Exception:
                pass
        return result

    def feedback(self, item_key, *, operation_id, session_id, rating=_OMITTED, engagement=False, request_id=None, viewed_event_id=None):
        """One ledger operation: a rating (explicit None clears) or one engagement increment."""
        if (rating is not _OMITTED) == bool(engagement):
            raise ValueError("feedback is one rating change or one engagement increment")
        operation = {"operation_id": operation_id, "kind": item_key[0], "item_id": item_key[1], "session_id": session_id,
                     "action": "engagement" if engagement else "rating"}
        if not engagement:
            operation["rating100"] = rating
        if request_id is not None:
            operation["request_id"] = request_id
        if viewed_event_id is not None:
            operation["viewed_event_id"] = viewed_event_id
        # the ledger receipt as recorded, plus the operation it validated, so undo() can consume it directly
        return {**self._feedback(operation), "operation": dict(operation)}

    def undo(self, receipt, *, operation_id):
        """Undo a feedback operation from its receipt under a new operation id. The ledger
        decides whether the original can be reversed; nothing here relabels its status."""
        original = receipt["operation"]
        operation = {"operation_id": operation_id, "kind": original["kind"], "item_id": original["item_id"],
                     "session_id": original["session_id"], "action": "undo", "undo_of": receipt["operation_id"]}
        for key in ("request_id", "viewed_event_id"):
            if original.get(key) is not None:
                operation[key] = original[key]
        return {**self._feedback(operation), "operation": operation}

    def reset_caches(self):
        with self.lock:
            self.reset_generation += 1
            self._cursors.clear()
        self.tuner.invalidate()

    # ------------------------------------------------------------ attribution
    def advance_attribution(self, now=None):
        """Advance the configured attribution policy to a monotone cutoff."""
        now = self.clock() if now is None else now
        with ledger._connection(self.ledger_path) as conn:
            last = conn.execute("SELECT max(through_ts) FROM rec_attribution_runs").fetchone()[0]
        if last is not None and now < last + self.attribution["min_advance_s"]:
            return 0
        return ledger.attribute_outcomes(self.ledger_path, through_ts=max(now, last or now),
                                         window_s=self.attribution["window_s"], policy_revision=self.attribution["policy_revision"])

    def tick(self, now=None):
        """Attribution first, then the tuner against its completed, ripened boundary."""
        now = self.clock() if now is None else now
        attributed = self.advance_attribution(now)
        return {"attributed": attributed, "tuner": self.tuner.tick(now=now)}

    def scorecard(self):
        return serving.build_scorecard(self.tuner, ledger_path=self.ledger_path, clock=self.clock)


__all__ = ["Engine", "initialize_stores", "DEFAULT_CONFIG", "DEFAULT_ATTRIBUTION", "RANKING_REVISION"]

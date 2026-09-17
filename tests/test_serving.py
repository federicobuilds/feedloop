"""The explicit serving boundary: envelope building, request/intent validation,
delivery recording from immutable provenance, skip-aware cursor continuation,
measured views, fatigue and the scorecard, over in-memory slots and local stores."""
import copy
import hashlib
import json
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from feedloop import engine as engine_module, ledger, serving, tuning
from feedloop.engine import Engine, initialize_stores
from fl2_helpers import MemoryCatalog, MemorySignals, MemorySpaces, catalog_row, md5_file, unit_rows

NOW = 2000000.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_catalog(videos=8, images=4):
    rows, features = [], {}
    for kind, count in (("video", videos), ("image", images)):
        for item_id in range(1, count + 1):
            tags = [10, 20 + item_id % 3]
            rows.append(catalog_row(kind, item_id, files=[{"fingerprints": [{"type": "md5", "value": f"{kind[0]}{item_id:031d}".replace("v", "a").replace("i", "b")}]}],
                                    duration=600.0 if kind == "video" else 0.0, tags=[f"tag {t}" for t in tags], contributors=["c1"]))
            features[(kind, item_id)] = {"tag_seconds": {10: 300.0, tags[1]: 100.0} if kind == "video" else {10: 1.0},
                                         "watched_tag_seconds": None, "tag_categories": {10: "acts", tags[1]: "other"}}
    return MemoryCatalog(rows, features, {10: "Tag ten", 20: "Tag twenty", 21: "Tag twentyone", 22: "Tag twentytwo"})


@pytest.fixture
def clock(monkeypatch):
    now = [NOW]
    fake = lambda: now[0]
    for module in (ledger, tuning, engine_module):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=fake, perf_counter=lambda: 0.0))
    return now, fake


def make_engine(tmp_path, fake, *, initialize=True, signals=None, config=None, **overrides):
    catalog = build_catalog()
    keys = sorted(catalog.rows)
    # nonnegative unit vectors: every secondary item has a positive visual score against the profile
    spaces = MemorySpaces({"visual": (keys, np.abs(unit_rows(len(keys), 8, 1)))})
    signals = signals or MemorySignals({("video", 1): {"rating": 90, "engagement_count": 0}}, observed_at=NOW - 100)
    ledger_path, tuner_path = str(tmp_path / "events.sqlite"), str(tmp_path / "tuner.sqlite")
    if initialize:
        initialize_stores(ledger_path=ledger_path, tuner_path=tuner_path, cutover_ts=NOW - 1000, clock=fake)
    eng = Engine(catalog=catalog, signals=signals, spaces=spaces, ledger_path=ledger_path, tuner_path=tuner_path, clock=fake,
                 read_current=signals.current, apply_change=signals.apply,
                 config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0, "images_share": 0.5, **(config or {})}, **overrides)
    return eng


def rows(path, sql):
    with sqlite3.connect(path) as conn:
        return conn.execute(sql).fetchall()


def requests(path):
    return ledger.read_evidence(path, since_ts=0, through_ts=NOW + 10 ** 6)["requests"]


PAYLOAD = {"request_id": "delivery-one", "client_request_id": "client-one", "session_id": "tab", "images": True,
           "surface": "home", "limit": 24, "offset": 0}


class Provenance:
    """A frozen, stable generation as the ranker would publish it."""

    def __init__(self):
        self.data = {"ranking_revision": "feed/v3", "ranking_generation_id": "0" * 63 + "1",
                     "config": {"include_images": True, "diversity": .2}, "seed": 17,
                     "revisions": {name: "original-" + name for name in ("features", "tag_projection", "catalog_fingerprints", "watch",
                                                                          "item_preferences", "secondary_preferences", "views")},
                     "captured_at": NOW - 10, "intent": {"seed_video_ids": [], "tag_ids": [], "exclude_video_ids": [], "exclude_image_ids": []},
                     "eligible_ids": {"video": None, "image": None}, "filter_identity": "intent:0", "experiment": None,
                     "revision_status": "stable"}
        self.rehash()

    def rehash(self):
        self.data["config_hash"] = digest(self.data["config"])

    def response(self, extra=None):
        items = [{"kind": kind, "id": 1, "score": .5, "duration_s": 100.0 if kind == "video" else 0.0, "title": f"item {kind} 1",
                  "provenance": copy.deepcopy(self.data),
                  "explanation": {"source_rank": index, "sources": ["synthetic"], "score": .5, **(extra or {}).get(kind, {})}}
                 for index, kind in enumerate(("video", "image"))]
        return {"items": items, "total": 2, "has_more": False, "next_offset": None, "next_cursor": None, "ranking": copy.deepcopy(self.data)}


@pytest.fixture
def cached(tmp_path, clock):
    """An engine whose ranker returns a frozen generation; the tuner must never be consulted for delivery."""
    now, fake = clock
    eng = make_engine(tmp_path, fake, automatic_tuning=False)
    provenance = Provenance()
    eng._rank = lambda cfg, **kw: provenance.response()
    eng.tuner.snapshot = Mock(side_effect=AssertionError("delivery relabeled cached tuner state"))
    return eng, provenance


# --------------------------------------------------------------- FeedContracts

class TestFeedContracts:
    def test_transport_failure_is_unavailable_without_raw_exception(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        eng._rank = Mock(side_effect=OSError("synthetic private diagnostic"))
        result = eng.feed(PAYLOAD)
        assert result["status"] == "unavailable"
        assert "private diagnostic" not in json.dumps(result)
        assert result["error_detail"] == "OSError"
        assert requests(eng.ledger_path) == []

    def test_internal_failure_codes_are_named_in_error_detail(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        eng._rank = Mock(side_effect=RuntimeError("catalog_enumeration_changed"))
        result = eng.feed(PAYLOAD)
        assert (result["status"], result["error_detail"]) == ("unavailable", "RuntimeError: catalog_enumeration_changed")
        eng._rank = Mock(side_effect=KeyError("taste_v2"))
        assert eng.feed(PAYLOAD)["error_detail"] == "KeyError"
        eng._rank = Mock(side_effect=RuntimeError("SELECT secret FROM table -- not a code"))
        assert eng.feed(PAYLOAD)["error_detail"] == "RuntimeError"

    def test_successful_empty_is_not_failure(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        eng._rank = lambda cfg, **kw: {"items": [], "total": 0, "has_more": False, "status": "empty"}
        result = eng.feed(PAYLOAD)
        assert result["items"] == [] and result["status"] == "empty"
        assert "error" not in result and "error_code" not in result

    def test_malformed_success_is_error_not_empty(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        eng._rank = lambda cfg, **kw: {}
        assert eng.feed(PAYLOAD)["status"] == "error"
        assert serving.build_feed({}, names={}, offset=0)["error_code"] == "ranking_response_invalid"
        assert serving.build_feed("nonsense", names={}, offset=0)["status"] == "error"

    def test_intent_is_forwarded_without_prefetch_impressions(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        seen = []
        def rank(cfg, **kw):
            seen.append((cfg, kw))
            return {"items": [], "status": "empty", "total": 0, "has_more": False}
        eng._rank = rank
        intent = {"revision": 3, "seed_video_id": 7, "tag_ids": [2, 2], "excluded_items": [{"kind": "image", "id": 7}]}
        cursor = {"generation_id": "a" * 64, "offset": 24, "after": "video:1"}
        result = eng.feed({**PAYLOAD, "intent": intent, "offset": 24, "cursor": cursor, "session_id": "session"})
        cfg, kw = seen[0]
        assert (kw["limit"], kw["offset"], kw["include_secondary"]) == (24, 24, True)
        wire = json.loads(cfg["intent"])
        assert wire["tag_ids"] == [2] and wire["seed_video_ids"] == [7]
        assert wire["exclude_image_ids"] == [7] and wire["exclude_video_ids"] == []
        assert cfg["filter_identity"] == "intent:3" and cfg["session_id"] == "session"
        assert json.loads(cfg["cursor"]) == cursor
        assert result["intent_revision"] is None, "missing generation metadata must not be relabeled with current intent"
        assert requests(eng.ledger_path) == [], "an empty page records no delivery"

    def test_invalid_intent_does_not_call_a_backend(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        eng._rank = Mock(side_effect=AssertionError("ranking called"))
        for intent in (False, 0, [], {"revision": -1}, {"seed_video_id": True}, {"tag_ids": ["2"]},
                       {"excluded_items": [{"kind": "unknown", "id": 3}]}, {"query": "private"}):
            assert eng.feed({**PAYLOAD, "intent": intent})["error_code"] == "invalid_intent", intent
        eng._rank.assert_not_called()

    def test_pure_ranking_never_receives_a_server_credential(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        rank = eng._rank = Mock(return_value={"items": [], "status": "empty", "total": 0, "has_more": False})
        eng.feed({**PAYLOAD, "images": False})
        text = json.dumps(rank.call_args.args[0]).lower()
        assert not any(word in text for word in ("api_key", "authorization", "credential", "token"))
        assert set(rank.call_args.args[0]) == {"intent", "eligibility", "filter_identity", "session_id", "client_request_id", "request_id", "cursor"}

    def test_partial_transport_preserves_identity_explanation_and_finite_seeks(self):
        response = {"schema_version": 1, "request_id": "r", "ranking": {"ranking_revision": "rank"}, "status": "partial",
                    "error_code": "voice_encoder_unavailable", "components": {"voice": {"status": "unavailable"}},
                    "has_more": True, "next_offset": 2, "next_cursor": {"generation_id": "a" * 64, "offset": 2, "after": "image:1"},
                    "items": [{"kind": "video", "id": 1, "duration_s": 100, "best_t": 50, "served_item_id": "s", "source_rank": 8,
                               "explanation": {"components": {"visual": .8}}, "rating100": 60},
                              {"kind": "image", "id": 1, "served_item_id": "i", "rating100": 20}]}
        result = serving.build_feed(copy.deepcopy(response), names={}, offset=0)
        assert result["items"][0]["served_item_id"] is None
        assert result["request_id"] is None, "observation cannot forward an upstream delivery identity"
        assert result["items"][0]["source_rank"] == 8
        assert result["items"][0]["explanation"] == response["items"][0]["explanation"]
        assert result["items"][0]["best_t"] == 50
        assert result["items"][1]["rating100"] == 20
        assert result["status"] == "partial"
        assert result["pagination"]["next_cursor"] == response["next_cursor"]
        for bad in (-1, 100, float("inf"), float("nan"), True, "50", None):
            assert serving.feed_time(bad, 100) is None
        assert serving.feed_time(1, 0) is None

    def test_secondary_lane_does_not_skip_primary_pagination_or_invent_scores(self):
        ranked = {"items": [{"kind": "video", "id": 1}, {"kind": "video", "id": 2},
                            {"kind": "image", "id": 1, "reason": "Actual image reason"}], "has_more": True, "next_offset": 13,
                  "next_cursor": {"generation_id": "a" * 64, "offset": 13, "after": "image:1"}}
        result = serving.build_feed(ranked, names={}, offset=10)
        assert result["pagination"]["next_offset"] == 13 and result["pagination"]["has_more"]
        assert result["items"][2]["reason"] == "Actual image reason"
        assert all(item["score"] is None and item["source_rank"] is None for item in result["items"])
        assert result["items"][2]["category"] == "image" and "tag_names" not in result["items"][2]

    def test_component_feature_failure_is_partial_not_empty(self):
        assert serving.build_feed({"items": [], "components": {"audio": {"status": "no-feature"}}}, names={}, offset=0)["status"] == "partial"

    def test_skip_aware_cursor_advances_by_original_positions_not_returned_count(self):
        ranked = {"items": [{"kind": "video", "id": 4}], "has_more": True, "next_offset": 4,
                  "next_cursor": {"generation_id": "a" * 64, "offset": 4, "after": "video:4"}}
        result = serving.build_feed(ranked, names={}, offset=1)
        assert result["status"] != "error", result
        assert result["pagination"]["next_offset"] == 4
        # the cursor logic itself: items 2 and 3 became ineligible, so one returned item advances three positions
        generation = "b" * 64
        frozen = [{"kind": "video", "id": i, "score": 1 - i / 10, "provenance": {"ranking_generation_id": generation}} for i in range(1, 7)]
        snapshot = {"items": frozen, "catalog": {"groups": {}}, "context": ("ctx",), "created_at": 0.0, "reset_generation": 0,
                    "generation_id": generation}
        current = {"present": {("video", i) for i in range(1, 7)}, "groups": {}, "allowed": {"video": None, "image": None},
                   "excluded": {("video", 2), ("video", 3)}}
        page = serving.continue_cursor(snapshot, {"generation_id": generation, "offset": 1, "after": "video:1"}, offset=1, limit=1,
                                       cursor_context=("ctx",), reset_generation=0, now=10.0, current=current)
        assert [i["id"] for i in page["items"]] == [4]
        assert page["next_offset"] == 4 and page["next_cursor"]["after"] == "video:4" and page["has_more"]
        # a stale, mismatched or expired snapshot is explicit
        for bad in ({"generation_id": "c" * 64, "offset": 1, "after": "video:1"}, {"generation_id": generation, "offset": 1, "after": "video:9"},
                    {"generation_id": generation, "offset": 9, "after": "video:6"}):
            with pytest.raises(ValueError, match="stale_ranking_cursor"):
                serving.continue_cursor(snapshot, bad, offset=bad["offset"], limit=1, cursor_context=("ctx",), reset_generation=0, now=10.0, current=current)
        with pytest.raises(ValueError, match="stale_ranking_cursor"):
            serving.continue_cursor(snapshot, {"generation_id": generation, "offset": 1, "after": "video:1"}, offset=1, limit=1,
                                    cursor_context=("ctx",), reset_generation=1, now=10.0, current=current)
        with pytest.raises(ValueError, match="stale_ranking_cursor"):
            serving.continue_cursor(snapshot, {"generation_id": generation, "offset": 1, "after": "video:1"}, offset=1, limit=1,
                                    cursor_context=("ctx",), reset_generation=0, now=serving.CURSOR_TTL_S + 1, current=current)


# ---------------------------------------------------- DeliveryAdapterContracts

class TestDeliveryAdapterContracts:
    def test_post_records_real_mixed_deliveries_and_cache_hits_preserve_generation(self, cached):
        eng, provenance = cached
        first = eng.feed(PAYLOAD)
        assert first["status"] == "ok", first
        assert first["request_id"] == PAYLOAD["request_id"]
        assert all(item["served_item_id"] for item in first["items"])
        first_events = {item["kind"]: item["served_item_id"] for item in first["items"]}
        retry = eng.feed(PAYLOAD)
        assert retry["status"] == "ok"
        assert first_events == {item["kind"]: item["served_item_id"] for item in retry["items"]}
        second = eng.feed({**PAYLOAD, "request_id": "delivery-two", "client_request_id": "client-two", "surface": "feed"})
        assert second["status"] == "ok"
        assert first["ranking_content_id"] == second["ranking_content_id"]
        assert all(item["served_item_id"] != first_events[item["kind"]] for item in second["items"])
        assert rows(eng.ledger_path, "SELECT event_type,count(*) FROM rec_events GROUP BY event_type") == [("served", 4)]
        assert rows(eng.ledger_path, "SELECT DISTINCT seed,config_hash,ranker_revision FROM rec_requests") == [(17, provenance.data["config_hash"], "feed/v3")]
        eng.tuner.snapshot.assert_not_called()

    def test_primary_items_name_every_known_contributor_and_never_invent_one(self, cached):
        eng, provenance = cached
        eng.catalog.names = {9: "Sunrise", 7: "Kitchen"}
        contributions = [{"tag_id": 9, "contribution": .02}, {"tag_id": 8, "contribution": -.01}, {"tag_id": 7, "contribution": .001}]
        eng._rank = lambda cfg, **kw: provenance.response({"video": {"tag_contributions": contributions}})
        result = eng.feed(PAYLOAD)
        video = next(item for item in result["items"] if item["kind"] == "video")
        assert video["tag_names"] == {"9": "Sunrise", "7": "Kitchen"}
        assert video["tags"] == ["sunrise", "kitchen"]
        image = next(item for item in result["items"] if item["kind"] == "image")
        assert "tag_names" not in image

    def test_new_served_secondary_item_can_only_be_viewed_after_qualification(self, cached):
        eng, _ = cached
        result = eng.feed(PAYLOAD)
        image = next(item for item in result["items"] if item["kind"] == "image")
        view = dict(client_event_id="qualified-image", session_id="tab", request_id=result["request_id"], served_item_id=image["served_item_id"],
                    surface="home", position=1, dwell_ms=1199, visible_fraction=.6, visibility_policy="foreground-60pct-1200ms-v1",
                    kind="image", item_id=1, occurred_at=NOW)
        assert eng.view(view)["status"] == "error"
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_events WHERE event_type='viewed'") == [(0,)]
        view["dwell_ms"] = 1200
        assert eng.view(view)["status"] == "confirmed"
        assert rows(eng.ledger_path, "SELECT kind,count(*) FROM rec_events WHERE event_type='viewed' GROUP BY kind") == [("image", 1)]
        fatigue = serving.build_fatigue(eng.ledger_path, now=NOW)
        assert fatigue["source"] == "qualified_views" and fatigue["unit"] == "distinct_utc_days"
        assert [(item["kind"], item["id"], item["days_shown"]) for item in fatigue["items"]] == [("image", 1, 1)]
        assert "multiplier" not in fatigue["items"][0]

    def test_cached_experiment_retains_original_arms_config_and_trace(self, cached):
        eng, provenance = cached
        provenance.data["config"]["embedding_weight"] = .35
        provenance.data["experiment"] = {"id": "original-experiment", "knob": "embedding_weight", "base": .35, "candidate": .4,
                                         "since_ts": provenance.data["captured_at"], "revision": 1}
        provenance.rehash()
        eng._rank = lambda cfg, **kw: provenance.response({"video": {"arm": "base"}})
        first = eng.feed(PAYLOAD)
        second = eng.feed({**PAYLOAD, "request_id": "new-delivery", "client_request_id": "new-client"})
        assert (first["status"], second["status"]) == ("ok", "ok")
        saved = rows(eng.ledger_path, "SELECT DISTINCT experiment_id,arms_json FROM rec_requests")
        assert len(saved) == 1 and saved[0][0] == "original-experiment"
        arms = json.loads(saved[0][1])
        assert (arms["base"]["embedding_weight"], arms["cand"]["embedding_weight"]) == (.35, .4)
        served = [json.loads(r[0]) for r in rows(eng.ledger_path, "SELECT payload_json FROM rec_events WHERE kind='video'")]
        assert {row["arm"] for row in served} == {"base"}
        assert all(row["trace"]["ranking_provenance"]["experiment"]["id"] == "original-experiment" for row in served)
        eng.tuner.snapshot.assert_not_called()

    def test_same_request_with_changed_pagination_conflicts_without_overwrite(self, cached):
        eng, provenance = cached
        eng.feed(PAYLOAD)
        before = rows(eng.ledger_path, "SELECT * FROM rec_requests")
        result = eng.feed({**PAYLOAD, "offset": 24, "cursor": {"generation_id": provenance.data["ranking_generation_id"], "offset": 24, "after": "image:1"}})
        assert result["error_code"] == "request_conflict"
        assert before == rows(eng.ledger_path, "SELECT * FROM rec_requests")

    def test_sensitive_generation_fields_never_enter_the_event_ledger(self, cached):
        eng, provenance = cached
        provenance.data["config"]["query"] = "synthetic private query"
        provenance.rehash()
        result = eng.feed(PAYLOAD)
        assert result["error_code"] == "sensitive_field"
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_requests") == [(0,)]
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_events") == [(0,)]

    def test_same_client_request_cannot_overwrite_changed_generation_or_context(self, cached):
        eng, provenance = cached
        assert eng.feed(PAYLOAD)["status"] == "ok"
        before = rows(eng.ledger_path, "SELECT * FROM rec_requests")
        provenance.data["config"]["diversity"] = .4
        provenance.rehash()
        result = eng.feed(PAYLOAD)
        assert (result["status"], result["items"], result["error_code"]) == ("error", [], "request_conflict")
        assert before == rows(eng.ledger_path, "SELECT * FROM rec_requests")
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_events") == [(2,)]

    def test_pure_observation_and_missing_or_unstable_provenance_never_record_delivery(self, cached):
        eng, provenance = cached
        observed = serving.build_feed(provenance.response(), names={}, offset=0)
        assert observed["status"] == "ok" and observed["request_id"] is None
        assert all(item["served_item_id"] is None for item in observed["items"])
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_requests") == [(0,)]
        provenance.data["revision_status"] = "unavailable_or_changed"
        result = eng.feed(PAYLOAD)
        assert result["status"] == "partial" and result["error_code"] == "ranking_provenance_unavailable"
        assert result["request_id"] is None
        assert result["items"] and all(item["served_item_id"] is None for item in result["items"])
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_requests") == [(0,)]

    def test_cached_context_mismatch_is_not_shown_as_current_intent(self, cached):
        eng, provenance = cached
        provenance.data["intent"]["tag_ids"] = [99]
        result = eng.feed(PAYLOAD)
        assert (result["status"], result["error_code"], result["items"]) == ("error", "ranking_context_conflict", [])
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_requests") == [(0,)]

    def test_saved_filter_provenance_must_match_before_delivery(self, cached):
        eng, provenance = cached
        payload = {**PAYLOAD, "filter_identity": "intent:0:saved:" + "a" * 64, "eligibility": {"video_ids": [1], "image_ids": [1]}}
        provenance.data["filter_identity"] = payload["filter_identity"]
        provenance.data["eligible_ids"] = {"video": [1], "image": [1]}
        assert eng.feed(payload)["status"] == "ok"
        payload.update(request_id="changed-filter", client_request_id="changed-filter", filter_identity="intent:0:saved:" + "b" * 64)
        result = eng.feed(payload)
        assert (result["status"], result["items"]) == ("error", [])
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_requests") == [(1,)]

    def test_missing_event_store_stays_uninitialized_and_exposures_unavailable(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1], initialize=False)
        provenance = Provenance()
        eng._rank = lambda cfg, **kw: provenance.response()
        result = eng.feed(PAYLOAD)
        assert (result["status"], result["error_code"]) == ("partial", "store_unavailable")
        assert not (tmp_path / "events.sqlite").exists()
        assert serving.build_fatigue(eng.ledger_path, now=NOW)["status"] == "unavailable"
        assert eng.view(dict(client_event_id="v", session_id="tab", request_id="r", served_item_id="s", surface="home", position=0,
                             dwell_ms=1200, visible_fraction=.6, visibility_policy="foreground-60pct-1200ms-v1", kind="video",
                             item_id=1, occurred_at=NOW))["status"] == "unavailable"
        assert not (tmp_path / "events.sqlite").exists() and not (tmp_path / "tuner.sqlite").exists()

    def test_cursor_records_current_constraints_without_relabeling_frozen_generation(self, cached):
        eng, provenance = cached
        payload = {**PAYLOAD, "offset": 3, "cursor": {"generation_id": provenance.data["ranking_generation_id"], "offset": 3, "after": "video:9"},
                   "eligibility": {"video_ids": [1], "image_ids": [1]}, "filter_identity": "narrowed",
                   "intent": {"revision": 1, "excluded_items": [{"kind": "video", "id": 7}]}}
        result = eng.feed(payload)
        assert result["status"] == "ok", result
        trace = json.loads(rows(eng.ledger_path, "SELECT payload_json FROM rec_events WHERE event_type='served' LIMIT 1")[0][0])["trace"]
        assert trace["ranking_provenance"] == provenance.data
        assert trace["delivery_context"]["eligible_ids"] == {"video": [1], "image": [1]}
        assert trace["delivery_context"]["filter_identity"] == "narrowed"
        assert trace["delivery_context"]["intent"]["exclude_video_ids"] == [7]

    def test_cursor_cannot_bypass_current_membership_or_seed_context(self, cached):
        eng, provenance = cached
        payload = {**PAYLOAD, "offset": 3, "cursor": {"generation_id": provenance.data["ranking_generation_id"], "offset": 3, "after": "video:9"}}
        for changes in ({"eligibility": {"video_ids": []}}, {"intent": {"excluded_items": [{"kind": "image", "id": 1}]}},
                        {"intent": {"seed_video_id": 7}}, {"intent": {"tag_ids": [7]}}):
            result = eng.feed({**payload, **changes})
            assert (result["status"], result["items"]) == ("error", []), changes
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_requests") == [(0,)]

    def test_concurrent_identical_post_retries_share_one_committed_delivery(self, cached, monkeypatch):
        eng, _ = cached
        barrier, lock, attempts = threading.Barrier(2), threading.Lock(), []
        original = ledger.record_served
        def record(*args, **kwargs):
            with lock:
                first_attempt = len(attempts) < 2
                attempts.append(1)
            if first_attempt:
                barrier.wait(3)
            return original(*args, **kwargs)
        monkeypatch.setattr(serving.ledger, "record_served", record)
        results = []
        workers = [threading.Thread(target=lambda: results.append(eng.feed(PAYLOAD))) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(5)
        assert all(not worker.is_alive() for worker in workers)
        assert len(results) == 2 and all(result["status"] == "ok" for result in results)
        assert [item["served_item_id"] for item in results[0]["items"]] == [item["served_item_id"] for item in results[1]["items"]]
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_requests") == [(1,)]
        assert rows(eng.ledger_path, "SELECT count(*) FROM rec_events") == [(2,)]


# -------------------------------------------------------- LiveServingContracts

LIVE = {"limit": 2, "images": True, "surface": "home", "offset": 0, "session_id": "tab-live", "request_id": "request-live-0",
        "client_request_id": "client-live-0", "intent": {"revision": 7, "tag_ids": [10]},
        "eligibility": {"video_ids": [1, 2, 3, 4], "image_ids": [1, 2]}}


class TestLiveServingContracts:
    def test_real_normalized_ranker_pages_commit_deliveries_without_loss_or_duplicates(self, tmp_path, clock, monkeypatch):
        now, fake = clock
        eng = make_engine(tmp_path, fake)
        ranks = Mock(wraps=engine_module.ranking.rank)
        monkeypatch.setattr(engine_module.ranking, "rank", ranks)
        body, pages = dict(LIVE), []
        for page_number in range(4):
            data = eng.feed(body)
            assert data["status"] == "ok", data
            assert all(item["request_id"] == body["request_id"] and item["served_item_id"] for item in data["items"])
            pages.append(data)
            if not data["pagination"]["has_more"]:
                break
            body.update(offset=data["pagination"]["next_offset"], cursor=data["pagination"]["next_cursor"],
                        request_id=f"request-live-{page_number + 1}", client_request_id=f"client-live-{page_number + 1}")
        assert len(pages) == 3
        keys = [(row["kind"], row["id"]) for page in pages for row in page["items"]]
        assert len(keys) == len(set(keys))
        assert set(keys) == {("video", i) for i in range(1, 5)} | {("image", 1), ("image", 2)}
        assert sorted(row["source_rank"] for page in pages for row in page["items"]) == list(range(6))
        assert [row["source_rank"] for page in pages for row in page["items"]] == list(range(6))
        ranks.assert_called_once(), "continuation pages reuse the frozen generation; only the first page ranks"
        evidence = ledger.read_evidence(eng.ledger_path, since_ts=0, through_ts=NOW + 10)
        assert len(evidence["requests"]) == 3
        generation = pages[0]["items"][0]["provenance"]
        for saved in evidence["requests"]:
            assert saved["config"] == generation["config"] and saved["config_hash"] == generation["config_hash"]
            assert saved["ranker_revision"] == generation["ranking_revision"] and saved["seed"] == generation["seed"]
            for field, names in (("feature_revision", ("features", "tag_projection", "catalog_fingerprints")),
                                 ("preference_revision", ("watch", "item_preferences", "secondary_preferences", "views"))):
                assert saved[field] == digest({name: generation["revisions"][name] for name in names})
        assert all(item["provenance"] == generation for page in pages for item in page["items"])
        fresh = eng.feed({**LIVE, "request_id": "fresh", "client_request_id": "fresh"})
        retry = eng.feed(LIVE)
        assert [i["served_item_id"] for i in retry["items"]] == [i["served_item_id"] for i in pages[0]["items"]]
        assert fresh["items"][0]["served_item_id"] != pages[0]["items"][0]["served_item_id"]
        assert fresh["ranking_content_id"] == pages[0]["ranking_content_id"], "a fresh delivery reuses the frozen content"
        ranks.assert_called_once()
        assert len(ledger.read_evidence(eng.ledger_path, since_ts=0, through_ts=NOW + 10)["requests"]) == 4

    def test_pure_observation_and_stale_cursor_do_not_create_delivery(self, tmp_path, clock):
        now, fake = clock
        eng = make_engine(tmp_path, fake)
        ranked = eng._rank({"intent": json.dumps({"tag_ids": [10]}), "eligibility": json.dumps({"video_ids": [1, 2, 3, 4], "image_ids": [1, 2]}),
                            "filter_identity": "intent:7", "session_id": "tab-live", "client_request_id": "c", "request_id": "r", "cursor": ""},
                           limit=2, offset=0, include_secondary=True)
        observation = serving.build_feed(ranked, names=eng.tag_names(), offset=0)
        assert observation["request_id"] is None and observation["items"]
        assert all(item["served_item_id"] is None for item in observation["items"])
        assert requests(eng.ledger_path) == []
        first = eng.feed(LIVE)
        assert first["status"] == "ok", first
        cursor = {**first["pagination"]["next_cursor"], "generation_id": "0" * 64}
        result = eng.feed({**LIVE, "offset": 2, "cursor": cursor, "request_id": "bad-cursor", "client_request_id": "bad-cursor"})
        assert (result["status"], result["error_code"], result["items"]) == ("error", "stale_ranking_cursor", [])
        assert len(requests(eng.ledger_path)) == 1
        # another session cannot continue this generation either
        result = eng.feed({**LIVE, "offset": 2, "cursor": first["pagination"]["next_cursor"], "session_id": "other-tab",
                           "request_id": "other", "client_request_id": "other"})
        assert (result["status"], result["error_code"]) == ("error", "stale_ranking_cursor")
        assert len(requests(eng.ledger_path)) == 1

    def test_snapshot_cursor_preserves_generation_and_binds_current_delivery_membership(self, tmp_path, clock):
        now, fake = clock
        eng = make_engine(tmp_path, fake)
        original = ledger.put_eligibility_snapshot(eng.ledger_path, saved_filter_id="7", mode="video", predicate_sha256="a" * 64,
                                                   eligible_ids={"video": [1, 2, 3, 4], "image": []}, observed_at=NOW)
        body = {**LIVE, "eligibility": {"snapshot_id": original["snapshot_id"]}, "filter_identity": "saved:original"}
        first = eng.feed(body)
        assert first["status"] == "ok", first
        assert [item["kind"] for item in first["items"]] == ["video", "video"] and first["pagination"]["has_more"]
        narrowed = ledger.put_eligibility_snapshot(eng.ledger_path, saved_filter_id="7", mode="video", predicate_sha256="b" * 64,
                                                   eligible_ids={"video": [4], "image": []}, observed_at=NOW)
        body.update(request_id="snapshot-continued", client_request_id="snapshot-continued", offset=first["pagination"]["next_offset"],
                    cursor=first["pagination"]["next_cursor"], eligibility={"snapshot_id": narrowed["snapshot_id"]}, filter_identity="saved:narrowed")
        second = eng.feed(body)
        assert second["status"] == "ok", second
        assert [(item["kind"], item["id"]) for item in second["items"]] == [("video", 4)]
        assert not second["pagination"]["has_more"] and second["pagination"]["next_offset"] is None
        with sqlite3.connect(eng.ledger_path) as conn:
            bindings = dict(conn.execute("SELECT role,snapshot_id FROM rec_request_eligibility WHERE request_id=?", (body["request_id"],)))
            trace = json.loads(conn.execute("SELECT payload_json FROM rec_events WHERE request_id=?", (body["request_id"],)).fetchone()[0])["trace"]
        assert bindings == {"generation": original["snapshot_id"], "delivery": narrowed["snapshot_id"]}
        assert trace["ranking_provenance"]["eligible_ids"] == {"snapshot_id": original["snapshot_id"]}
        assert trace["delivery_context"]["eligible_ids"] == {"snapshot_id": narrowed["snapshot_id"]}
        assert second["items"][0]["provenance"] == first["items"][0]["provenance"]

    def test_real_cursor_survives_visibility_and_skips_changed_constraints(self, tmp_path, clock):
        now, fake = clock
        eng = make_engine(tmp_path, fake)
        first = eng.feed(LIVE)
        assert first["status"] == "ok", first
        shown = first["items"][0]
        exposure = eng.view({"client_event_id": "visible-first", "session_id": LIVE["session_id"], "request_id": first["request_id"],
                             "served_item_id": shown["served_item_id"], "kind": shown["kind"], "item_id": shown["id"], "surface": "home",
                             "position": 0, "dwell_ms": 1200, "visible_fraction": .6, "visibility_policy": "foreground-60pct-1200ms-v1",
                             "occurred_at": NOW})
        assert exposure["status"] == "confirmed", exposure
        snapshot = eng._cursors[first["items"][0]["provenance"]["ranking_generation_id"]]
        offset = first["pagination"]["next_offset"]
        skipped = snapshot["items"][offset]
        intent = {**LIVE["intent"], "revision": 8, "excluded_items": [{"kind": skipped["kind"], "id": skipped["id"]}]}
        eligibility = copy.deepcopy(LIVE["eligibility"])
        eligibility[skipped["kind"] + "_ids"].remove(skipped["id"])
        body = {**LIVE, "request_id": "continued", "client_request_id": "continued", "intent": intent, "eligibility": eligibility,
                "offset": offset, "cursor": first["pagination"]["next_cursor"]}
        second = eng.feed(body)
        assert second["status"] == "ok", second
        assert second["pagination"]["next_offset"] == offset + len(second["items"]) + 1
        assert (skipped["kind"], skipped["id"]) not in [(item["kind"], item["id"]) for item in second["items"]]
        assert first["items"][0]["provenance"] == second["items"][0]["provenance"]
        with sqlite3.connect(eng.ledger_path) as conn:
            trace = json.loads(conn.execute("SELECT payload_json FROM rec_events WHERE request_id='continued' LIMIT 1").fetchone()[0])["trace"]
        assert trace["delivery_context"]["eligible_ids"][skipped["kind"]] == eligibility[skipped["kind"] + "_ids"]
        assert trace["delivery_context"]["intent"]["exclude_" + skipped["kind"] + "_ids"] == [skipped["id"]]

    def test_primary_and_secondary_rating_and_undo_use_supported_journal_fields(self, tmp_path, clock):
        now, fake = clock
        signals = MemorySignals({("video", 1): {"rating": 90, "engagement_count": 0}}, observed_at=NOW - 100)
        eng = make_engine(tmp_path, fake, signals=signals)
        data = eng.feed({**LIVE, "limit": 6})
        assert data["status"] == "ok", data
        for kind in ("video", "image"):
            item = next(item for item in data["items"] if item["kind"] == kind)
            key = (kind, item["id"])
            signals.rows[key] = {"rating": 60, "engagement_count": 2}
            rated = eng.feedback(key, operation_id="rating-" + kind, session_id=LIVE["session_id"], rating=80, request_id=data["request_id"])
            assert (rated["status"], signals.rows[key]["rating"]) == ("confirmed", 80)
            assert rated["operation_id"] == "rating-" + kind and rated["event_id"]
            result = eng.undo(rated, operation_id="undo-" + kind)
            assert (result["status"], signals.rows[key]["rating"], signals.rows[key]["engagement_count"]) == ("confirmed", 60, 2)
            assert result["event_id"]
            assert rows(eng.ledger_path, f"SELECT corrects_id FROM rec_events WHERE event_id='{result['event_id']}'") == [(rated["event_id"],)]


# ---------------------------------------------------- DashboardSharedContracts

class TestDashboardSharedContracts:
    def test_observations_do_not_create_missing_stores(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1], initialize=False)
        card = eng.scorecard()
        assert card["tuner"]["promotion_eligible"] is False
        assert card["capture"] == {"status": "unavailable", "verified": False, "reason": "store_unavailable"}
        assert card["evidence"]["status"] == "unavailable" and card["tuner"]["knob"] is None and card["tuner"]["arms"] == {}
        assert serving.build_fatigue(eng.ledger_path, now=NOW)["status"] == "unavailable"
        assert not (tmp_path / "events.sqlite").exists() and not (tmp_path / "tuner.sqlite").exists()

    def test_trial_validity_reasons_survive_the_scorecard_adapter(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        eng.tuner.read_evidence = Mock(return_value={"status": "ok", "valid": True, "promotion_enabled": True,
                                                     "validity_reasons": [], "promotion_reasons": []})
        summary = {"valid": False, "trials": [], "validity_reasons": ["cumulative_verdict_unavailable"],
                   "promotion_reasons": ["session_evidence_policy_required"]}
        eng.tuner.summarize_trials = Mock(return_value=summary)
        card = eng.scorecard()
        assert card["evidence"]["validity_reasons"] == ["cumulative_verdict_unavailable"]
        assert card["evidence"]["promotion_reasons"] == ["session_evidence_policy_required"]
        assert card["tuner"]["promotion_eligible"] is False
        summary.update(valid=True, validity_reasons=[], promotion_reasons=[])
        assert eng.scorecard()["tuner"]["promotion_eligible"] is False, "a green summary is not the report-only promotion flag"

    def test_scorecard_uses_actual_active_knob_and_missing_evidence_disables_promotion(self, tmp_path, clock):
        eng = make_engine(tmp_path, clock[1])
        verdict, reward = object(), object()
        eng.tuner.verdict, eng.tuner.reward = verdict, reward
        eng.tuner.read_evidence = Mock(return_value={"status": "unavailable", "validity_reasons": ["attribution_not_complete"]})
        eng.tuner.summarize_trials = Mock(return_value={"valid": False, "trials": []})
        eng.tuner.cumulative_facts = Mock(side_effect=AssertionError("scorecard cannot reconstruct history rewards"))
        with eng.tuner.transaction() as conn:
            conn.execute("UPDATE experiment SET active_knob='contributor_affinity_weight'")
        card = eng.scorecard()
        assert card["tuner"]["knob"] == "contributor_affinity_weight"
        assert (card["tuner"]["default"], card["tuner"]["step"]) == (.5, .1)
        assert card["tuner"]["promotion_eligible"] is False and card["tuner"]["arms"] == {}
        assert card["tuner"]["base"] is None, "the rotated-in knob has no arms until explicit initialization"
        assert eng.tuner.write("reset")["ok"]
        card = eng.scorecard()
        assert card["tuner"]["base"] == .5 and card["tuner"]["tuned_values"]["contributor_affinity_weight"] == .5
        eng.tuner.cumulative_facts.assert_not_called()
        assert eng.tuner.summarize_trials.call_args.kwargs["verdict"] is verdict
        assert eng.tuner.summarize_trials.call_args.kwargs["trial_reward"] is reward

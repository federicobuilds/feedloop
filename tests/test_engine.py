"""End-to-end: in-memory slots, local fixture stores, a fake clock, one view, one
watch outcome, attribution after a five-second demo window, and reversible feedback."""
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from feedloop import ledger, engine as engine_module, tuning
from feedloop.engine import Engine, initialize_stores
from fl2_helpers import MemoryCatalog, MemorySignals, MemorySpaces, catalog_row, unit_rows

TAGS = list(range(1, 11))


def build_catalog():
    rows, features = [], {}
    for kind, prefix in (("video", "v"), ("image", "i")):
        for item_id in range(1, 101):
            tags = [TAGS[item_id % 10], TAGS[(item_id + 3) % 10], TAGS[(item_id + 6) % 10]]
            files = [{"fingerprints": [{"type": "md5", "value": f"{prefix}{item_id:031d}".replace("v", "a").replace("i", "b")}]}]
            rows.append(catalog_row(kind, item_id, files=files, duration=600.0 if kind == "video" else 0.0,
                                    tags=[f"tag {t}" for t in tags], contributors=[f"c{item_id % 7}"]))
            seconds = {tags[0]: 300.0, tags[1]: 200.0, tags[2]: 100.0} if kind == "video" else {tags[0]: 1.0, tags[1]: 1.0}
            features[(kind, item_id)] = {"tag_seconds": seconds, "watched_tag_seconds": None,
                                         "tag_categories": {t: "acts" if t % 2 else "other" for t in tags}}
    return MemoryCatalog(rows, features, {t: f"tag {t}" for t in TAGS})


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    fake = lambda: now[0]
    for module in (ledger, tuning, engine_module):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=fake, perf_counter=lambda: 0.0))
    return now, fake


def test_in_memory_engine_feedback_loop(tmp_path, clock):
    now, fake = clock
    catalog = build_catalog()
    keys = sorted(catalog.rows)
    spaces = MemorySpaces({"visual": (keys, unit_rows(len(keys), 16, 1)), "semvisual": (keys, unit_rows(len(keys), 16, 2))})
    # real watch history on four liked videos, observed before the fixture events
    signals = MemorySignals({("video", i): {"rating": None, "engagement_count": 0,
                                            "watch": {"watched_s": 500.0, "last_at": 40.0 - i, "visit_days": [0], "intervals": [(0, 500)]}}
                             for i in (1, 11, 21, 31)}, observed_at=50.0)
    signals.rows[("image", 3)] = {"rating": 90, "engagement_count": 0}
    ledger_path, tuner_path = str(tmp_path / "events.sqlite"), str(tmp_path / "tuner.sqlite")
    now[0] = 90.0
    initialize_stores(ledger_path=ledger_path, tuner_path=tuner_path, cutover_ts=80.0, clock=fake)
    eng = Engine(catalog=catalog, signals=signals, spaces=spaces, ledger_path=ledger_path, tuner_path=tuner_path,
                 read_current=signals.current, apply_change=signals.apply, clock=fake,
                 config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0},
                 attribution={"window_s": 5, "policy_revision": "demo-explicit-w5-v1", "min_advance_s": 0})
    # 2-3. an explicit delivery with both kinds, a page smaller than the catalog
    now[0] = 100.0
    request = {"limit": 12, "images": True, "surface": "feed", "session_id": "session-1", "request_id": "req-1",
               "client_request_id": "client-1"}
    page = eng.feed(request)
    assert page["status"] == "ok", page
    assert 0 < len(page["items"]) <= 12
    assert page["pagination"]["has_more"] and page["pagination"]["next_cursor"]["offset"] == len(page["items"])
    assert len({(i["kind"], i["id"]) for i in page["items"]}) == len(page["items"])
    provenance = page["items"][0]["provenance"]
    assert provenance["revision_status"] == "stable" and provenance["experiment"] is not None
    assert all(i["provenance"] == provenance for i in page["items"])
    assert all(i["served_item_id"] and i["request_id"] == "req-1" for i in page["items"])
    experiment = provenance["experiment"]
    chosen = next(i for i in page["items"] if i["kind"] == "video" and i["explanation"].get("arm") in ("base", "cand"))
    arm = chosen["explanation"]["arm"]
    # 5. a qualified view at 101
    now[0] = 101.0
    viewed = eng.view({"client_event_id": "view-1", "session_id": "session-1", "request_id": "req-1",
                       "served_item_id": chosen["served_item_id"], "surface": "feed", "position": chosen["source_rank"],
                       "dwell_ms": 1200, "visible_fraction": 0.6, "visibility_policy": "foreground-60pct-1200ms-v1",
                       "kind": "video", "item_id": chosen["id"], "occurred_at": 101.0})
    assert viewed["status"] == "confirmed"
    # 6-7. a committed watch batch: start at 101, progress at 105 -> one four-second outcome
    now[0] = 105.0
    rows = [{"id": "w-start", "stream_session_id": "stream-1", "type": "view_start", "item_id": chosen["id"], "occurred_at": 101.0,
             "position": 0, "duration": 600.0, "session_id": "session-1", "viewed_event_id": viewed["event_id"],
             "previous_event_id": None, "playback_rate": 1, "canonical_session_id": "session-1"},
            {"id": "w-progress", "stream_session_id": "stream-1", "type": "view_progress", "item_id": chosen["id"], "occurred_at": 105.0,
             "position": 4, "duration": 600.0, "session_id": "session-1", "viewed_event_id": viewed["event_id"],
             "previous_event_id": "w-start", "playback_rate": 1, "canonical_session_id": "session-1"}]
    batch = {"capture_id": "capture-1", "source_id": "player", "received_at": 105.0, "source_revision": "source-rev-1",
             "status": "committed", "reason": None, "events_json": rows}
    first, again = eng.record([{"type": "watch_capture", "batch": batch}, {"type": "watch_capture", "batch": batch}])
    assert first == {"status": "imported", "outcomes": 1, "quarantined": 0}
    assert again == {"status": "duplicate", "outcomes": 0, "quarantined": 0}
    # 8. authoritative cumulative facts: 304 s of a 600 s item, no fresh rating or engagement
    signals.rows[("video", chosen["id"])] = {"rating": None, "engagement_count": 0,
                                             "watch": {"watched_s": 304.0, "last_at": 105.0, "visit_days": [0], "intervals": [(0, 304)]}}
    signals.observed_at = 105.0
    # 9-10. attribution credits only once the five-second window closes
    assert eng.tick(now=105.0)["attributed"] == 0
    now[0] = 106.0
    result = eng.tick(now=106.0)
    assert result["attributed"] == 1
    assert result["tuner"]["action"] == "wait"
    # 11. the scorecard at the completed cutoff with cumulative facts
    card = eng.scorecard()
    assert card["capture"]["verified"] is True
    assert card["capture"]["captured_outcomes"] == 1 and card["capture"]["attributed_outcomes"] == 1
    assert "cumulative_verdict_unavailable" not in card["evidence"]["validity_reasons"]
    arms = card["tuner"]["arms"]
    assert arms[arm]["trials"] == 1 and arms[arm]["successes"] == 1
    assert abs(arms[arm]["mean_reward"] - 4 / 3600) < 1e-9
    assert card["tuner"]["promotion_eligible"] is False
    # 12. reversible rating and engagement
    now[0] = 110.0
    key = ("video", chosen["id"])
    rated = eng.feedback(key, operation_id="op-rate", session_id="session-1", rating=90, viewed_event_id=viewed["event_id"])
    assert rated["status"] == "confirmed" and signals.current(key)["rating100"] == 90
    assert eng.feedback(key, operation_id="op-rate", session_id="session-1", rating=90, viewed_event_id=viewed["event_id"]) == rated
    assert rated["operation"] == {"operation_id": "op-rate", "kind": key[0], "item_id": key[1], "session_id": "session-1", "action": "rating",
                                  "rating100": 90, "viewed_event_id": viewed["event_id"]}
    undone = eng.undo(rated, operation_id="op-undo")
    assert undone["status"] == "confirmed" and signals.current(key)["rating100"] is None
    bumped = eng.feedback(key, operation_id="op-bump", session_id="session-1", engagement=True)
    assert bumped["status"] == "confirmed" and signals.current(key)["engagement_count"] == 1
    reverted = eng.undo(bumped, operation_id="op-unbump")
    assert reverted["status"] == "confirmed" and signals.current(key)["engagement_count"] == 0
    assert eng.undo(bumped, operation_id="op-unbump-2")["status"] == "conflict"
    # 13. an unknown generation in a continuation cursor is an explicit stale cursor, no delivery
    before = ledger.read_evidence(ledger_path, since_ts=0, through_ts=200)["requests"]
    cursor = dict(page["pagination"]["next_cursor"], generation_id="0" * 64)
    stale = eng.feed({**request, "offset": cursor["offset"], "cursor": cursor, "request_id": "req-2", "client_request_id": "client-2"})
    assert (stale["status"], stale["error_code"], stale["items"]) == ("error", "stale_ranking_cursor", [])
    assert ledger.read_evidence(ledger_path, since_ts=0, through_ts=200)["requests"] == before
    # 14. search without an encoder is an explicit no-feature result
    found = eng.search("blue skies", "look")
    assert found["status"] == "no-feature" and found["items"] == [] and found["components"]["look"]["status"] == "no-feature"
    # 15. similar returns eligible non-seed items sharing tags, with a positive tag explanation
    alike = eng.similar(("video", 5), limit=5)
    assert alike["status"] == "ok" and alike["items"]
    assert all(i["kind"] == "video" and i["id"] != 5 for i in alike["items"])
    assert all(i["similar"]["tag_similarity"] > 0 for i in alike["items"])


def make_engine(tmp_path, fake, *, config=None):
    catalog = build_catalog()
    keys = sorted(catalog.rows)
    spaces = MemorySpaces({"visual": (keys, np.abs(unit_rows(len(keys), 16, 1)))})
    signals = MemorySignals({("video", i): {"rating": None, "engagement_count": 0,
                                            "watch": {"watched_s": 500.0, "last_at": 40.0 - i, "visit_days": [0], "intervals": [(0, 500)]}}
                             for i in (1, 11, 21, 31)}, observed_at=50.0)
    ledger_path, tuner_path = str(tmp_path / "events.sqlite"), str(tmp_path / "tuner.sqlite")
    initialize_stores(ledger_path=ledger_path, tuner_path=tuner_path, cutover_ts=80.0, clock=fake)
    eng = Engine(catalog=catalog, signals=signals, spaces=spaces, ledger_path=ledger_path, tuner_path=tuner_path,
                 read_current=signals.current, apply_change=signals.apply, clock=fake, automatic_tuning=False,
                 config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0, **(config or {})})
    return eng, signals, spaces


REQUEST = {"limit": 24, "images": True, "surface": "feed", "session_id": "session-1", "request_id": "req-1", "client_request_id": "client-1"}


def test_served_categories_come_from_tag_categories_and_drive_the_entropy_veto(tmp_path, clock):
    now, fake = clock
    now[0] = 90.0
    eng, _signals, _spaces = make_engine(tmp_path, fake)
    now[0] = 100.0
    page = eng.feed(REQUEST)
    assert page["status"] == "ok", page
    import sqlite3
    with sqlite3.connect(eng.ledger_path) as conn:
        served = conn.execute("SELECT kind, json_extract(payload_json, '$.category') FROM rec_events WHERE event_type='served' ORDER BY rowid").fetchall()
    primary = [category for kind, category in served if kind == "video"]
    assert set(primary) == {"acts", "other"}, "a two-category corpus yields two distinct ledger categories"
    assert all(category == "image" for kind, category in served if kind == "image")
    # each recorded category is the source's dominant category: the generation's profile weights over the item's own tags
    for item in page["items"]:
        if item["kind"] != "video":
            continue
        features = eng.catalog.feature_rows[("video", item["id"])]
        expected = max(("acts", "other"), key=lambda c: sum(s for t, s in features["tag_seconds"].items() if features["tag_categories"][t] == c
                                                            and any(c2["tag_id"] == t and c2["weight"] > 0 for c2 in item["explanation"]["tag_contributions"])))
        assert item["category"] == item["explanation"]["dominant_category"]
        assert item["category"] in ("acts", "other")
        if any(c["weight"] > 0 for c in item["explanation"]["tag_contributions"]):
            assert item["category"] == expected
    # the tuner's veto reads exactly these categories: real mix -> promote, collapsed -> entropy veto
    rng = random.Random(3)
    def summary(cand_categories):
        trials, grouped = [], {}
        for s in range(20):
            for i in range(4):
                for arm, reward in (("base", .2 + rng.uniform(-.05, .05)), ("cand", .5 + rng.uniform(-.05, .05))):
                    pool = primary if arm == "base" else cand_categories
                    trial = {"arm": arm, "reward": reward, "category": pool[(s * 4 + i) % len(pool)], "viewed_id": f"{arm}-{s}-{i}",
                             "kind": "video", "item_id": s * 10 + i}
                    trials.append(trial)
                    grouped.setdefault(f"session-{s}", []).append(trial)
        return {"status": "ok", "valid": True, "validity_reasons": [], "trials": trials, "sessions": grouped}
    assert tuning.decide(summary(primary), now=2000000.0, window_started=1990000.0)[0] == "promote"
    action, detail = tuning.decide(summary(["other"]), now=2000000.0, window_started=1990000.0)
    assert (action, detail["reason"], detail["winner_blocked"]) == ("stall", "entropy_veto", "cand")


def test_changed_revisions_block_publication_and_first_page_reuse(tmp_path, clock, monkeypatch):
    now, fake = clock
    now[0] = 90.0
    eng, signals, spaces = make_engine(tmp_path, fake)
    now[0] = 100.0
    # a feature revision that moves while the vectors hydrate (spaces.matrix), or during the
    # ranking itself: unstable provenance, no cursor published, the emitted cursor is stale
    original_matrix, original_rank = spaces.matrix, engine_module.pipeline.prepare
    def moving_matrix(space):
        spaces.revisions["visual"] = 2
        return original_matrix(space)
    def moving_rank(*args, **kwargs):
        spaces.revisions["visual"] = 3
        return original_rank(*args, **kwargs)
    for moved, restore in ((lambda: setattr(spaces, "matrix", moving_matrix), lambda: setattr(spaces, "matrix", original_matrix)),
                           (lambda: monkeypatch.setattr(engine_module.pipeline, "prepare", moving_rank),
                            lambda: monkeypatch.setattr(engine_module.pipeline, "prepare", original_rank))):
        moved()
        unstable = eng.feed(REQUEST)
        restore()
        assert unstable["status"] == "partial" and unstable["error_code"] == "ranking_provenance_unavailable"
        assert unstable["items"] and unstable["items"][0]["provenance"]["revision_status"] == "unavailable_or_changed"
        assert eng._cursors == {}, "an unstable generation is never published for continuation"
        stale = eng.feed({**REQUEST, "request_id": "req-stale", "client_request_id": "client-stale", "offset": unstable["pagination"]["next_offset"],
                          "cursor": unstable["pagination"]["next_cursor"]})
        assert (stale["status"], stale["error_code"]) == ("error", "stale_ranking_cursor")
    # a stable build publishes; a signal change between two first pages yields a new generation
    ranks = Mock(wraps=engine_module.pipeline.prepare)
    monkeypatch.setattr(engine_module.pipeline, "prepare", ranks)
    first = eng.feed(REQUEST)
    assert first["status"] == "ok", first
    generation = first["items"][0]["provenance"]["ranking_generation_id"]
    assert generation in eng._cursors
    retry = eng.feed(REQUEST)
    assert retry["status"] == "ok" and retry["items"] == first["items"], "a retry of the same request reuses the frozen page"
    ranks.assert_called_once()
    fresh = eng.feed({**REQUEST, "request_id": "req-2", "client_request_id": "client-2"})
    assert fresh["items"][0]["provenance"]["ranking_generation_id"] != generation, "a fresh open builds a new generation"
    assert ranks.call_count == 2
    signals.rows[("video", 1)]["watch"]["watched_s"] = 2.0
    signals.observed_at = 60.0
    changed = eng.feed({**REQUEST, "request_id": "req-3", "client_request_id": "client-3"})
    assert changed["status"] == "ok", changed
    assert changed["items"][0]["provenance"]["ranking_generation_id"] != generation
    assert changed["items"][0]["provenance"]["revisions"]["watch"] != first["items"][0]["provenance"]["revisions"]["watch"]
    assert ranks.call_count == 3
    # the continuation cursor of the earlier generation still pages that generation, as the source did
    continued = eng.feed({**REQUEST, "request_id": "req-4", "client_request_id": "client-4", "offset": first["pagination"]["next_offset"],
                          "cursor": first["pagination"]["next_cursor"]})
    assert continued["status"] == "ok" and continued["items"][0]["provenance"]["ranking_generation_id"] == generation
    assert ranks.call_count == 3


def test_qualified_view_lowers_the_history_multiplier_by_the_impression_discount(tmp_path, clock):
    now, fake = clock
    now[0] = 90.0
    eng, signals, _spaces = make_engine(tmp_path, fake, config={"impression_discount": 0.5})
    now[0] = 100.0
    first = eng.feed(REQUEST)
    assert first["status"] == "ok", first
    shown = next(item for item in first["items"] if item["kind"] == "video")
    assert shown["explanation"]["history_multiplier"] == 1.0
    viewed = eng.view({"client_event_id": "view-1", "session_id": "session-1", "request_id": "req-1", "served_item_id": shown["served_item_id"],
                       "kind": "video", "item_id": shown["id"], "surface": "feed", "position": 0, "dwell_ms": 1500, "visible_fraction": .8,
                       "visibility_policy": "foreground-60pct-1200ms-v1", "occurred_at": 100.0})
    assert viewed["status"] == "confirmed", viewed
    now[0] = 101.0
    second = eng.feed({**REQUEST, "request_id": "req-2", "client_request_id": "client-2", "session_id": "session-2"})
    assert second["status"] == "ok", second
    assert second["items"][0]["provenance"]["revisions"]["views"] != first["items"][0]["provenance"]["revisions"]["views"]
    before = {(item["kind"], item["id"]): item["explanation"].get("history_multiplier")
              for item in eng._cursors[first["items"][0]["provenance"]["ranking_generation_id"]]["items"]}
    after = {(item["kind"], item["id"]): item["explanation"].get("history_multiplier")
             for item in eng._cursors[second["items"][0]["provenance"]["ranking_generation_id"]]["items"]}
    assert after[("video", shown["id"])] == pytest.approx(0.5), "one distinct qualified view day: discount ** 1"
    # the rest of the first page was delivered without a view: one skip each; every other unwatched item keeps
    # its multiplier (watched items carry a clock-dependent cooldown, not fatigue)
    skipped = {(item["kind"], item["id"]) for item in first["items"] if item["kind"] == "video"} - {("video", shown["id"])}
    others = {key: value for key, value in after.items() if key != ("video", shown["id"]) and value is not None and key not in signals.rows}
    assert skipped and all(others[key] == pytest.approx(before[key] * 0.5) for key in skipped & set(others))
    unchanged = {key: value for key, value in others.items() if key not in skipped}
    assert unchanged and all(before[key] == value for key, value in unchanged.items())


def test_feed_without_recording_delivery_writes_nothing(tmp_path, clock, monkeypatch):
    import os
    import sqlite3
    now, fake = clock
    now[0] = 90.0
    eng, _signals, _spaces = make_engine(tmp_path, fake)
    now[0] = 100.0

    def counts(path):
        with sqlite3.connect("file:" + path + "?mode=ro", uri=True) as conn:
            tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            return {table: conn.execute('SELECT COUNT(*) FROM "' + table + '"').fetchone()[0] for table in tables}

    stores = (eng.ledger_path, eng.tuner_path)
    before = [(os.stat(path).st_mtime_ns, counts(path)) for path in stores]
    with monkeypatch.context() as patch:
        patch.setattr(engine_module.serving, "serve_feed", lambda *a, **k: pytest.fail("serve_feed reached"))
        patch.setattr(eng.tuner, "maybe_tick", lambda: pytest.fail("tuner tick reached"))
        preview = eng.feed(REQUEST, record_delivery=False)
    assert [(os.stat(path).st_mtime_ns, counts(path)) for path in stores] == before
    assert preview["status"] == "ok" and preview["delivery_recorded"] is False and preview["request_id"] == "req-1"
    assert preview["items"] and all(item["served_item_id"] is None for item in preview["items"])
    delivered = eng.feed(REQUEST)
    assert delivered["status"] == "ok" and [(i["kind"], i["id"]) for i in preview["items"]] == [(i["kind"], i["id"]) for i in delivered["items"]]
    assert counts(eng.ledger_path) != before[0][1]


# ---------------------------------------------------------------- WP2b envelope and performance (2026-09-29)
def _paging_engine(tmp_path, fake, now):
    catalog = build_catalog()
    keys = sorted(catalog.rows)
    spaces = MemorySpaces({"visual": (keys, unit_rows(len(keys), 16, 1)), "semvisual": (keys, unit_rows(len(keys), 16, 2))})
    signals = MemorySignals({("video", i): {"rating": None, "engagement_count": 0,
                                            "watch": {"watched_s": 500.0, "last_at": 40.0 - i, "visit_days": [0], "intervals": [(0, 500)]}}
                             for i in (1, 11, 21, 31)}, observed_at=50.0)
    ledger_path, tuner_path = str(tmp_path / "events.sqlite"), str(tmp_path / "tuner.sqlite")
    now[0] = 90.0
    initialize_stores(ledger_path=ledger_path, tuner_path=tuner_path, cutover_ts=80.0, clock=fake)
    eng = Engine(catalog=catalog, signals=signals, spaces=spaces, ledger_path=ledger_path, tuner_path=tuner_path, clock=fake,
                 config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0})
    now[0] = 100.0
    return eng, signals, spaces



def test_a_later_signal_on_one_item_leaves_other_items_facts_at_the_cutoff(tmp_path, clock):
    now, fake = clock
    catalog = build_catalog()
    keys = sorted(catalog.rows)
    spaces = MemorySpaces({"visual": (keys, unit_rows(len(keys), 16, 1)), "semvisual": (keys, unit_rows(len(keys), 16, 2))})
    signals = MemorySignals({("video", i): {"rating": None, "engagement_count": 0, "updated_at": 40.0 - i,
                                            "watch": {"watched_s": 500.0, "last_at": 40.0 - i, "visit_days": [0], "intervals": [(0, 500)]}}
                             for i in (1, 11, 21, 31)}, observed_at=50.0)
    ledger_path, tuner_path = str(tmp_path / "events.sqlite"), str(tmp_path / "tuner.sqlite")
    now[0] = 90.0
    initialize_stores(ledger_path=ledger_path, tuner_path=tuner_path, cutover_ts=80.0, clock=fake)
    eng = Engine(catalog=catalog, signals=signals, spaces=spaces, ledger_path=ledger_path, tuner_path=tuner_path, clock=fake,
                 config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0},
                 attribution={"window_s": 5, "policy_revision": "demo-explicit-w5-v1", "min_advance_s": 0})
    now[0] = 100.0
    page = eng.feed({**PAGE, "limit": 12})
    chosen = next(i for i in page["items"] if i["explanation"].get("arm") in ("base", "cand"))
    arm = chosen["explanation"]["arm"]
    now[0] = 101.0
    viewed = eng.view({"client_event_id": "view-1", "session_id": "s1", "request_id": "r1", "served_item_id": chosen["served_item_id"],
                       "surface": "feed", "position": chosen["source_rank"], "dwell_ms": 1200, "visible_fraction": 0.6,
                       "visibility_policy": "foreground-60pct-1200ms-v1", "kind": "video", "item_id": chosen["id"], "occurred_at": 101.0})
    now[0] = 105.0
    common = {"stream_session_id": "st", "item_id": chosen["id"], "duration": 600.0, "session_id": "s1", "viewed_event_id": viewed["event_id"],
              "playback_rate": 1, "canonical_session_id": "s1"}
    rows = [{**common, "id": "w-start", "type": "view_start", "occurred_at": 101.0, "position": 0, "previous_event_id": None},
            {**common, "id": "w-progress", "type": "view_progress", "occurred_at": 105.0, "position": 4, "previous_event_id": "w-start"}]
    eng.record([{"type": "watch_capture", "batch": {"capture_id": "cap-1", "source_id": "player", "received_at": 105.0,
                                                    "source_revision": "rev-1", "status": "committed", "reason": None, "events_json": rows}}])
    a, b = ("video", chosen["id"]), next(k for k in keys if k[0] == "video" and k[1] != chosen["id"])
    signals.rows[a] = {"rating": None, "engagement_count": 0, "updated_at": 105.0,
                       "watch": {"watched_s": 304.0, "last_at": 105.0, "visit_days": [0], "intervals": [(0, 304)]}}
    # item B is rated long after the evidence cutoff; only B's own facts become unavailable
    signals.rows[b] = {"rating": 90, "engagement_count": 0, "updated_at": 1000.0}
    signals.observed_at = 1000.0
    facts = eng.cumulative_facts({a, b}, 106.0)["items"]
    assert facts[a]["watched_s"] == 304.0 and b not in facts
    now[0] = 106.0
    assert eng.tick(now=106.0)["attributed"] == 1
    card = eng.scorecard()
    assert "cumulative_verdict_unavailable" not in card["evidence"]["validity_reasons"]
    assert card["tuner"]["arms"][arm]["trials"] == 1 and card["tuner"]["arms"][arm]["successes"] == 1

PAGE = {"limit": 5, "images": False, "surface": "feed", "session_id": "s1", "request_id": "r1", "client_request_id": "c1"}


def test_cumulative_facts_observed_after_the_cutoff_are_unavailable(tmp_path, clock):
    now, fake = clock
    eng, signals, _spaces = _paging_engine(tmp_path, fake, now)
    assert eng.cumulative_facts({("video", 1)}, 60.0)["items"][("video", 1)]["watched_s"] == 500.0
    # a rating observed at 70 is current data, not the state at the older cutoff 60
    signals.rows[("video", 1)]["rating"] = 90
    signals.observed_at = 70.0
    assert eng.cumulative_facts({("video", 1)}, 60.0) == {"cutoff_ts": 60.0, "items": {}}
    assert eng.cumulative_facts({("video", 1)}, 70.0)["items"][("video", 1)]["rating"] == 90


def test_continuation_serves_frozen_page_best_t_and_profile_without_hydrating_means(tmp_path, clock, monkeypatch):
    now, fake = clock
    eng, _signals, _spaces = _paging_engine(tmp_path, fake, now)
    prepare = engine_module.pipeline.prepare
    monkeypatch.setattr(engine_module.pipeline, "prepare", lambda **kw: {**prepare(**kw), "windows": lambda keys: {k: 7.0 for k in keys}})
    first = eng.feed(PAGE, record_delivery=False)
    assert first["status"] == "ok" and first["pagination"]["has_more"], first
    assert all(item["best_t"] == 7.0 for item in first["items"])
    assert first["profile"] and "weights" not in first["profile"]
    cursor = first["pagination"]["next_cursor"]
    frozen = eng._cursors[cursor["generation_id"]]["items"]
    means = Mock(wraps=eng.sources.means)
    monkeypatch.setattr(eng.sources, "means", means)
    now[0] = 101.0
    second = eng.feed({**PAGE, "request_id": "r2", "client_request_id": "c2", "offset": cursor["offset"], "cursor": cursor},
                      record_delivery=False)
    assert second["status"] == "ok", second
    assert means.call_count == 0, "a continuation reuses the frozen page and hydrates no vectors"
    assert [(i["kind"], i["id"]) for i in second["items"]] == [(i["kind"], i["id"]) for i in frozen[cursor["offset"]:cursor["offset"] + 5]]
    assert all(item["best_t"] == 7.0 for item in second["items"]) and second["profile"] == first["profile"]


def test_continuation_refuses_changed_feature_revisions(tmp_path, clock):
    now, fake = clock
    eng, _signals, spaces = _paging_engine(tmp_path, fake, now)
    first = eng.feed(PAGE, record_delivery=False)
    cursor = first["pagination"]["next_cursor"]
    spaces.revisions["visual"] = 2
    now[0] = 101.0
    page = eng.feed({**PAGE, "request_id": "r2", "client_request_id": "c2", "offset": cursor["offset"], "cursor": cursor},
                    record_delivery=False)
    assert page == {"items": [], "status": "error", "error_code": "stale_ranking_cursor"}


def test_qualified_view_query_searches_the_view_time_range(tmp_path):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    for sql in ledger._SCHEMA:
        conn.execute(sql)
    plan = " ".join(row[-1] for row in conn.execute("EXPLAIN QUERY PLAN " + ledger._QUALIFIED_VIEWS_SQL, (0.0, 1.0, 1.0)))
    assert "SEARCH rec_events USING INDEX rec_view_time (occurred_at>? AND occurred_at<?)" in plan, plan
    assert "SCAN rec_events" not in plan, plan
    db = str(tmp_path / "old.db")
    ledger.initialize_event_store(db, cutover_ts=0.0)
    with sqlite3.connect(db) as old:
        old.execute("DROP INDEX rec_view_time")
    with ledger._connection(db, write=True):
        pass
    with sqlite3.connect(db) as reopened:
        assert reopened.execute("SELECT 1 FROM sqlite_master WHERE name='rec_view_time'").fetchone()


def test_watch_receipt_retry_with_a_fresh_received_at_is_a_duplicate(tmp_path, clock):
    now, fake = clock
    eng, _signals, _spaces = _paging_engine(tmp_path, fake, now)
    page = eng.feed(PAGE)
    item = page["items"][0]
    now[0] = 101.0
    viewed = eng.view({"client_event_id": "view-1", "session_id": "s1", "request_id": "r1", "served_item_id": item["served_item_id"],
                       "surface": "feed", "position": item["source_rank"], "dwell_ms": 1200, "visible_fraction": 0.6,
                       "visibility_policy": "foreground-60pct-1200ms-v1", "kind": "video", "item_id": item["id"], "occurred_at": 101.0})
    row = {"id": "w-start", "stream_session_id": "st", "type": "view_start", "item_id": item["id"], "occurred_at": 101.0, "position": 0,
           "duration": 600.0, "session_id": "s1", "viewed_event_id": viewed["event_id"], "previous_event_id": None,
           "playback_rate": 1, "canonical_session_id": "s1"}
    batch = {"capture_id": "cap-1", "source_id": "player", "received_at": 102.0, "source_revision": "rev", "status": "committed",
             "reason": None, "events_json": [row]}
    assert ledger.watch_capture_status(eng.ledger_path, source_id="player", capture_id="cap-1") == {"status": "not_found"}
    now[0] = 103.0
    assert eng.record([{"type": "watch_capture", "batch": batch}])[0]["status"] == "imported"
    now[0] = 110.0
    assert eng.record([{"type": "watch_capture", "batch": {**batch, "received_at": 110.0}}])[0]["status"] == "duplicate"
    with pytest.raises(ledger.ContractError, match="watch_receipt_conflict"):
        eng.record([{"type": "watch_capture", "batch": {**batch, "received_at": 110.0, "source_revision": "other"}}])
    assert ledger.watch_capture_status(eng.ledger_path, source_id="player", capture_id="cap-1")["status"] == "imported"


def _moment_windows(query):
    """Engine._read_windows over a fake paired look space: item 1 has four windows within the
    margin of its best and one outside it, item 2 one window, item 3 a runner-up past the margin."""
    rows = [(1, 0.0, 1.0), (1, 10.0, 0.995), (1, 20.0, 0.985), (1, 30.0, 0.99), (1, 40.0, 0.9),
            (2, 5.0, 0.8), (3, 1.0, 1.0), (3, 2.0, 0.95)]
    keys = [("video", sid) for sid, _t, _s in rows]
    times = np.array([t for _sid, t, _s in rows], dtype=np.float32)
    matrix = np.array([[s, np.sqrt(1.0 - s * s)] for _sid, _t, s in rows], dtype=np.float32)
    fake = SimpleNamespace(primary="video", roles={"visual": "v", "semantic": "s"},
                           spaces=SimpleNamespace(windows=lambda space: (keys, times, matrix)))
    return Engine._read_windows(fake, query, [1, 2, 3])


QUERY = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32) / np.sqrt(2.0)


def test_moment_candidates_are_the_top_three_windows_within_the_margin_best_first():
    assert _moment_windows(QUERY) == {1: (0.0, 10.0, 30.0), 2: (5.0,), 3: (1.0,)}


def test_moment_same_seed_same_time():
    moments = _moment_windows(QUERY)[1]
    assert {engine_module.choose_moment(moments, seed=1234, sid=1) for _ in range(20)} == {engine_module.choose_moment(moments, seed=1234, sid=1)}


def test_moment_different_seeds_vary_among_the_candidates():
    moments = _moment_windows(QUERY)[1]
    drawn = {engine_module.choose_moment(moments, seed=seed, sid=1) for seed in range(50)}
    assert len(drawn) > 1 and drawn <= set(moments)


def test_moment_single_qualifying_window_is_unchanged():
    moments = _moment_windows(QUERY)
    for seed in range(20):
        assert engine_module.choose_moment(moments[2], seed=seed, sid=2) == 5.0
        assert engine_module.choose_moment(moments[3], seed=seed, sid=3) == 1.0


def test_moment_selection_is_deterministic_across_two_runs():
    def run():
        read = _moment_windows(QUERY)
        return [{sid: engine_module.choose_moment(times, seed=seed, sid=sid) for sid, times in read.items()} for seed in (0, 7, 2**62)]
    assert run() == run()


def test_check_signals_reports_contract_breaks():
    from feedloop.slots import check_signals
    signals = MemorySignals({("video", 1): {"rating": 80, "engagement_count": 1, "updated_at": 90.0},
                             ("video", 2): {"rating": 60, "engagement_count": 0},
                             ("video", 3): {"rating": float("nan"), "engagement_count": 0, "updated_at": 120.0}},
                            observed_at=100.0)
    assert check_signals(signals, [("video", 1)]) == []
    assert check_signals(signals) == ["video:2: missing updated_at", "video:3: updated_at is after observed_at",
                                      "video:3: rating is not finite"]

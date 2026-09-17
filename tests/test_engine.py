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
    original_matrix, original_rank = spaces.matrix, engine_module.ranking.rank
    def moving_matrix(space):
        spaces.revisions["visual"] = 2
        return original_matrix(space)
    def moving_rank(*args, **kwargs):
        spaces.revisions["visual"] = 3
        return original_rank(*args, **kwargs)
    for moved, restore in ((lambda: setattr(spaces, "matrix", moving_matrix), lambda: setattr(spaces, "matrix", original_matrix)),
                           (lambda: monkeypatch.setattr(engine_module.ranking, "rank", moving_rank),
                            lambda: monkeypatch.setattr(engine_module.ranking, "rank", original_rank))):
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
    ranks = Mock(wraps=engine_module.ranking.rank)
    monkeypatch.setattr(engine_module.ranking, "rank", ranks)
    first = eng.feed(REQUEST)
    assert first["status"] == "ok", first
    generation = first["items"][0]["provenance"]["ranking_generation_id"]
    assert generation in eng._cursors
    same = eng.feed({**REQUEST, "request_id": "req-2", "client_request_id": "client-2"})
    assert same["items"][0]["provenance"]["ranking_generation_id"] == generation
    ranks.assert_called_once()
    signals.rows[("video", 1)]["watch"]["watched_s"] = 2.0
    signals.observed_at = 60.0
    changed = eng.feed({**REQUEST, "request_id": "req-3", "client_request_id": "client-3"})
    assert changed["status"] == "ok", changed
    assert changed["items"][0]["provenance"]["ranking_generation_id"] != generation
    assert changed["items"][0]["provenance"]["revisions"]["watch"] != first["items"][0]["provenance"]["revisions"]["watch"]
    assert ranks.call_count == 2
    # the continuation cursor of the earlier generation still pages that generation, as the source did
    continued = eng.feed({**REQUEST, "request_id": "req-4", "client_request_id": "client-4", "offset": first["pagination"]["next_offset"],
                          "cursor": first["pagination"]["next_cursor"]})
    assert continued["status"] == "ok" and continued["items"][0]["provenance"]["ranking_generation_id"] == generation
    assert ranks.call_count == 2


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
    # every other unwatched item keeps its multiplier (watched items carry a clock-dependent cooldown, not fatigue)
    unchanged = {key: value for key, value in after.items() if key != ("video", shown["id"]) and value is not None and key not in signals.rows}
    assert unchanged and all(before[key] == value for key, value in unchanged.items())

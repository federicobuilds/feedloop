"""Offline ledger contracts. Only the import-pure ledger module is loaded, from its installed file."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import feedloop.ledger
from feedloop.profiles import trial_reward
from feedloop.taste import verdict


MODULE = Path(feedloop.ledger.__file__)


def load_events():
    spec = importlib.util.spec_from_file_location("isolated_ledger", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def request(client="page-1", session="session-1", experiment="experiment-1"):
    return dict(request_id="request-" + client, client_request_id=client,
                schema_version=1, created_at=100.0, session_id=session,
                surface="Feed", recommender="feed", context_id="opaque-1",
                config={"finished_ratio": .45, "abandon_ratio": .15,
                        "rating_strength": 1.0}, ranker_revision="code-1",
                feature_revision="features-1", preference_revision="preferences-1",
                preference_cutoff=90.0, intent_revision="intent-1",
                eligibility_revision="eligibility-1", seed=7,
                experiment_id=experiment, ranking_content_id="ranking-" + client,
                arms={"base": {"embedding_weight": .35},
                      "cand": {"embedding_weight": .4}} if experiment else {})


def item(item_id=1, kind="video", arm="base"):
    return dict(kind=kind, id=item_id, source_rank=0, score=.75, arm=arm,
                control=False, explore=False, category="other",
                duration_s=600.0 if kind == "video" else 0.0, trace={"sources": ["visual"]})


def view(parent, key="view-1", ts=110.0, session="session-1", item_id=1, kind="video"):
    return dict(event_type="viewed", source="browser", source_event_id=key,
                session_id=session, kind=kind, item_id=item_id, occurred_at=ts,
                parent_id=parent,
                payload={"visible_fraction": .6, "dwell_ms": 1200,
                         "foreground": True, "display_rank": 0, "placement": "Feed"})


def outcome(key="watch-1", ts=150.0, parent=None, item_id=1, kind="video",
            session="session-1", payload=None, corrects=None):
    return dict(event_type="outcome", source="bridge", source_event_id=key,
                session_id=session, kind=kind, item_id=item_id, occurred_at=ts,
                parent_id=parent, corrects_id=corrects,
                payload=payload or {"signal": "watch", "watched_s_delta": 30.0,
                                   "started_at": ts - 30, "provenance": "confirmed_delta_v1"})


def concurrent_record(db, parent, ready, release, results):
    events = load_events()
    ready.put(True)
    release.wait(5)
    try:
        results.put(events.record_event(db, event=view(parent)))
    except Exception as exc:
        results.put(type(exc).__name__)


class EventContracts(unittest.TestCase):
    def setUp(self):
        self.events = load_events()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = str(Path(self.temp.name) / "events.sqlite")
        self.events.initialize_event_store(self.db, cutover_ts=95.0)
        self.clock = patch.object(self.events.time, "time", return_value=200.0)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.events.record_session_mapping(self.db, alias_session_id="session-1",
            canonical_session_id="session-1", mapping_revision="initial")

    def serve(self, client="page-1", arm="base", **kwargs):
        req = request(client, kwargs.pop("session", "session-1"), kwargs.pop("experiment", "experiment-1"))
        return self.events.record_served(self.db, request=req, items=[item(arm=arm, **kwargs)])

    def record_view(self, parent=None, **kwargs):
        if parent is None:
            parent = self.serve()[("video", 1)]
        return self.events.record_event(self.db, event=view(parent, **kwargs))

    def evidence(self):
        return self.events.read_evidence(self.db, since_ts=95, through_ts=500)

    def attribute(self):
        return self.events.attribute_outcomes(self.db, through_ts=500, window_s=100,
                                             policy_revision="policy-1")

    def test_initialization_idempotent_and_cutover_immutable(self):
        self.events.initialize_event_store(self.db, cutover_ts=95)
        with self.assertRaises(self.events.ContractError):
            self.events.initialize_event_store(self.db, cutover_ts=96)

    def test_cache_retry_identity_and_conflicting_payload(self):
        first = self.serve()
        self.assertEqual(first, self.serve())
        self.assertEqual(len(self.evidence()["events"]), 1)
        with self.assertRaises(self.events.ContractError):
            self.serve(arm="cand")
        other = request()
        other["request_id"] = "different-request"
        with self.assertRaises(self.events.ContractError):
            self.events.record_served(self.db, request=other, items=[item()])

    def test_serves_are_not_views_and_same_id_other_kind_is_distinct(self):
        refs = self.events.record_served(self.db, request=request(), items=[item(), item(kind="image", arm="")])
        self.assertEqual(self.events.read_view_counts(self.db, since_ts=95, through_ts=500)["counts"], {})
        self.record_view(refs[("image", 1)], kind="image")
        counts = self.events.read_view_counts(self.db, since_ts=95, through_ts=500)["counts"]
        self.assertEqual(counts, {("image", 1): 1})

    def test_view_duplicate_event_and_duplicate_parent(self):
        parent = self.serve()[("video", 1)]
        first = self.record_view(parent)
        self.assertEqual(first, self.record_view(parent))
        self.assertEqual(first, self.record_view(parent, key="observer-recreated"))
        changed = view(parent)
        changed["payload"]["display_rank"] = 2
        with self.assertRaises(self.events.ContractError):
            self.events.record_event(self.db, event=changed)

    def test_unqualified_and_wrong_session_views_fail(self):
        parent = self.serve()[("video", 1)]
        for field, value in [("visible_fraction", .59), ("dwell_ms", 1199), ("foreground", False)]:
            ev = view(parent)
            ev["payload"][field] = value
            with self.subTest(field=field), self.assertRaises(self.events.ContractError):
                self.events.record_event(self.db, event=ev)
        with self.assertRaises(self.events.ContractError):
            self.record_view(parent, session="other")

    def test_one_explicit_outcome_cannot_credit_two_arms(self):
        a = self.record_view()
        b = self.record_view(self.serve("page-2", "cand")[("video", 1)], key="view-2", ts=115)
        oid = self.events.record_event(self.db, event=outcome(parent=b))
        self.assertEqual(self.attribute(), 1)
        self.assertEqual(self.attribute(), 0)
        self.assertEqual(self.evidence()["attributions"][0]["viewed_id"], b)
        self.assertNotEqual(a, b)
        self.assertEqual(self.evidence()["attributions"][0]["outcome_id"], oid)

    def test_explicit_unviewed_serve_does_not_fall_back(self):
        self.record_view()
        unviewed = self.serve("page-2", "cand")[("video", 1)]
        self.events.record_event(self.db, event=outcome(parent=unviewed))
        self.assertEqual(self.attribute(), 0)

    def test_newer_unassigned_surface_cannot_credit_older_arm(self):
        self.record_view()
        unassigned = self.serve("search", "", experiment=None)[("video", 1)]
        vid = self.record_view(unassigned, key="search-view", ts=115)
        self.events.record_event(self.db, event=outcome(parent=vid))
        self.attribute()
        self.assertEqual(self.evidence()["attributions"][0]["viewed_id"], vid)
        filtered = self.events.read_evidence(self.db, since_ts=95, through_ts=500, experiment_id="experiment-1")
        self.assertEqual(filtered["attributions"], [])

    def test_unknown_provenance_pre_exposure_and_wrong_session_stay_unattributed(self):
        self.record_view()
        cases = [outcome("old", ts=105), outcome("other", session="session-2"),
                 outcome("unknown", payload={"signal": "watch", "watched_s_delta": 30,
                                              "started_at": 120, "provenance": "unknown"})]
        for ev in cases:
            self.events.record_event(self.db, event=ev)
        self.assertEqual(self.attribute(), 0)
        self.assertFalse(self.evidence()["promotion_enabled"])

    def test_delayed_middle_view_cannot_make_earlier_fallback_final(self):
        self.record_view()
        self.events.record_event(self.db, event=outcome())
        self.events.attribute_outcomes(self.db, through_ts=250, window_s=100, policy_revision="policy-1")
        self.assertEqual(self.evidence()["attributions"], [])
        self.record_view(self.serve("middle", "cand")[("video", 1)], key="middle-view", ts=115)
        self.attribute()
        self.assertEqual(self.evidence()["attributions"], [])
        self.assertIn("source_completeness_unavailable", self.evidence()["validity_reasons"])

    def test_missing_original_cycles_cross_item_and_conflicting_corrections_rejected(self):
        vid = self.record_view()
        original = self.events.record_event(self.db, event=outcome(parent=vid))
        correction_payload = {"signal": "correction", "provenance": "confirmed_delta_v1"}
        corrected = self.events.record_event(self.db, event=outcome("correction", corrects=original,
            ts=160, payload=correction_payload))
        for ev in (outcome("missing-original", corrects="missing", payload=correction_payload),
                   outcome("cycle", corrects=corrected, ts=170, payload=correction_payload),
                   outcome("wrong-item", corrects=original, item_id=2, ts=170, payload=correction_payload),
                   outcome("conflict", corrects=original, ts=170, payload=correction_payload)):
            with self.assertRaises(self.events.ContractError):
                self.events.record_event(self.db, event=ev)

    def test_later_alias_merge_invalidates_claim_without_reassigning(self):
        self.events.record_session_mapping(self.db, alias_session_id="session-1",
            canonical_session_id="canonical-1", mapping_revision="mapping-1")
        vid = self.record_view()
        original = self.events.record_event(self.db, event=outcome(parent=vid))
        self.events.attribute_outcomes(self.db, through_ts=250, window_s=100, policy_revision="policy-1")
        with patch.object(self.events.time, "time", return_value=300):
            self.events.record_session_mapping(self.db, alias_session_id="canonical-1",
                canonical_session_id="canonical-2", mapping_revision="mapping-2")
        self.attribute()
        evidence = self.evidence()
        self.assertIn("session_mapping_changed", evidence["validity_reasons"])
        self.assertEqual([(r["outcome_id"], r["viewed_id"]) for r in evidence["attributions"]], [(original, vid)])
        old = self.events.read_evidence(self.db, since_ts=95, through_ts=250)
        self.assertNotIn("session_mapping_changed", old["validity_reasons"])

    def test_cached_page_cannot_be_relabelled_with_new_experiment(self):
        first = request()
        first["ranking_content_id"] = "cached-page"
        self.events.record_served(self.db, request=first, items=[item()])
        second = request("fresh-delivery", experiment="different-experiment")
        second["ranking_content_id"] = "cached-page"
        with self.assertRaises(self.events.ContractError):
            self.events.record_served(self.db, request=second, items=[item()])

    def test_correction_keeps_original_attribution_and_cancels_reward(self):
        vid = self.record_view()
        original = self.events.record_event(self.db, event=outcome(parent=vid, payload={
            "signal": "engagement", "engagement_delta": 1, "provenance": "confirmed_delta_v1"}))
        self.events.attribute_outcomes(self.db, through_ts=250, window_s=100, policy_revision="policy-1")
        correction = self.events.record_event(self.db, event=outcome("undo", ts=350, corrects=original,
            payload={"signal": "correction", "provenance": "confirmed_delta_v1"}))
        self.attribute()
        claims = {r["outcome_id"]: r["viewed_id"] for r in self.evidence()["attributions"]}
        self.assertEqual(claims, {original: vid, correction: vid})
        summary = self.summary()
        self.assertEqual(summary["trials"][0]["reward"], 0)

    def summary(self, *, watched=0, rating=None, engagement_count=0, cutoff=500):
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=cutoff)
        cumulative = {"cutoff_ts": cutoff, "items": {("video", 1): {
            "watched_s": watched, "duration_s": 600, "rating": rating, "engagement_count": engagement_count}}}
        return self.events.summarize_trials(evidence, verdict=verdict, trial_reward=trial_reward,
                                            cumulative_at_cutoff=cumulative)

    def test_shared_reward_uses_only_new_deltas_once_per_exposure(self):
        vid = self.record_view()
        for i in range(3):
            self.events.record_event(self.db, event=outcome("watch-" + str(i), ts=150 + i, parent=vid,
                payload={"signal": "watch", "watched_s_delta": 100, "started_at": 120,
                         "provenance": "confirmed_delta_v1"}))
        self.attribute()
        summary = self.summary(watched=300)
        self.assertEqual(len(summary["trials"]), 1)
        self.assertAlmostEqual(summary["trials"][0]["reward"], 300 / 3600)
        self.assertEqual(len(summary["sessions"]), 1)

    def test_strict_values_and_sensitive_configuration(self):
        for bad in [float("nan"), float("inf"), -1, True]:
            it = item()
            it["score"] = bad
            if bad == -1:
                it["id"] = -1
            with self.subTest(bad=bad), self.assertRaises(self.events.ContractError):
                self.events.record_served(self.db, request=request(), items=[it])
        req = request()
        req["config"]["query"] = "private query"
        with self.assertRaises(self.events.ContractError):
            self.events.record_served(self.db, request=req, items=[item()])

    def test_absent_and_existing_reads_do_not_create_or_modify_files(self):
        missing = str(Path(self.temp.name) / "absent.sqlite")
        self.assertEqual(self.events.read_evidence(missing, since_ts=0, through_ts=500)["status"], "unavailable")
        self.assertFalse(Path(missing).exists())
        self.serve()
        before = {p.name: p.read_bytes() for p in Path(self.temp.name).iterdir()}
        self.evidence()
        self.events.read_view_counts(self.db, since_ts=0, through_ts=500)
        self.assertEqual(before, {p.name: p.read_bytes() for p in Path(self.temp.name).iterdir()})

    def test_legacy_tables_retained_and_never_attributed(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("CREATE TABLE predictions(ts REAL, item_id INTEGER)")
            conn.execute("INSERT INTO predictions VALUES(1,1)")
        self.attribute()
        self.assertEqual(self.evidence()["events"], [])
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT * FROM predictions").fetchall(), [(1, 1)])

    def test_observer_alias_cannot_be_reused_for_another_parent(self):
        parent = self.serve()[("video", 1)]
        self.record_view(parent)
        self.record_view(parent, key="alias")
        other = self.serve("page-2")[("video", 1)]
        with self.assertRaises(self.events.ContractError):
            self.record_view(other, key="alias")

    def test_late_evidence_is_not_assigned_on_later_runs(self):
        self.record_view()
        self.events.attribute_outcomes(self.db, through_ts=250, window_s=100, policy_revision="policy-1")
        self.events.record_event(self.db, event=outcome())
        self.assertEqual(self.attribute(), 0)
        self.assertIn("source_completeness_unavailable", self.evidence()["validity_reasons"])
        self.assertEqual(self.events.attribute_outcomes(self.db, through_ts=600, window_s=100,
                                                       policy_revision="policy-1"), 0)

    def test_future_attribution_run_does_not_leak_into_earlier_read(self):
        vid = self.record_view()
        self.events.record_event(self.db, event=outcome(parent=vid))
        self.attribute()
        older = self.events.read_evidence(self.db, since_ts=95, through_ts=250)
        self.assertEqual(older["attributions"], [])
        self.assertFalse(older["valid"])

    def test_append_only_tables_reject_sql_update_and_delete(self):
        self.serve()
        with closing(sqlite3.connect(self.db)) as conn:
            for sql in ("DELETE FROM rec_events", "UPDATE rec_requests SET context_id='changed'"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(sql)

    def test_one_experiment_cannot_silently_change_arm_configuration(self):
        self.serve()
        changed = request("page-2")
        changed["arms"]["cand"]["embedding_weight"] = .5
        with self.assertRaises(self.events.ContractError):
            self.events.record_served(self.db, request=changed, items=[item()])
        changed["experiment_id"] = "experiment-2"
        self.events.record_served(self.db, request=changed, items=[item()])

    def test_experiment_filter_does_not_hide_unknown_unlinked_outcomes(self):
        self.record_view()
        self.events.record_event(self.db, event=outcome(payload={"signal": "engagement", "engagement_delta": 1,
                                                               "provenance": "unknown"}))
        self.attribute()
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=500, experiment_id="experiment-1")
        self.assertIn("unknown_source_provenance", evidence["validity_reasons"])

    def test_unrelated_late_history_does_not_invalidate_another_cohort(self):
        self.record_view()
        self.events.attribute_outcomes(self.db, through_ts=250, window_s=100, policy_revision="policy-1")
        self.events.record_event(self.db, event=outcome(session="other-session"))
        self.attribute()
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=500, experiment_id="experiment-1")
        self.assertNotIn("late_evidence", evidence["validity_reasons"])

    def test_feedback_validity_respects_snapshot_cutoff(self):
        self.record_view()
        self.attribute()
        op = {"operation_id": "later-feedback", "kind": "image", "item_id": 9,
              "action": "engagement", "session_id": "session-1"}
        with patch.object(self.events.time, "time", return_value=600):
            self.events.perform_feedback(self.db, operation=op,
                read_current=lambda _: {"status": "ok", "rating100": None, "engagement_count": 0},
                apply_change=lambda *_: {"status": "indeterminate"})
        self.assertNotIn("feedback_unresolved", self.evidence()["validity_reasons"])
        with patch.object(self.events.time, "time", return_value=800):
            self.events.reconcile_feedback(self.db, operation_id="later-feedback",
                read_current=lambda _: {"status": "ok", "rating100": None, "engagement_count": 1})
        snapshot = self.events.read_evidence(self.db, since_ts=95, through_ts=700)
        self.assertIn("feedback_unresolved", snapshot["validity_reasons"])

    def test_same_timestamp_later_view_does_not_get_prior_outcome(self):
        self.events.record_event(self.db, event=outcome(ts=150, payload={"signal": "engagement", "engagement_delta": 1,
                                                                      "provenance": "confirmed_delta_v1"}))
        self.record_view(ts=150)
        self.assertEqual(self.attribute(), 0)

    def test_strict_delta_and_parent_schemas(self):
        vid = self.record_view()
        cases = [outcome(parent="missing"), outcome(parent=vid, item_id=2),
                 outcome(payload={"signal": "watch", "watched_s_delta": float("nan"),
                                  "started_at": 120, "provenance": "confirmed_delta_v1"}),
                 outcome(payload={"signal": "watch", "watched_s_delta": -1,
                                  "started_at": 120, "provenance": "confirmed_delta_v1"}),
                 outcome(payload={"signal": "watch", "watched_s_delta": 1,
                                  "watched_s": 3600, "started_at": 120, "provenance": "confirmed_delta_v1"}),
                 outcome(payload={"signal": "rating", "rating_before": 80, "rating_after": 80,
                                  "provenance": "confirmed_delta_v1"}),
                 outcome(payload={"signal": "rating", "rating_before": None, "rating_after": 101,
                                  "provenance": "confirmed_delta_v1"}),
                 outcome(payload={"signal": "engagement", "engagement_delta": True, "provenance": "confirmed_delta_v1"})]
        for i, ev in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(self.events.ContractError):
                self.events.record_event(self.db, event=ev)
        with self.assertRaises(self.events.ContractError):
            self.events.record_event(self.db, event=view(vid))

    def test_immature_expired_and_explicit_original_arm(self):
        a = self.record_view()
        self.record_view(self.serve("second", "cand")[("video", 1)], key="second-view", ts=120)
        self.events.record_event(self.db, event=outcome(parent=a))
        self.events.attribute_outcomes(self.db, through_ts=205, window_s=100, policy_revision="policy-1")
        self.assertEqual(self.evidence()["attributions"], [])
        self.assertEqual(self.attribute(), 1)
        self.assertEqual(self.evidence()["attributions"][0]["viewed_id"], a)
        with self.assertRaises(self.events.ContractError):
            self.events.attribute_outcomes(self.db, through_ts=600, window_s=200, policy_revision="policy-1")

    def test_advance_attribution_pins_production_policy_monotone_and_rate_limited(self):
        vid = self.record_view()
        self.events.record_event(self.db, event=outcome(parent=vid))
        # First advance records the pinned production policy; the window has not
        # elapsed at the current clock, so nothing finalizes yet.
        self.assertEqual(self.events.advance_attribution(self.db), 0)
        runs = self.runs()
        self.assertEqual([(r[1], r[2]) for r in runs],
                         [(self.events.ATTRIBUTION_WINDOW_S, self.events.ATTRIBUTION_POLICY_REVISION)])
        self.assertEqual(runs[-1][0], 200.0)
        # Rate limit: another advance inside the interval records no new run.
        self.assertEqual(self.events.advance_attribution(self.db, now=200.0 + 1), 0)
        self.assertEqual(len(self.runs()), 1)
        # A clock that moved backwards is absorbed, never an error.
        self.assertEqual(self.events.advance_attribution(self.db, now=100.0), 0)
        self.assertEqual(len(self.runs()), 1)
        # Once the window has elapsed, the outcome finalizes under the same policy.
        self.assertEqual(self.events.advance_attribution(self.db, now=5000.0), 1)
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=5000)
        self.assertEqual(evidence["attributions"][0]["viewed_id"], vid)
        self.assertEqual(evidence["attributions"][0]["policy_revision"],
                         self.events.ATTRIBUTION_POLICY_REVISION)
        # The pin is exclusive: an ad hoc policy on the same ledger is rejected.
        with self.assertRaises(self.events.ContractError):
            self.events.attribute_outcomes(self.db, through_ts=6000, window_s=100, policy_revision="policy-1")

    def runs(self):
        with self.events._connection(self.db) as conn:
            return conn.execute("SELECT through_ts,window_s,policy_revision FROM rec_attribution_runs ORDER BY through_ts").fetchall()

    def test_old_rating_or_o_floor_cannot_be_smuggled_into_watch_delta(self):
        vid = self.record_view()
        for field in ("rating", "engagement_count", "rating_before", "engagement_count"):
            ev = outcome()
            ev["payload"][field] = 80
            with self.subTest(field=field), self.assertRaises(self.events.ContractError):
                self.events.record_event(self.db, event=ev)
        self.events.record_event(self.db, event=outcome(parent=vid, payload={"signal": "watch", "watched_s_delta": 5,
            "started_at": 140, "provenance": "confirmed_delta_v1"}))
        self.attribute()
        self.assertEqual(self.summary()["trials"][0]["reward"], 0)

    def test_window_inclusion_and_fatigue_days(self):
        vid = self.record_view()
        self.events.record_event(self.db, event=outcome("on-boundary", ts=210, parent=vid, payload={"signal": "engagement", "engagement_delta": 1,
                                                                                      "provenance": "confirmed_delta_v1"}))
        self.events.record_event(self.db, event=outcome("past-boundary", ts=211, parent=vid, payload={"signal": "engagement", "engagement_delta": 1,
                                                                                        "provenance": "confirmed_delta_v1"}))
        self.assertEqual(self.attribute(), 1)
        self.record_view(self.serve("another-day")[("video", 1)], key="later-view", ts=86500)
        counts = self.events.read_view_counts(self.db, since_ts=95, through_ts=90000)
        self.assertEqual(counts["counts"], {("video", 1): 2})

    def test_cumulative_like_and_o_gate_never_mint_new_floors(self):
        vid = self.record_view()
        self.events.attribute_outcomes(self.db, through_ts=250, window_s=100, policy_revision="policy-1")
        empty = self.summary(watched=3600, rating=100, engagement_count=5, cutoff=250)["trials"][0]
        self.assertTrue(empty["liked"])
        self.assertEqual(empty["reward"], 0)
        self.events.record_event(self.db, event=outcome(parent=vid, payload={"signal": "watch", "watched_s_delta": 5,
            "started_at": 140, "provenance": "confirmed_delta_v1"}))
        self.attribute()
        fresh = self.summary(watched=3605, rating=100, engagement_count=5)["trials"][0]
        self.assertTrue(fresh["liked"])
        self.assertAlmostEqual(fresh["reward"], 5 / 3600)

    def test_missing_cutoff_facts_cannot_substitute_delta_verdict(self):
        self.record_view()
        self.attribute()
        def forbidden(*args, **kwargs):
            self.fail("missing cumulative inputs must not call shared math")
        summary = self.events.summarize_trials(self.evidence(), verdict=forbidden, trial_reward=forbidden)
        self.assertIsNone(summary["trials"][0]["reward"])
        self.assertIn("cumulative_verdict_unavailable", summary["validity_reasons"])
        with self.assertRaises(self.events.ContractError):
            self.events.summarize_trials(self.evidence(), verdict=forbidden, trial_reward=forbidden,
                                         cumulative_at_cutoff={"cutoff_ts": 600, "items": {}})

    def test_correction_folds_before_reward_cap_and_keeps_owner(self):
        vid = self.record_view()
        ids = [self.events.record_event(self.db, event=outcome("long-" + str(i), parent=vid,
               payload={"signal": "watch", "watched_s_delta": 2000, "started_at": 120,
                        "provenance": "confirmed_delta_v1"})) for i in range(2)]
        self.events.attribute_outcomes(self.db, through_ts=250, window_s=100, policy_revision="policy-1")
        self.assertEqual(self.summary(watched=6000, cutoff=250)["trials"][0]["reward"], 1)
        self.events.record_event(self.db, event=outcome("correct-long", ts=350, corrects=ids[0],
            payload={"signal": "correction", "provenance": "confirmed_delta_v1"}))
        self.attribute()
        corrected = self.summary(watched=4000)["trials"][0]
        self.assertAlmostEqual(corrected["reward"], 2000 / 3600)
        self.assertEqual(corrected["viewed_id"], vid)

    def test_unresolved_original_cannot_mature_correction_into_zero_or_new_credit(self):
        self.record_view()
        original = self.events.record_event(self.db, event=outcome())
        self.events.record_event(self.db, event=outcome("pending-correction", corrects=original, ts=160,
            payload={"signal": "correction", "provenance": "confirmed_delta_v1"}))
        self.attribute()
        self.assertEqual(self.evidence()["attributions"], [])
        trial = self.summary(watched=3600, engagement_count=5)["trials"][0]
        self.assertIsNone(trial["reward"])
        self.assertIn("original_unresolved", trial["validity_reasons"])

    def test_unknown_correction_blocks_effective_reward_without_reassigning_owner(self):
        vid = self.record_view()
        original = self.events.record_event(self.db, event=outcome(parent=vid, payload={
            "signal": "engagement", "engagement_delta": 1, "provenance": "confirmed_delta_v1"}))
        self.events.attribute_outcomes(self.db, through_ts=250, window_s=100, policy_revision="policy-1")
        self.events.record_event(self.db, event=outcome("uncertain-undo", corrects=original, ts=350,
            payload={"signal": "correction", "provenance": "unknown"}))
        self.attribute()
        self.assertEqual(len(self.evidence()["attributions"]), 1)
        self.assertIsNone(self.summary(watched=3600, engagement_count=5)["trials"][0]["reward"])

    def test_aliases_resolve_before_explicit_attribution_and_cycles_rejected(self):
        self.events.record_session_mapping(self.db, alias_session_id="alias-a",
            canonical_session_id="canonical", mapping_revision="r1")
        self.events.record_session_mapping(self.db, alias_session_id="alias-b",
            canonical_session_id="canonical", mapping_revision="r1")
        parent = self.serve(session="alias-a")[("video", 1)]
        vid = self.record_view(parent, session="alias-b")
        self.events.record_event(self.db, event=outcome(parent=vid, session="canonical"))
        self.assertEqual(self.attribute(), 1)
        self.assertEqual(set(self.evidence()["sessions"]), {"canonical"})
        self.assertTrue(all(e["session_mapping_revision"] for e in self.evidence()["events"]))
        with self.assertRaises(self.events.ContractError):
            self.events.record_session_mapping(self.db, alias_session_id="canonical",
                canonical_session_id="alias-a", mapping_revision="cycle")

    def test_unknown_alias_stays_unattributed_even_with_explicit_reference(self):
        parent = self.serve(session="unresolved")[("video", 1)]
        vid = self.record_view(parent, session="unresolved")
        self.events.record_event(self.db, event=outcome(parent=vid, session="unresolved"))
        self.assertEqual(self.attribute(), 0)
        self.assertIn("session_mapping_unresolved", self.evidence()["validity_reasons"])

    def test_cache_delivery_preserves_original_binding_and_order(self):
        first = request()
        refs = self.events.record_served(self.db, request=first, items=[item(), item(2)])
        with self.assertRaises(self.events.ContractError):
            self.events.record_served(self.db, request=first, items=[item(2), item()])
        delivery = {**first, "request_id": "new-delivery", "client_request_id": "new-client", "created_at": 101}
        new_refs = self.events.record_served(self.db, request=delivery, items=[item(), item(2)])
        self.assertNotEqual(new_refs, refs)
        self.assertEqual({r["experiment_id"] for r in self.evidence()["requests"]}, {first["experiment_id"]})

    def test_import_does_not_open_files_database_network_or_start_threads(self):
        with patch("sqlite3.connect", side_effect=AssertionError("database")), \
             patch("threading.Thread.start", side_effect=AssertionError("thread")), \
             patch("socket.socket", side_effect=AssertionError("network")):
            load_events()

    def test_reads_on_readonly_file_and_corrupt_store_are_safe(self):
        self.serve()
        Path(self.db).chmod(0o444)
        try:
            self.assertEqual(self.evidence()["status"], "ok")
        finally:
            Path(self.db).chmod(0o600)
        bad = Path(self.temp.name) / "bad.sqlite"
        bad.write_bytes(b"not a database")
        before = bad.read_bytes()
        self.assertEqual(self.events.read_evidence(str(bad), since_ts=0, through_ts=500)["status"], "unavailable")
        self.assertEqual(bad.read_bytes(), before)

    def test_real_processes_deduplicate_visibility(self):
        parent = self.serve()[("video", 1)]
        ctx = multiprocessing.get_context("spawn")
        ready, results, release = ctx.Queue(), ctx.Queue(), ctx.Event()
        workers = [ctx.Process(target=concurrent_record, args=(self.db, parent, ready, release, results)) for _ in range(2)]
        for p in workers:
            p.start()
        for _ in workers:
            ready.get(timeout=10)
        release.set()
        ids = [results.get(timeout=10) for _ in workers]
        for p in workers:
            p.join(10)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(len(ids[0]), 36)

    def test_evidence_explicitly_reports_missing_watch_capture(self):
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=99999999999)
        self.assertFalse(evidence["watch_capture_supported"])
        self.assertIn("watch_capture_unavailable", evidence["promotion_reasons"])


class EligibilitySnapshotContracts(unittest.TestCase):
    setUp = EventContracts.setUp

    def publish(self, ids=(1, 2), *, mode="video", **changes):
        values = dict(saved_filter_id="7", mode=mode, predicate_sha256="a" * 64,
                      eligible_ids={"video": list(ids) if mode == "video" else [],
                                    "image": list(ids) if mode == "image" else []}, observed_at=199)
        values.update(changes)
        return self.events.put_eligibility_snapshot(self.db, **values)

    def deliver(self, snapshot, *, req=None, items=None, roles=None):
        req = req or {**request(), "created_at": 201, "preference_cutoff": 200}
        roles = roles if roles is not None else {"generation": snapshot, "delivery": snapshot}
        return self.events.record_served(self.db, request=req, items=items or [item()],
                                         eligibility_snapshots=roles)

    def test_large_sparse_snapshot_nested_event_receipt_and_export(self):
        ids = [1 + i * 100003 for i in range(100000)]
        published = self.publish(ids)
        sid = published["snapshot_id"]
        self.assertEqual(published["counts"], {"video": 100000, "image": 0})
        self.assertEqual(self.events.read_eligibility_snapshot(self.db, snapshot_id=sid)["eligible_ids"],
                         {"video": ids, "image": []})
        ref = {"snapshot_id": sid}
        rows = [{**item(i), "source_rank": n, "trace": {
            "ranking_provenance": {"eligible_ids": ref}, "delivery_context": {"eligible_ids": ref},
            "details": "x" * 10000}} for n, i in enumerate(ids[:60])]
        with patch.object(self.events.time, "time", return_value=202):
            first = self.deliver(sid, items=rows)
            self.assertEqual(self.deliver(sid, items=rows), first)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_eligibility_snapshots").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_request_eligibility").fetchone()[0], 2)
            self.assertLess(conn.execute("SELECT max(length(payload_json)) FROM rec_events").fetchone()[0], 262144)
            self.assertLess(conn.execute("SELECT max(length(submitted_json)) FROM rec_event_receipts").fetchone()[0], 262144)
        before = Path(self.db).read_bytes()
        plain = self.events.read_evidence(self.db, since_ts=95, through_ts=500)
        self.assertNotIn("eligibility_snapshots", plain)
        with patch.object(self.events, "read_eligibility_snapshot", wraps=self.events.read_eligibility_snapshot) as reader:
            exported = self.events.read_evidence(self.db, since_ts=95, through_ts=500, include_eligibility_snapshots=True)
            reader.assert_called_once_with(self.db, snapshot_id=sid)
        self.assertEqual(set(exported["eligibility_snapshots"]), {sid})
        self.assertEqual(exported["eligibility_snapshots"][sid]["eligible_ids"]["video"], ids)
        self.assertEqual(exported["requests"][0]["eligibility_snapshots"], {"generation": sid, "delivery": sid})
        self.assertEqual(Path(self.db).read_bytes(), before)
        with self.assertRaisesRegex(self.events.ContractError, "payload_too_large"):
            self.events._json({"value": "x" * 262144})

    def test_canonical_publication_concurrent_and_immutable(self):
        import hashlib
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.publish([2, 1]), range(8)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(self.publish([1, 2], observed_at=200), results[0])
        sid = results[0]["snapshot_id"]
        membership_hash = hashlib.sha256(b'{"image":[],"video":[1,2]}').hexdigest()
        expected = "elig:v1:" + hashlib.sha256(json.dumps(
            [1, "7", "video", "a" * 64, membership_hash], separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(sid, expected)
        read = self.events.read_eligibility_snapshot(self.db, snapshot_id=sid)
        self.assertEqual(read["metadata"]["membership_sha256"], membership_hash)
        self.assertEqual(read["metadata"]["observed_at"], 199)
        read["eligible_ids"]["video"].clear()
        self.assertEqual(self.events.read_eligibility_snapshot(self.db, snapshot_id=sid)["eligible_ids"]["video"], [1, 2])
        self.deliver(sid)
        with closing(sqlite3.connect(self.db)) as conn:
            for table in ("rec_eligibility_snapshots", "rec_request_eligibility"):
                for statement in (f"DELETE FROM {table}", f"UPDATE {table} SET snapshot_id=snapshot_id"):
                    with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                        conn.execute(statement)

    def test_role_conflicts_and_membership_are_atomic(self):
        first, second = self.publish([1, 2])["snapshot_id"], self.publish([1, 3])["snapshot_id"]
        self.deliver(first)
        for roles in ({"generation": second, "delivery": first}, {"generation": first}, {}):
            with self.subTest(roles=roles), self.assertRaises(self.events.ContractError):
                self.deliver(first, roles=roles)
        new = {**request("next"), "created_at": 202, "preference_cutoff": 200}
        with self.assertRaisesRegex(self.events.ContractError, "eligibility_snapshot_missing"):
            self.deliver("elig:v1:" + "0" * 64, req=new)
        with self.assertRaises(self.events.ContractError):
            self.deliver(first, req=new, items=[item(3)])
        with self.assertRaises(self.events.ContractError):
            self.deliver(first, req=new, items=[item(1, kind="image")])
        bad_trace = {**item(), "trace": {"ranking_provenance": {"eligible_ids": {"snapshot_id": second}}}}
        with self.assertRaises(self.events.ContractError):
            self.deliver(first, req=new, items=[bad_trace])
        same_content = {**new, "ranking_content_id": request()["ranking_content_id"]}
        with self.assertRaises(self.events.ContractError):
            self.deliver(second, req=same_content)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_requests").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_request_eligibility").fetchone()[0], 2)

    def test_generation_delivery_can_differ_but_both_constrain_rows(self):
        original = self.publish([1, 2])["snapshot_id"]
        with patch.object(self.events.time, "time", return_value=205):
            current = self.publish([2, 3], observed_at=204)["snapshot_id"]
        roles = {"generation": original, "delivery": current}
        req = {**request(), "created_at": 206, "preference_cutoff": 200}
        with self.assertRaises(self.events.ContractError):
            self.deliver(original, req=req, roles=roles, items=[item(3)])
        with self.assertRaises(self.events.ContractError):
            self.deliver(original, req=req, roles=roles, items=[item(1)])
        self.deliver(original, req=req, roles=roles, items=[{**item(2), "trace": {
            "ranking_provenance": {"eligible_ids": {"snapshot_id": original}},
            "delivery_context": {"eligible_ids": {"snapshot_id": current}}}}])
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=500, include_eligibility_snapshots=True)
        self.assertEqual(set(evidence["eligibility_snapshots"]), {original, current})

    def test_associations_rollback_with_failed_served_insert_and_require_migration(self):
        sid = self.publish()["snapshot_id"]
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("CREATE TRIGGER fail_served BEFORE INSERT ON rec_events BEGIN SELECT RAISE(ABORT,'fixture'); END")
        with self.assertRaises(self.events.ContractError):
            self.deliver(sid)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_request_eligibility").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_requests").fetchone()[0], 0)
            conn.execute("DROP TRIGGER fail_served")
            conn.execute("DROP TABLE rec_request_eligibility")
            conn.execute("DROP TABLE rec_eligibility_snapshots")
        before = Path(self.db).read_bytes()
        with self.assertRaises(self.events.ContractError):
            self.publish()
        with self.assertRaises(self.events.ContractError):
            self.deliver(sid)
        self.assertEqual(Path(self.db).read_bytes(), before)
        self.events.record_served(self.db, request=request(), items=[item()])
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertIsNone(conn.execute("SELECT name FROM sqlite_master WHERE name='rec_eligibility_snapshots'").fetchone())

    def test_experiment_generation_binding_and_export_are_scoped(self):
        first = self.publish([1, 2])["snapshot_id"]
        other = self.publish([1, 3])["snapshot_id"]
        self.deliver(first, req={**request(experiment="q7-experiment"), "created_at": 201, "preference_cutoff": 200})
        with self.assertRaisesRegex(self.events.ContractError, "experiment_conflict"):
            self.deliver(other, req={**request("other", experiment="q7-experiment"), "created_at": 202, "preference_cutoff": 200})
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("DROP TRIGGER rec_eligibility_snapshots_no_update")
            conn.execute("UPDATE rec_eligibility_snapshots SET membership_json='{}' WHERE snapshot_id=?", (other,))
        exported = self.events.read_evidence(self.db, since_ts=95, through_ts=500, include_eligibility_snapshots=True)
        self.assertEqual(exported["status"], "ok")
        self.assertEqual(set(exported["eligibility_snapshots"]), {first})

    def test_empty_kind_validation_and_integer_limits(self):
        for mode in ("video", "image"):
            result = self.publish([], mode=mode)
            self.assertEqual(result["counts"], {"video": 0, "image": 0})
            with self.assertRaises(self.events.ContractError):
                self.deliver(result["snapshot_id"])
            valid = self.publish([9223372036854775807], mode=mode)
            self.assertEqual(sum(valid["counts"].values()), 1)
        invalid = [dict(eligible_ids={"video": [1, 1], "image": []}),
                   dict(eligible_ids={"video": [], "image": [1]}),
                   dict(eligible_ids={"video": None, "image": []}),
                   dict(mode="OTHER"), dict(saved_filter_id="0"), dict(predicate_sha256="A" * 64),
                   dict(observed_at=201), dict(observed_at=float("nan"))]
        invalid += [dict(eligible_ids={"video": [value], "image": []}) for value in
                    (True, 1.0, -1, 0, "1", 9223372036854775808)]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(self.events.ContractError):
                self.publish(**changes)

    def test_missing_or_corrupt_lookup_never_initializes_or_repairs(self):
        sid = self.publish()["snapshot_id"]
        self.deliver(sid)
        missing = str(Path(self.temp.name) / "missing.sqlite")
        with self.assertRaises(self.events.ContractError):
            self.events.read_eligibility_snapshot(missing, snapshot_id=sid)
        self.assertFalse(Path(missing).exists())
        for invalid in ("elig:v1:" + "0" * 64, "../private", True):
            with self.assertRaises(self.events.ContractError):
                self.events.read_eligibility_snapshot(self.db, snapshot_id=invalid)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("DROP TRIGGER rec_eligibility_snapshots_no_update")
            conn.execute("UPDATE rec_eligibility_snapshots SET membership_json=?", ('{"video":[99],"image":[]}',))
        before = Path(self.db).read_bytes()
        with self.assertRaises(self.events.ContractError):
            self.events.read_eligibility_snapshot(self.db, snapshot_id=sid)
        with self.assertRaises(self.events.ContractError):
            self.deliver(sid)
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=500, include_eligibility_snapshots=True)
        self.assertEqual(evidence["status"], "unavailable")
        self.assertEqual(Path(self.db).read_bytes(), before)



if __name__ == "__main__":
    unittest.main()

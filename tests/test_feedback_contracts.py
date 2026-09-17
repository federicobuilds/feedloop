"""Real SQLite journal tests with fake, injected authorities; no host imports."""
from contextlib import closing
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from test_ledger_contracts import load_events, request, item, view


def operation(key="op-1", *, kind="video", item_id=1, action="rating", rating=80, undo=None):
    op = dict(operation_id=key, kind=kind, item_id=item_id, action=action, session_id="session-1")
    if action == "rating":
        op["rating100"] = rating
    elif action == "undo":
        op["undo_of"] = undo
    return op


class Authority:
    def __init__(self, rating=None, engagement=0):
        self.values = {"rating100": rating, "engagement_count": engagement}
        self.reads, self.writes = 0, 0

    def read(self, key):
        self.reads += 1
        return {"status": "ok", **self.values}

    def apply(self, key, change):
        self.writes += 1
        if change["action"] == "rating":
            self.values["rating100"] = change["rating100"]
        else:
            self.values["engagement_count"] += change["delta"]
        return {"status": "confirmed", **self.values}


def hold_feedback(db, ready, release, results):
    events = load_events()
    authority = Authority()
    def apply(key, change):
        ready.set()
        release.wait(10)
        return authority.apply(key, change)
    results.put(events.perform_feedback(db, operation=operation("owner"), read_current=authority.read, apply_change=apply))


def crash_feedback(db, counter, stage):
    events = load_events()
    def read(key):
        if stage == "planned":
            os._exit(17)
        return {"status": "ok", "rating100": None, "engagement_count": counter.value}
    def apply(key, change):
        with counter.get_lock():
            counter.value += change["delta"]
        os._exit(17)
    events.perform_feedback(db, operation=operation("crashed", action="engagement"), read_current=read, apply_change=apply)


class FeedbackContracts(unittest.TestCase):
    def setUp(self):
        self.events = load_events()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = str(Path(self.temp.name) / "events.sqlite")
        self.events.initialize_event_store(self.db, cutover_ts=95)
        with patch.object(self.events.time, "time", return_value=100):
            self.events.record_session_mapping(self.db, alias_session_id="session-1",
                canonical_session_id="session-1", mapping_revision="initial")
        self.authority = Authority()

    def perform(self, op=None, read=None, apply=None):
        return self.events.perform_feedback(self.db, operation=op or operation(),
                                            read_current=read or self.authority.read,
                                            apply_change=apply or self.authority.apply)

    def records(self):
        with closing(sqlite3.connect(self.db)) as conn:
            return conn.execute("SELECT state,facts_json FROM feedback_steps ORDER BY step_id").fetchall()

    def test_journal_precedes_external_write_and_no_transaction_spans_callbacks(self):
        def check_lock():
            with closing(sqlite3.connect(self.db, timeout=.1)) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.rollback()
        def read(key):
            check_lock()
            self.assertEqual(self.records()[-1][0], "planned")
            return self.authority.read(key)
        def apply(key, change):
            check_lock()
            state, facts = self.records()[-1]
            self.assertEqual(state, "sent")
            self.assertIsNone(json.loads(facts)["before"]["rating100"])
            return self.authority.apply(key, change)
        result = self.perform(read=read, apply=apply)
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual([r[0] for r in self.records()], ["planned", "sent", "confirmed"])

    def test_duplicate_operation_never_repeats_callbacks_and_conflicting_reuse_fails(self):
        result = self.perform()
        self.assertEqual(self.perform(), result)
        self.assertEqual((self.authority.reads, self.authority.writes), (1, 1))
        with self.assertRaises(self.events.ContractError):
            self.perform(operation(rating=20))

    def test_confirmation_outcome_and_release_rollback_together(self):
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("""CREATE TRIGGER reject_confirmation BEFORE INSERT ON feedback_steps
                WHEN NEW.state='confirmed' BEGIN SELECT RAISE(ABORT,'test_failure'); END""")
        with self.assertRaises(self.events.ContractError):
            self.perform()
        self.assertEqual(self.authority.writes, 1)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_events").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT active_operation FROM feedback_ownership").fetchone()[0], "op-1")
            self.assertEqual(conn.execute("SELECT state FROM feedback_steps ORDER BY step_id DESC LIMIT 1").fetchone()[0], "sent")
            owner = conn.execute("SELECT owner_token FROM feedback_ownership").fetchone()[0]
            self.assertEqual(owner, conn.execute("SELECT owner_token FROM feedback_operations").fetchone()[0])

    def test_reconciliation_rotates_fence_and_stale_reconciler_cannot_finish(self):
        self.perform(apply=lambda *_: {"status": "indeterminate"})
        with closing(sqlite3.connect(self.db)) as conn:
            first = conn.execute("SELECT owner_token FROM feedback_ownership").fetchone()[0]
        def superseded_read(key):
            self.events.reconcile_feedback(self.db, operation_id="op-1", read_current=self.authority.read)
            return {"status": "ok", "rating100": 80, "engagement_count": 0}
        result = self.events.reconcile_feedback(self.db, operation_id="op-1", read_current=superseded_read)
        self.assertEqual(result["reason"], "stale_owner")
        with closing(sqlite3.connect(self.db)) as conn:
            token = conn.execute("SELECT owner_token FROM feedback_ownership").fetchone()[0]
            self.assertNotEqual(first, token)
            self.assertEqual(token, conn.execute("SELECT owner_token FROM feedback_steps ORDER BY step_id DESC LIMIT 1").fetchone()[0])
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_events").fetchone()[0], 0)

    def test_busy_rejection_still_reserves_operation_identity(self):
        self.perform(operation("busy", action="engagement"), apply=lambda *_: {"status": "indeterminate"})
        conflict = self.perform(operation("rejected", rating=20))
        self.assertEqual(conflict["reason"], "item_owned")
        with self.assertRaises(self.events.ContractError):
            self.perform(operation("rejected", rating=60))

    def test_authoritative_conflict_is_terminal_without_outcome(self):
        result = self.perform(apply=lambda *_: {"status": "conflict", "rating100": 60, "engagement_count": 0})
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(self.perform(), result)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_events WHERE event_type='outcome'").fetchone()[0], 0)

    def test_reload_cannot_reconcile_live_dispatch(self):
        def apply(key, change):
            reloaded = load_events()
            result = reloaded.reconcile_feedback(self.db, operation_id="op-1", read_current=self.authority.read)
            self.assertEqual(result["reason"], "dispatch_in_progress")
            return self.authority.apply(key, change)
        self.assertEqual(self.perform(apply=apply)["status"], "confirmed")

    def test_stale_worker_cannot_confirm_or_release_reconciled_new_owner(self):
        def late_apply(key, change):
            confirmed = self.authority.apply(key, change)
            # A real SQLite handoff, emulating a reconciler in a different process
            # after its liveness check reported the original worker dead.
            with patch.object(self.events.os, "getpid", return_value=987654321), \
                 patch.object(self.events.os, "kill", side_effect=ProcessLookupError):
                reconciled = self.events.reconcile_feedback(self.db, operation_id="op-1", read_current=self.authority.read)
            self.assertEqual(reconciled["status"], "confirmed")
            self.perform(operation("new-owner", action="engagement"), apply=lambda *_: {"status": "indeterminate"})
            return confirmed
        result = self.perform(apply=late_apply)
        self.assertEqual(result["reason"], "stale_owner")
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT active_operation FROM feedback_ownership WHERE kind='video' AND item_id=1").fetchone()[0], "new-owner")
            payloads = [json.loads(r[0]) for r in conn.execute("SELECT payload_json FROM rec_events WHERE source_event_id='op-1'")]
            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0]["provenance"], "unknown")

    def test_external_increment_matching_expected_counter_does_not_confirm_our_write(self):
        self.perform(operation(action="engagement"), apply=lambda *_: {"status": "indeterminate"})
        self.authority.values["engagement_count"] = 1
        result = self.events.reconcile_feedback(self.db, operation_id="op-1", read_current=self.authority.read)
        self.assertEqual(result["status"], "indeterminate")
        self.assertEqual(result["reason"], "counter_confirmation_ambiguous")
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rec_events").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT active_operation FROM feedback_ownership").fetchone()[0], "op-1")

    def test_counter_undo_is_once_only_and_restores_original(self):
        self.authority.values["engagement_count"] = 4
        original = self.perform(operation(action="engagement"))
        undo = operation("undo", action="undo", undo="op-1")
        result = self.perform(undo)
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(self.authority.values["engagement_count"], 4)
        self.assertEqual(self.perform(undo), result)
        self.assertEqual(self.authority.writes, 2)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT corrects_id FROM rec_events WHERE event_id=?", (result["event_id"],)).fetchone()[0], original["event_id"])

    def test_wrong_reference_rejected_before_any_callback(self):
        op = operation()
        op["viewed_event_id"] = "missing"
        with self.assertRaises(self.events.ContractError):
            self.perform(op)
        self.assertEqual((self.authority.reads, self.authority.writes), (0, 0))

    def test_exact_null_neutral_and_other_rating_undo(self):
        for index, original in enumerate([None, 0, 20, 60, 90]):
            with self.subTest(original=original):
                auth = Authority(original)
                key = "rating-" + str(index)
                op = operation(key, item_id=index + 1)
                result = self.perform(op, auth.read, auth.apply)
                undo = operation("undo-" + str(index), item_id=index + 1, action="undo", undo=key)
                restored = self.perform(undo, auth.read, auth.apply)
                self.assertEqual(restored["status"], "confirmed")
                self.assertEqual(auth.values["rating100"], original)
                self.assertEqual(self.perform(undo, auth.read, auth.apply), restored)
                self.assertEqual(auth.writes, 2)
                with closing(sqlite3.connect(self.db)) as conn:
                    row = conn.execute("SELECT corrects_id FROM rec_events WHERE event_id=?", (restored["event_id"],)).fetchone()
                    self.assertEqual(row[0], result["event_id"])

    def test_new_managed_operation_or_external_change_blocks_undo(self):
        self.perform()
        self.perform(operation("later", rating=20))
        result = self.perform(operation("undo", action="undo", undo="op-1"))
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(self.authority.writes, 2)
        self.authority.values["rating100"] = 60
        result = self.perform(operation("undo-later", action="undo", undo="later"))
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(self.authority.values["rating100"], 60)

    def test_unchanged_rating_has_no_outcome_and_never_calls_mutation(self):
        self.authority.values["rating100"] = 80
        result = self.perform()
        self.assertEqual(result["status"], "confirmed")
        self.assertIsNone(result["event_id"])
        self.assertEqual(self.authority.writes, 0)

    def test_uncertain_increment_cannot_repeat_or_be_proven_by_counter(self):
        def uncertain(key, change):
            self.authority.apply(key, change)
            raise TimeoutError("PRIVATE URL and request payload")
        op = operation(action="engagement")
        result = self.perform(op, apply=uncertain)
        self.assertEqual(result["status"], "indeterminate")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertEqual(self.perform(op), result)
        self.assertEqual(self.authority.writes, 1)
        other = self.perform(operation("other", action="engagement"))
        self.assertEqual(other["reason"], "item_owned")
        reconciled = self.events.reconcile_feedback(self.db, operation_id="op-1", read_current=self.authority.read)
        self.assertEqual(reconciled["status"], "indeterminate")
        self.assertEqual(reconciled["reason"], "counter_confirmation_ambiguous")
        self.assertEqual(self.authority.values["engagement_count"], 1)
        self.assertEqual(self.authority.writes, 1)

    def test_unresolved_increment_retains_ownership(self):
        result = self.perform(operation(action="engagement"), apply=lambda *_: {"status": "indeterminate"})
        self.assertEqual(result["status"], "indeterminate")
        again = self.events.reconcile_feedback(self.db, operation_id="op-1", read_current=self.authority.read)
        self.assertEqual(again["reason"], "counter_confirmation_ambiguous")
        self.assertEqual(self.perform(operation("other"))["reason"], "item_owned")

    def test_secondary_and_primary_items_have_independent_ownership(self):
        self.perform(operation(action="engagement"), apply=lambda *_: {"status": "indeterminate"})
        result = self.perform(operation("image", kind="image"))
        self.assertEqual(result["status"], "confirmed")

    def test_failed_initial_read_never_sends_and_can_be_closed_without_retry(self):
        def fail(_):
            raise RuntimeError("private-query")
        result = self.perform(read=fail)
        self.assertEqual(result["reason"], "read_unavailable")
        self.assertEqual(self.authority.writes, 0)
        reconciled = self.events.reconcile_feedback(self.db, operation_id="op-1", read_current=self.authority.read)
        self.assertEqual(reconciled["reason"], "not_dispatched")
        self.assertEqual(self.perform(), reconciled)

    def test_qualified_reference_and_confirmed_secondary_outcome_are_linked(self):
        with patch.object(self.events.time, "time", return_value=200):
            parent = self.events.record_served(self.db, request=request(), items=[item(kind="image", arm="")])[("image", 1)]
            vid = self.events.record_event(self.db, event=view(parent, kind="image"))
            op = operation(kind="image")
            op.update(request_id="request-page-1", viewed_event_id=vid)
            result = self.perform(op)
            self.events.attribute_outcomes(self.db, through_ts=400, window_s=200, policy_revision="p1")
            evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=400)
            self.assertEqual(evidence["attributions"][0]["outcome_id"], result["event_id"])
            self.assertEqual(evidence["attributions"][0]["viewed_id"], vid)

    def test_real_process_holds_item_ownership_without_database_lock(self):
        ctx = multiprocessing.get_context("spawn")
        ready, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
        child = ctx.Process(target=hold_feedback, args=(self.db, ready, release, results))
        child.start()
        try:
            self.assertTrue(ready.wait(10))
            self.assertEqual(self.perform()["reason"], "item_owned")
            rec = self.events.reconcile_feedback(self.db, operation_id="owner", read_current=self.authority.read)
            self.assertEqual(rec["reason"], "dispatch_in_progress")
            self.assertEqual(self.perform(operation("other-item", item_id=2))["status"], "confirmed")
        finally:
            release.set()
            child.join(10)
            if child.is_alive():
                child.kill()
                child.join()
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(results.get(timeout=5)["status"], "confirmed")

    def test_real_crash_after_external_increment_never_replays(self):
        ctx = multiprocessing.get_context("spawn")
        counter = ctx.Value("i", 0)
        child = ctx.Process(target=crash_feedback, args=(self.db, counter, "sent"))
        child.start()
        child.join(10)
        self.assertEqual(child.exitcode, 17)
        self.assertEqual(counter.value, 1)
        self.assertEqual(self.perform(operation("crashed", action="engagement"))["status"], "sent")
        self.assertEqual(self.authority.writes, 0)
        rec = self.events.reconcile_feedback(self.db, operation_id="crashed", read_current=lambda _: {
            "status": "ok", "rating100": None, "engagement_count": counter.value})
        self.assertEqual(rec["status"], "indeterminate")
        self.assertEqual(rec["reason"], "counter_confirmation_ambiguous")
        self.assertEqual(counter.value, 1)

    def test_real_crash_before_dispatch_can_release_without_external_mutation(self):
        ctx = multiprocessing.get_context("spawn")
        counter = ctx.Value("i", 0)
        child = ctx.Process(target=crash_feedback, args=(self.db, counter, "planned"))
        child.start()
        child.join(10)
        self.assertEqual(child.exitcode, 17)
        rec = self.events.reconcile_feedback(self.db, operation_id="crashed", read_current=self.authority.read)
        self.assertEqual(rec["reason"], "not_dispatched")
        self.assertEqual(counter.value, 0)
        self.assertEqual(self.authority.reads, 0)


if __name__ == "__main__":
    unittest.main()

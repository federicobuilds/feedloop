"""Committed raw-watch capture, immutable replay, and attributable interval contracts."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from test_ledger_contracts import load_events, request, item, view


class WatchEvidenceContracts(unittest.TestCase):
    def setUp(self):
        self.events = load_events()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = str(Path(self.temp.name) / 'ledger.db')
        self.events.initialize_event_store(self.db, cutover_ts=95)
        self.clock = self.enterContext(patch.object(self.events.time, 'time', return_value=110))
        self.parent = self.events.record_served(self.db, request=request(), items=[item()])[('video', 1)]
        self.viewed = self.events.record_event(self.db, event=view(self.parent))
        self.clock.return_value = 200

    def raw(self, event_id='start', event_type='view_start', ts=120, pos=50, previous=None, **changes):
        return dict(id=event_id, type=event_type, occurred_at=ts, position=pos, duration=600,
                    item_id=1, stream_session_id='host-browser', session_id='session-1',
                    canonical_session_id='canonical', viewed_event_id=self.viewed,
                    previous_event_id=previous, playback_rate=1, **changes)

    def batch(self, capture_id, rows, status='committed', reason=None):
        return dict(capture_id=capture_id, source_id='installation', received_at=190,
                    source_revision='source-revision-1',
                    status=status, reason=reason, events_json=rows)

    def import_batch(self, capture_id, rows, **kwargs):
        return self.events.import_watch_capture(self.db, batch=self.batch(capture_id, rows, **kwargs))

    def test_start_progress_pause_are_exact_immutable_intervals_not_lifetime_totals(self):
        self.import_batch('one', [self.raw()])
        result = self.import_batch('two', [self.raw('progress', 'view_progress', 125, 55, 'start'),
                                         self.raw('pause', 'view_pause', 128, 58, 'progress')])
        self.assertEqual(result['outcomes'], 2)
        self.assertEqual(self.events.attribute_outcomes(self.db, through_ts=500, window_s=100, policy_revision='watch-v1'), 2)
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=500)
        outcomes = [e for e in evidence['events'] if e['event_type'] == 'outcome']
        self.assertEqual([e['payload']['watched_s_delta'] for e in outcomes], [5, 3])
        self.assertEqual({a['viewed_id'] for a in evidence['attributions']}, {self.viewed})
        self.assertEqual(self.events.attribute_outcomes(self.db, through_ts=500, window_s=100, policy_revision='watch-v1'), 0)
        self.assertEqual(self.import_batch('two', [self.raw('progress', 'view_progress', 125, 55, 'start'),
                                                 self.raw('pause', 'view_pause', 128, 58, 'progress')])['status'], 'duplicate')
        self.assertEqual(self.events.imported_watch_captures(self.db, source_id='installation'), ['one', 'two'])

    def test_progress_without_start_or_missing_chain_never_earns_reward(self):
        for index, previous in enumerate((None, 'lost-start')):
            result = self.import_batch(str(index), [self.raw('progress-' + str(index), 'view_progress', 125, 55, previous)])
            self.assertEqual(result['outcomes'], 0)
        self.assertEqual(self.events.attribute_outcomes(self.db, through_ts=500, window_s=100, policy_revision='watch-v1'), 0)

    def test_capture_readiness_is_proven_only_by_committed_outcomes_and_counts_attribution(self):
        before = self.events.read_capture_readiness(self.db)
        self.assertEqual((before['status'], before['verified'], before['captured_outcomes'], before['attributed_outcomes']),
                         ('ok', False, 0, 0))
        self.import_batch('only-start', [self.raw()])
        anchored = self.events.read_capture_readiness(self.db)
        self.assertFalse(anchored['verified'], 'an imported batch without an outcome is not capture proof')
        self.assertEqual(anchored['receipts'], {'imported': 1})
        self.import_batch('quarantined', [self.raw('q')], status='quarantined', reason='host_partial_or_rollback')
        self.import_batch('two', [self.raw('progress', 'view_progress', 125, 55, 'start')])
        ready = self.events.read_capture_readiness(self.db)
        self.assertTrue(ready['verified'])
        self.assertEqual((ready['captured_outcomes'], ready['receipts'], ready['attributed_outcomes'], ready['attribution_through_ts']),
                         (1, {'imported': 2, 'quarantined': 1}, 0, None))
        self.assertEqual(ready['last_imported_at'], 200)
        self.events.attribute_outcomes(self.db, through_ts=500, window_s=100, policy_revision='watch-v1')
        after = self.events.read_capture_readiness(self.db)
        self.assertEqual((after['attributed_outcomes'], after['attribution_through_ts']), (1, 500))
        missing = self.events.read_capture_readiness(str(Path(self.temp.name) / 'absent.db'))
        self.assertEqual((missing['status'], missing['verified']), ('unavailable', False))

    def test_first_progress_after_visibility_only_anchors_then_measures_next_interval(self):
        first = self.import_batch('one', [self.raw('anchor', 'view_progress', 125, 55, 'unattributed-start')])
        self.assertEqual(first['outcomes'], 0)
        second = self.import_batch('two', [self.raw('next', 'view_progress', 130, 60, 'anchor')])
        self.assertEqual(second['outcomes'], 1)
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=500)
        watched = [e['payload'] for e in evidence['events'] if e['event_type'] == 'outcome']
        self.assertEqual(watched[0]['started_at'], 125)
        self.assertEqual(watched[0]['watched_s_delta'], 5)

    def test_seek_gap_and_late_event_are_not_watch_seconds(self):
        self.import_batch('one', [self.raw()])
        result = self.import_batch('two', [self.raw('seek', 'view_seek', 121, 500, 'start'),
            self.raw('late', 'view_progress', 119, 49, 'start'),
            self.raw('progress', 'view_progress', 126, 505, 'seek')])
        self.assertEqual(result['outcomes'], 1)
        self.assertEqual(self.events.attribute_outcomes(self.db, through_ts=500, window_s=100, policy_revision='watch-v1'), 1)
        evidence = self.events.read_evidence(self.db, since_ts=95, through_ts=500)
        self.assertEqual([e['payload']['watched_s_delta'] for e in evidence['events'] if e['event_type'] == 'outcome'], [5])

    def test_rejected_receipt_future_clock_and_wrong_view_quarantine(self):
        rows = [self.raw()]
        self.assertEqual(self.import_batch('failed', rows, status='quarantined', reason='host_partial_or_rollback')['outcomes'], 0)
        for index, changes in enumerate(({'occurred_at': 300}, {'viewed_event_id': 'missing'}, {'session_id': 'other'})):
            row = {**self.raw(event_id=str(index)), **changes}
            self.assertEqual(self.import_batch('bad-' + str(index), [row])['outcomes'], 0)

    def test_same_predecessor_cannot_pay_two_branches(self):
        self.import_batch('one', [self.raw()])
        result = self.import_batch('two', [self.raw('a', 'view_progress', 125, 55, 'start'),
                                          self.raw('b', 'view_progress', 126, 56, 'start')])
        self.assertEqual(result['outcomes'], 1)
        self.assertEqual(result['quarantined'], 1)

    def test_late_start_cannot_recount_an_already_credited_interval(self):
        self.import_batch('one', [self.raw(), self.raw('progress', 'view_progress', 125, 55, 'start')])
        result = self.import_batch('late', [self.raw('late-start', 'view_start', 122, 52),
                                          self.raw('late-progress', 'view_progress', 127, 57, 'late-start')])
        self.assertEqual(result['outcomes'], 0)
        self.assertEqual(result['quarantined'], 1)

    def test_source_restart_cannot_claim_the_same_source_event_again(self):
        rows = [self.raw(), self.raw('progress', 'view_progress', 125, 55, 'start')]
        self.import_batch('one', rows)
        batch = {**self.batch('two', rows), 'source_id': 'reconfigured-source'}
        self.assertEqual(self.events.import_watch_capture(self.db, batch=batch)['outcomes'], 0)

    def test_source_receipt_steps_and_bindings_are_append_only(self):
        self.import_batch('one', [self.raw()])
        with closing(sqlite3.connect(self.db)) as conn:
            for table in ('rec_watch_capture_imports', 'rec_watch_steps', 'rec_watch_view_bindings'):
                with self.subTest(table=table), self.assertRaises(sqlite3.IntegrityError):
                    conn.execute('DELETE FROM ' + table)
                conn.rollback()



if __name__ == '__main__':
    unittest.main()

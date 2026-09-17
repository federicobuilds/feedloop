"""The autonomous tick must use an actual completed attribution boundary."""
from types import SimpleNamespace
from unittest.mock import Mock

from feedloop import ledger
from feedloop.tuning import Tuner
from test_ledger_contracts import request, item, view, outcome


def test_tick_uses_completed_run_when_clock_falls_between_attribution_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "time", SimpleNamespace(time=lambda: 200.0))
    db = str(tmp_path / "events.sqlite")
    ledger.initialize_event_store(db, cutover_ts=95.0)
    ledger.record_session_mapping(db, alias_session_id="session-1", canonical_session_id="session-1", mapping_revision="initial")
    served = ledger.record_served(db, request=request(), items=[item()])[("video", 1)]
    viewed = ledger.record_event(db, event=view(served))
    ledger.record_event(db, event=outcome(parent=viewed))
    ledger.advance_attribution(db, now=5000.0)
    assert "attribution_pending" in ledger.read_evidence(db, since_ts=95, through_ts=8400)["validity_reasons"]
    reader = Mock(wraps=ledger.read_evidence)
    tuner = Tuner(str(tmp_path / "tuner.sqlite"), ledger_path=db, clock=lambda: 95.0, read_evidence=reader,
                  verdict_fn=lambda *a, **k: (True, False, 1), reward_fn=lambda *a, **k: .5,
                  cumulative_facts=lambda items, cutoff: {"cutoff_ts": cutoff, "items": {
                      ("video", 1): {"watched_s": 300., "duration_s": 600., "rating": None, "engagement_count": 0}}})
    tuner.initialize()
    tuner.snapshot = lambda: (("embedding_weight", 0, 95), {})
    tuner.arms = lambda knob=None: (.35, .4, 95)
    result = tuner.tick(now=30000)
    assert result == {"action": "wait", "detail": {"counts": {"base": 1, "cand": 0}, "sessions": {"base": 1, "cand": 0}}}
    assert reader.call_args.kwargs["through_ts"] == 5000.0

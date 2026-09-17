"""Synthetic tuner contracts: reads never seed, decisions gate on volume, sessions,
significance and entropy, promotion is one ledgered revertible step, stall and
rotation are transactional, obsolete evidence is discarded, undo rejects stale moves."""
import json
import os
import random
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from feedloop import ledger, tuning
from feedloop.tuning import Tuner, decide


def make_tuner(tmp_path, **overrides):
    clock = overrides.pop("clock", None) or (lambda: 2000000.0)
    facts = overrides.pop("cumulative_facts", lambda items, cutoff: {"cutoff_ts": cutoff, "items": {}})
    return Tuner(str(tmp_path / "tuner.sqlite"), ledger_path=str(tmp_path / "events.sqlite"), cumulative_facts=facts,
                 clock=clock, **overrides)


def rows(tuner, sql):
    with sqlite3.connect(tuner.path) as conn:
        return conn.execute(sql).fetchall()


def decision_summary(base_reward, cand_reward, *, sessions=20, per_session=4, base_categories=("acts",),
                     cand_categories=("acts", "other", "bodyparts")):
    trials, grouped = [], {}
    for s in range(sessions):
        for i in range(per_session):
            for arm, reward in (("base", base_reward(s, i)), ("cand", cand_reward(s, i))):
                pool = base_categories if arm == "base" else cand_categories
                trial = {"arm": arm, "reward": reward, "category": pool[(s + i) % len(pool)],
                         "viewed_id": f"{arm}-{s}-{i}", "kind": "video", "item_id": s * 10 + i}
                trials.append(trial)
                grouped.setdefault(f"session-{s}", []).append(trial)
    return {"status": "ok", "valid": True, "validity_reasons": [], "trials": trials, "sessions": grouped}


def test_tuner_snapshot_reads_without_schema_or_seeding(tmp_path):
    tuner = make_tuner(tmp_path)
    assert tuner.snapshot() == (None, {})
    assert not os.path.exists(tuner.path)
    tuner.initialize()
    assert os.path.exists(tuner.path)
    state, values = tuner.snapshot()
    assert state[0] == "embedding_weight" and values == {}


def test_missing_tuner_arm_is_not_seeded_on_read(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    assert tuner.arms("taste_mix_weight") is None
    assert rows(tuner, "SELECT knob FROM arms") == [("embedding_weight",)]
    assert tuner.arms("embedding_weight")[:2] == (0.35, 0.4)


def test_missing_store_remains_missing_on_snapshot_read(tmp_path):
    tuner = make_tuner(tmp_path)
    assert tuner.snapshot() == (None, {}) and tuner.arms() is None
    assert not os.path.exists(tuner.path)


def test_automatic_promotion_is_disabled_before_reading_evidence(tmp_path):
    tuner = make_tuner(tmp_path, read_evidence=Mock(side_effect=AssertionError("read")))
    assert tuner.evaluate()["status"] == "disabled"
    with pytest.raises(ValueError):
        tuner.evaluate(since_ts=1)
    assert not os.path.exists(tuner.path)


def test_explicit_evaluation_delegates_attributed_increments_and_shared_math(tmp_path):
    evidence = {"status": "ok", "events": [], "attributions": [], "sessions": {}}
    projected = {"status": "ok", "valid": False, "validity_reasons": ["no_eligible_trials"], "promotion_enabled": False, "trials": []}
    verdict, reward = object(), object()
    reader, summarizer = Mock(return_value=evidence), Mock(return_value=projected)
    tuner = make_tuner(tmp_path, read_evidence=reader, summarize_trials=summarizer, verdict_fn=verdict, reward_fn=reward)
    cumulative = {"cutoff_ts": 200, "items": {}}
    result = tuner.evaluate(since_ts=100, through_ts=200, experiment_id="trial-1", cumulative_at_cutoff=cumulative)
    reader.assert_called_once_with(tuner.ledger_path, since_ts=100, through_ts=200, experiment_id="trial-1")
    summarizer.assert_called_once_with(evidence, verdict=verdict, trial_reward=reward, cumulative_at_cutoff=cumulative)
    assert result == projected
    assert not os.path.exists(tuner.path)


def test_decision_waits_stalls_promotes_and_vetoes_on_entropy():
    rng = random.Random(7)
    noisy = lambda base: (lambda s, i: max(0.0, min(1.0, base + rng.uniform(-0.05, 0.05))))
    assert decide(decision_summary(noisy(.2), noisy(.5), sessions=10, per_session=2), now=2000000.0, window_started=1990000.0)[0] == "wait"
    assert decide(decision_summary(noisy(.2), noisy(.5), sessions=4, per_session=20), now=2000000.0, window_started=1990000.0)[0] == "wait"
    summary = decision_summary(noisy(.2), noisy(.5))
    summary.update(valid=False, validity_reasons=["cumulative_verdict_unavailable"])
    assert decide(summary, now=2000000.0, window_started=1990000.0)[0] == "wait"
    assert decide(decision_summary(noisy(.2), noisy(.5)), now=2000000.0, window_started=2000000.0 - 31 * 86400.0)[0] == "rotate"
    assert decide(decision_summary(noisy(.35), noisy(.35)), now=2000000.0, window_started=1990000.0)[0] == "stall"
    action, detail = decide(decision_summary(noisy(.2), noisy(.5)), now=2000000.0, window_started=1990000.0)
    assert (action, detail["winner"]) == ("promote", "cand")
    assert abs(detail["z"]) >= tuning.TUNER_Z
    action, detail = decide(decision_summary(noisy(.2), noisy(.5), base_categories=("acts", "other", "bodyparts"),
                                             cand_categories=("acts",)), now=2000000.0, window_started=1990000.0)
    assert (action, detail.get("reason")) == ("stall", "entropy_veto")
    assert detail["winner_blocked"] == "cand"


def test_promotion_adopts_winner_ledgers_the_step_and_rotates(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    revision = tuner.revision[0]
    assert tuner.promote({"winner": "cand", "z": 2.4}, expected_revision=revision) is True
    assert rows(tuner, "SELECT knob, value FROM tuned") == [("embedding_weight", 0.4)]
    applied, rotated = rows(tuner, "SELECT knob, old_value, new_value, evidence, status FROM ledger ORDER BY id")
    assert applied[:3] == ("embedding_weight", 0.35, 0.4) and json.loads(applied[3])["winner"] == "cand" and applied[4] == "applied"
    assert rotated[4] == "rotated" and json.loads(rotated[3]) == {"reason": "promoted", "next": "contributor_affinity_weight"}
    assert rows(tuner, "SELECT active_knob, stalls FROM experiment") == [("contributor_affinity_weight", 0)]
    # the incoming knob's arms are deleted until explicit initialization
    assert rows(tuner, "SELECT knob FROM arms") == [("embedding_weight",)]
    assert tuner.arms("contributor_affinity_weight") is None
    # an obsolete revision is discarded without touching the store
    before = rows(tuner, "SELECT * FROM ledger")
    assert tuner.promote({"winner": "cand"}, expected_revision=0) is False
    assert rows(tuner, "SELECT * FROM ledger") == before
    # a store with no experiment row promotes nothing
    empty = make_tuner(tmp_path / "empty")
    os.makedirs(tmp_path / "empty")
    assert empty.promote({"winner": "cand"}) is False


def test_tick_is_ripened_gated_and_routes_each_decision(tmp_path, monkeypatch):
    for action, target in (("promote", "promote"), ("stall", "stall"), ("rotate", "rotate")):
        evidence = {"status": "ok", "through_ts": 2000000.0 - 6 * 3600.0, "viewed_ids": ["v1"],
                    "events": [{"event_id": "v1", "kind": "video", "item_id": 4}], "attribution_run": None}
        facts = Mock(return_value={"cutoff_ts": 1})
        tuner = make_tuner(tmp_path / action, clock=lambda: 1900000.0, read_evidence=Mock(return_value=evidence),
                           summarize_trials=Mock(return_value={"status": "ok", "valid": True}), cumulative_facts=facts)
        tuner.initialize()
        monkeypatch.setattr(tuning, "decide", Mock(return_value=(action, {"winner": "cand", "reason": "window_aged_out"})))
        calls = {}
        for name in ("promote", "stall", "rotate"):
            calls[name] = Mock()
            monkeypatch.setattr(tuner, name, calls[name])
        tuner.clock = lambda: 2000000.0
        result = tuner.tick()
        assert result["action"] == action
        tuner.read_evidence.assert_called_once_with(tuner.ledger_path, since_ts=1900000.0, through_ts=2000000.0 - tuning.TUNER_RIPEN_S)
        facts.assert_called_once_with({("video", 4)}, evidence["through_ts"])
        for name in calls:
            assert calls[name].called == (name == target), name
        if action == "promote":
            assert calls["promote"].call_args.kwargs.get("expected_revision") == tuner.revision[0]
    # an unripe window waits and reads nothing; no experiment is disabled
    tuner = make_tuner(tmp_path / "unripe", clock=lambda: 2000000.0, read_evidence=Mock(side_effect=AssertionError("read")))
    tuner.initialize()
    assert tuner.tick() == {"action": "wait", "reason": "window_unripe"}
    idle = make_tuner(tmp_path / "idle", read_evidence=Mock(side_effect=AssertionError("read")))
    assert idle.tick() == {"action": "disabled"}


def test_stall_counts_restarts_window_and_rotates_after_four(tmp_path):
    ticks = [2000000.0]
    tuner = make_tuner(tmp_path, clock=lambda: ticks[0])
    tuner.initialize()
    for n in range(1, 4):
        ticks[0] += 100
        tuner.stall(json.dumps({"z": 0.1}), expected_revision=tuner.revision[0])
        assert rows(tuner, "SELECT stalls FROM experiment") == [(n,)]
        assert rows(tuner, "SELECT since_ts FROM arms WHERE knob='embedding_weight'") == [(ticks[0],)]
    ticks[0] += 100
    tuner.stall(json.dumps({"z": 0.1}), expected_revision=tuner.revision[0])
    assert rows(tuner, "SELECT active_knob, stalls FROM experiment") == [("contributor_affinity_weight", 0)]
    statuses = [r[0] for r in rows(tuner, "SELECT status FROM ledger ORDER BY id")]
    assert statuses == ["no_verdict"] * 4 + ["rotated"]
    # a stall with obsolete evidence is discarded
    before = rows(tuner, "SELECT * FROM ledger")
    tuner.stall("{}", expected_revision=0)
    assert rows(tuner, "SELECT * FROM ledger") == before


def test_initialization_is_explicit_and_uses_accepted_transaction(tmp_path):
    tuner = make_tuner(tmp_path)
    assert not os.path.exists(tuner.path)
    with tuner.transaction() as conn:
        conn.execute("INSERT INTO tuned VALUES ('embedding_weight', 0.4, 1.0)")
    tuner.initialize()
    assert rows(tuner, "SELECT knob, base, candidate FROM arms") == [("embedding_weight", 0.4, 0.45)]
    # idempotent: a second initialization changes nothing
    revision = tuner.revision[0]
    tuner.initialize()
    assert rows(tuner, "SELECT knob, base, candidate FROM arms") == [("embedding_weight", 0.4, 0.45)]
    assert tuner.revision[0] == revision


def test_import_does_not_start_maintenance_thread():
    import ast, inspect
    tree = ast.parse(inspect.getsource(tuning))
    assert not any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == "start" for node in tree.body)


def test_serving_schedules_the_autonomous_tick(tmp_path):
    import ast, inspect
    from feedloop import engine
    tree = ast.parse(inspect.getsource(engine))
    klass = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Engine")
    feed = next(m for m in klass.body if isinstance(m, ast.FunctionDef) and m.name == "feed")
    calls = {node.func.attr for node in ast.walk(feed) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "maybe_tick" in calls
    # the opt-out disables scheduling without changing the default
    tuner = make_tuner(tmp_path, automatic=False)
    assert tuner.maybe_tick() is None
    assert Tuner(str(tmp_path / "t2.sqlite"), ledger_path="x", cumulative_facts=None).automatic is True


def test_revert_each_actual_knob_and_invalidate_caches(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    for knob in ("embedding_weight", "contributor_affinity_weight", "taste_audio_weight", "taste_mix_weight"):
        with tuner.transaction() as conn:
            conn.execute("UPDATE experiment SET active_knob=? WHERE id=1", (knob,))
            conn.execute("INSERT OR REPLACE INTO arms VALUES (?,?,?,?)", (knob, tuning.knob_default(knob), tuning.knob_default(knob) + 0.05, 1.0))
        assert tuner.promote({"winner": "cand"}) is True
        row = rows(tuner, "SELECT id, knob, old_value, new_value FROM ledger WHERE status='applied' ORDER BY id DESC LIMIT 1")[0]
        revision = tuner.revision[0]
        tuner.snapshot()
        assert tuner.tuned_cache.value is not None
        result = tuner.write("revert", row[0])
        assert result == {"ok": True, "action": "revert", "restored": {knob: row[2]}}
        assert rows(tuner, "SELECT value FROM tuned WHERE knob=?".replace("?", repr(knob))) == [(row[2],)]
        assert rows(tuner, f"SELECT status FROM ledger WHERE id={row[0]}") == [("reverted",)]
        assert tuner.revision[0] > revision and tuner.tuned_cache.value is None


def test_reset_all_settled_values_and_reject_old_undo(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    assert tuner.promote({"winner": "cand"}) is True
    applied = rows(tuner, "SELECT id FROM ledger WHERE status='applied'")[0][0]
    result = tuner.write("reset")
    assert result["ok"] and result["restored"] == dict(tuning.KNOB_DEFAULTS)
    assert dict(rows(tuner, "SELECT knob, value FROM tuned")) == dict(tuning.KNOB_DEFAULTS)
    assert rows(tuner, f"SELECT status FROM ledger WHERE id={applied}") == [("superseded_by_reset",)]
    assert tuner.write("revert", applied) == {"ok": False, "error": "ledger row not found or not an applied move"}
    assert tuner.write("bogus") == {"ok": False, "error": "unknown action"}


def test_reject_stale_settled_value_without_writes(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    assert tuner.promote({"winner": "cand"}) is True
    applied = rows(tuner, "SELECT id FROM ledger WHERE status='applied'")[0][0]
    with tuner.transaction() as conn:
        conn.execute("UPDATE tuned SET value=0.55 WHERE knob='embedding_weight'")
    before = rows(tuner, "SELECT * FROM ledger"), rows(tuner, "SELECT * FROM tuned")
    assert tuner.write("revert", applied) == {"ok": False, "error": "knob changed since this move; undo rejected"}
    assert (rows(tuner, "SELECT * FROM ledger"), rows(tuner, "SELECT * FROM tuned")) == before


def test_write_failure_rolls_back_all_knobs(tmp_path, monkeypatch):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    monkeypatch.setitem(tuning.KNOB_DEFAULTS, "taste_mix_weight", 9.0)
    before = rows(tuner, "SELECT * FROM tuned"), rows(tuner, "SELECT * FROM ledger"), rows(tuner, "SELECT * FROM arms")
    revision = tuner.revision[0]
    assert tuner.write("reset") == {"ok": False, "error": "tuner change failed; no changes committed"}
    assert (rows(tuner, "SELECT * FROM tuned"), rows(tuner, "SELECT * FROM ledger"), rows(tuner, "SELECT * FROM arms")) == before
    assert tuner.revision[0] == revision


def test_autonomous_promotion_is_revertible_through_the_existing_control(tmp_path):
    rng = random.Random(3)
    noisy = lambda base: (lambda s, i: max(0.0, min(1.0, base + rng.uniform(-0.05, 0.05))))
    summary = decision_summary(noisy(.2), noisy(.5))
    evidence = {"status": "ok", "through_ts": 2000000.0 - 6 * 3600.0, "viewed_ids": [], "events": [], "attribution_run": None}
    tuner = make_tuner(tmp_path, clock=lambda: 1900000.0, read_evidence=Mock(return_value=evidence), summarize_trials=Mock(return_value=summary))
    tuner.initialize()
    tuner.clock = lambda: 2000000.0
    assert tuner.tick()["action"] == "promote"
    assert rows(tuner, "SELECT knob, value FROM tuned") == [("embedding_weight", 0.4)]
    move = rows(tuner, "SELECT id FROM ledger WHERE status='applied'")[0][0]
    assert tuner.write("revert", move)["ok"]
    assert rows(tuner, "SELECT knob, value FROM tuned") == [("embedding_weight", 0.35)]


def test_inflight_builder_cannot_recache_pre_reset_data(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    generation = tuner.tuned_cache.generation
    tuner.write("reset")
    tuner.tuned_cache.set(("stale", {}), generation=generation)
    assert tuner.tuned_cache.value is None
    assert tuner.snapshot()[0][0] == "embedding_weight"


def test_stale_evaluator_cannot_apply_after_reset(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    revision = tuner.revision[0]
    tuner.write("reset")
    assert tuner.promote({"winner": "cand"}, expected_revision=revision) is False
    tuner.stall("{}", expected_revision=revision)
    tuner.rotate("stalled", expected_revision=revision)
    assert [r[0] for r in rows(tuner, "SELECT status FROM ledger ORDER BY id")] == ["reset_to_standard"] * 4


def test_legacy_trial_volume_cannot_enable_promotion(tmp_path):
    summary = {"status": "ok", "valid": True, "validity_reasons": [], "trials": [], "sessions": {}}
    assert decide(summary, now=2000000.0, window_started=1990000.0) == ("wait", {"counts": {"base": 0, "cand": 0}, "sessions": {"base": 0, "cand": 0}})
    tuner = make_tuner(tmp_path, summarize_trials=Mock(return_value=summary), read_evidence=Mock(return_value={"status": "ok", "through_ts": 1}))
    assert tuner.evaluate(since_ts=0, through_ts=1)["promotion_enabled"] is False


def test_all_evaluator_decisions_reject_obsolete_evidence(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    stale = tuner.revision[0] - 1
    before = rows(tuner, "SELECT * FROM ledger"), rows(tuner, "SELECT * FROM tuned"), rows(tuner, "SELECT * FROM experiment")
    assert tuner.promote({"winner": "cand"}, expected_revision=stale) is False
    tuner.stall("{}", expected_revision=stale)
    tuner.rotate("aged", expected_revision=stale)
    assert (rows(tuner, "SELECT * FROM ledger"), rows(tuner, "SELECT * FROM tuned"), rows(tuner, "SELECT * FROM experiment")) == before


def test_rotation_chained_from_stall_cannot_follow_control(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    with tuner.transaction() as conn:
        conn.execute("UPDATE experiment SET stalls=3 WHERE id=1")
    original = tuner.rotate

    def hijacked(reason, expected_revision=None):
        tuner.write("reset")   # a control lands between the stall commit and its chained rotation
        return original(reason, expected_revision=expected_revision)
    tuner.rotate = hijacked
    tuner.stall("{}", expected_revision=tuner.revision[0])
    assert rows(tuner, "SELECT active_knob FROM experiment") == [("embedding_weight",)]
    assert "rotated" not in [r[0] for r in rows(tuner, "SELECT status FROM ledger")]


# --- interleavings: a paused read or publication, a concurrent control, neither crosses ---

def primed_tuner(tmp_path):
    """Every registry knob tuned one step above default with an applied ledger move (ids 1..N),
    arms one step further, the first knob active with three stalls: the pinned pre-promotion state."""
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    with tuner.transaction() as conn:
        for entry in tuner.registry:
            knob, default = entry["knob"], tuning.knob_default(entry["knob"])
            value = round(default + entry["step"], 6)
            conn.execute("INSERT INTO tuned VALUES (?,?,1) ON CONFLICT(knob) DO UPDATE SET value=excluded.value, updated_ts=1", (knob, value))
            conn.execute("INSERT OR REPLACE INTO arms VALUES (?,?,?,1)", (knob, value, round(value + entry["step"], 6)))
            conn.execute("INSERT INTO ledger (ts, knob, old_value, new_value, evidence, status) VALUES (1,?,?,?,'{}','applied')",
                         (knob, default, value))
        conn.execute("UPDATE experiment SET active_knob=?, stalls=3, started_ts=1", (tuner.registry[0]["knob"],))
    return tuner


CONTRIBUTOR = "contributor_affinity_weight"
CONTROLS = [("reset", None), ("revert", 2)]      # ledger id 2 is the contributor knob's applied move


def contributor_value(tuner, name):
    if name == "snapshot":
        return tuner.snapshot()[1][CONTRIBUTOR]
    return tuner.arms(CONTRIBUTOR)[0]


class Interleaving:
    def __init__(self):
        self.errors = []

    def start(self, fn):
        def run():
            try:
                fn()
            except BaseException as error:
                self.errors.append(error)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def finish(self, thread):
        thread.join(3)
        assert not thread.is_alive(), "tuner deadlock"
        assert self.errors == []


def pause_read(prefix):
    """A connection factory wrapper that pauses once at the first statement starting with prefix."""
    read, resume = threading.Event(), threading.Event()
    armed = [True]

    class PausedConnection:
        def __init__(self, conn):
            self.conn = conn

        def execute(inner, sql, *args):
            cursor = inner.conn.execute(sql, *args)
            if sql.startswith(prefix) and armed[0]:
                armed[0] = False
                fetched = cursor.fetchall()
                read.set()
                assert resume.wait(3), "paused read was not released"
                return fetched if sql == "SELECT knob, value FROM tuned" else SimpleNamespace(
                    fetchone=lambda: fetched[0] if fetched else None, fetchall=lambda: fetched, __iter__=lambda: iter(fetched))
            return cursor

        def __getattr__(inner, name):
            return getattr(inner.conn, name)

    return read, resume, PausedConnection


@pytest.mark.parametrize("action,ledger_id", CONTROLS)
@pytest.mark.parametrize("name,prefix", [("snapshot", "SELECT knob, value FROM tuned"), ("arms", "SELECT base, candidate, since_ts")])
def test_snapshot_and_arms_reads_cannot_recache_after_control(tmp_path, name, prefix, action, ledger_id):
    tuner = primed_tuner(tmp_path)
    race = Interleaving()
    read, resume, Paused = pause_read(prefix)
    original = tuner._read_only
    tuner._read_only = lambda: Paused(original())
    worker = race.start(lambda: contributor_value(tuner, name))
    assert read.wait(3)
    assert tuner.write(action, ledger_id)["ok"]      # a control commits while the read is in flight
    resume.set()
    race.finish(worker)
    tuner._read_only = original
    assert contributor_value(tuner, name) == .5, "the pre-control read must not have been published"


@pytest.mark.parametrize("action,ledger_id", CONTROLS)
@pytest.mark.parametrize("name", ["stall", "rotate"])
def test_stall_and_rotate_serialize_their_read_through_commit(tmp_path, name, action, ledger_id):
    """The decision's revision check and its BEGIN IMMEDIATE are one critical section: a control
    that arrives between them must wait, or it would commit under a revision the decision already
    validated. The pause sits after the check, before the connection opens, so SQLite holds nothing yet."""
    tuner = primed_tuner(tmp_path)
    race = Interleaving()
    paused, resume = threading.Event(), threading.Event()
    original, armed = tuner.db, [True]

    def pausing_db():
        if armed[0]:
            armed[0] = False
            paused.set()
            assert resume.wait(3), "paused decision was not released"
        return original()
    tuner.db = pausing_db
    expected = tuner.revision[0]
    worker = race.start(lambda: getattr(tuner, name)("{}", expected_revision=expected))
    assert paused.wait(3)
    attempted, done = threading.Event(), threading.Event()

    def control():
        attempted.set()
        assert tuner.write(action, ledger_id)["ok"]
        done.set()
    controller = race.start(control)
    assert attempted.wait(3)
    assert not done.wait(.05), "control crossed the revision check of a live decision"
    resume.set()
    race.finish(worker)
    race.finish(controller)
    tuner.db = original
    statuses = [r[0] for r in rows(tuner, "SELECT status FROM ledger ORDER BY id")]
    control_rows = ["reset_to_standard"] * len(tuner.registry) if action == "reset" else ["revert"]   # reset writes one row per knob
    experiment = rows(tuner, "SELECT active_knob, stalls FROM experiment")
    if name == "rotate":
        assert statuses[-1 - len(control_rows):] == ["rotated"] + control_rows, "the decision commits first, then the control"
        assert experiment == [(CONTRIBUTOR, 0)]
    else:
        # the fourth stall commits first (the blocking assertion above), then releases the lock before its
        # chained rotation; either the rotation lands before the control, or the control lands first and
        # the now-stale rotation is correctly rejected. Both are legal; the stall itself never crosses.
        rotated_first = statuses[-2 - len(control_rows):] == ["no_verdict", "rotated"] + control_rows
        control_first = statuses[-1 - len(control_rows):] == ["no_verdict"] + control_rows and "rotated" not in statuses
        assert rotated_first or control_first, statuses
        if rotated_first:
            assert experiment == [(CONTRIBUTOR, 0)]
        else:
            assert experiment == [("embedding_weight", 0 if action == "reset" else 4)]


@pytest.mark.parametrize("action,ledger_id", CONTROLS)
@pytest.mark.parametrize("name,cache_name", [("snapshot", "tuned_cache"), ("arms", "arms_cache")])
def test_cache_check_and_store_share_control_lock(tmp_path, name, cache_name, action, ledger_id):
    tuner = primed_tuner(tmp_path)
    race = Interleaving()
    read, resume, attempted, done = (threading.Event() for _ in range(4))
    cache = getattr(tuner, cache_name)
    real_set = cache.set

    def paused_set(*args, **kwargs):
        read.set()
        assert resume.wait(3)
        return real_set(*args, **kwargs)
    cache.set = paused_set
    worker = race.start(lambda: contributor_value(tuner, name))
    assert read.wait(3)

    def control():
        attempted.set()
        assert tuner.write(action, ledger_id)["ok"]
        done.set()
    controller = race.start(control)
    assert attempted.wait(3)
    assert not done.wait(.05), "control slipped between cache check and store"
    resume.set()
    race.finish(worker)
    race.finish(controller)
    cache.set = real_set
    assert contributor_value(tuner, name) == .5


def test_tuner_mutation_failure_rolls_back_and_preserves_revision(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    revision = tuner.revision[0]
    before = rows(tuner, "SELECT * FROM ledger")
    with pytest.raises(sqlite3.OperationalError):
        with tuner.transaction() as conn:
            conn.execute("INSERT INTO ledger (ts, knob, status) VALUES (1, 'embedding_weight', 'applied')")
            conn.execute("INSERT INTO nowhere VALUES (1)")
    assert rows(tuner, "SELECT * FROM ledger") == before and tuner.revision[0] == revision


def test_current_stall_and_rotation_helpers_remain_transactional(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    revision = tuner.revision[0]
    tuner.stall("{}", expected_revision=revision)
    assert tuner.revision[0] == revision + 1
    tuner.rotate("manual", expected_revision=tuner.revision[0])
    assert tuner.revision[0] == revision + 2
    assert rows(tuner, "SELECT active_knob FROM experiment") == [("contributor_affinity_weight",)]


def test_concurrent_evaluators_cannot_apply_the_same_evidence_twice(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    revision = tuner.revision[0]
    results = []
    threads = [threading.Thread(target=lambda: results.append(tuner.promote({"winner": "cand"}, expected_revision=revision))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert sorted(results) == [False, False, False, True]
    assert rows(tuner, "SELECT count(*) FROM ledger WHERE status='applied'") == [(1,)]


def test_concurrent_missing_reads_never_initialize_state(tmp_path):
    tuner = make_tuner(tmp_path)
    results = []
    threads = [threading.Thread(target=lambda: results.append((tuner.snapshot(), tuner.arms("embedding_weight")))) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert results == [((None, {}), None)] * 6
    assert not os.path.exists(tuner.path)


def test_real_serving_respects_present_defaults_zero_and_absence(tmp_path):
    tuner = make_tuner(tmp_path)
    tuner.initialize()
    with tuner.transaction() as conn:
        conn.execute("INSERT INTO tuned VALUES ('taste_mix_weight', 0.3, 1.0)")
    absent = tuner.resolve_knobs({})
    assert absent["values"]["embedding_weight"] == 0.35 and absent["values"]["taste_mix_weight"] == 0.3
    assert absent["experiment"]["knob"] == "embedding_weight" and absent["experiment"]["candidate"] == 0.4
    pinned_default = tuner.resolve_knobs({"embedding_weight": 0.35})
    assert pinned_default["experiment"] is None and pinned_default["values"]["embedding_weight"] == 0.35
    zero = tuner.resolve_knobs({"embedding_weight": 0.0, "taste_mix_weight": 0.0})
    assert zero["experiment"] is None and zero["values"]["embedding_weight"] == 0.0 and zero["values"]["taste_mix_weight"] == 0.0
    other = tuner.resolve_knobs({"taste_audio_weight": 0.0})
    assert other["experiment"]["knob"] == "embedding_weight" and other["values"]["taste_audio_weight"] == 0.0


def test_cumulative_facts_come_from_owning_engines_at_the_cutoff(tmp_path):
    from feedloop.engine import Engine
    from fl2_helpers import MemoryCatalog, MemorySignals, MemorySpaces, catalog_row
    catalog = MemoryCatalog([catalog_row("video", 4, duration=600.0), catalog_row("video", 5, duration=900.0), catalog_row("image", 9, duration=0.0)])
    signals = MemorySignals({("video", 4): {"rating": 80.0, "engagement_count": 2,
                                            "watch": {"watched_s": 300.0, "last_at": 1000.0, "visit_days": [0, 1]}},
                             ("image", 9): {"rating": 90.0, "engagement_count": 1}})
    eng = Engine(catalog=catalog, signals=signals, spaces=MemorySpaces({}), ledger_path=str(tmp_path / "e.sqlite"),
                 tuner_path=str(tmp_path / "t.sqlite"))
    facts = eng.cumulative_facts({("video", 4), ("image", 9), ("video", 5)}, 1234.0)
    assert facts["cutoff_ts"] == 1234.0
    assert facts["items"][("video", 4)] == {"watched_s": 300.0, "duration_s": 600.0, "rating": 80.0, "engagement_count": 2}
    assert facts["items"][("video", 5)] == {"watched_s": 0.0, "duration_s": 900.0, "rating": None, "engagement_count": 0}
    assert facts["items"][("image", 9)] == {"watched_s": 0.0, "duration_s": 0.0, "rating": 90.0, "engagement_count": 1}
    assert eng.tuner.cumulative_facts == eng.cumulative_facts
    assert not os.path.exists(eng.ledger_path) and not os.path.exists(eng.tuner_path)

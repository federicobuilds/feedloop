"""The self-tuner: one knob at a time, evidence-gated, every move ledgered and
reversible. Configuration and rollback state live in the tuner store, which is
separate from the event ledger and never holds evidence.

Reads never create or seed the store; ``initialize`` is the explicit
maintenance entry. Every write runs in one transaction that is discarded when
its evidence revision is obsolete, rolls back on failure and, on commit,
advances the in-process revision under the same lock as cache publication.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
import sqlite3
import threading
import time
from typing import Any, Callable, Mapping

from feedloop import ledger
from feedloop.profiles import TTLCache, trial_reward
from feedloop.taste import verdict

TUNER_MIN_TRIALS = 60
TUNER_MIN_SESSIONS = 8
TUNER_RIPEN_S = 6 * 3600.0
TUNER_EVAL_EVERY_S = 6 * 3600.0
TUNER_Z = 1.645
TUNER_ENTROPY_VETO = 0.85
TUNER_STALL_EVALS = 4
TUNER_WINDOW_MAX_S = 30 * 86400.0
TUNER_REGISTRY = (
    {"knob": "embedding_weight", "min": 0.10, "max": 0.60, "step": 0.05},
    {"knob": "contributor_affinity_weight", "min": 0.0, "max": 1.5, "step": 0.1},
    {"knob": "taste_audio_weight", "min": 0.0, "max": 0.40, "step": 0.05},
    {"knob": "taste_mix_weight", "min": 0.0, "max": 0.40, "step": 0.05},
)
KNOB_DEFAULTS = {"embedding_weight": 0.35, "contributor_affinity_weight": 0.5,
                 "taste_audio_weight": 0.15, "taste_mix_weight": 0.10}
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS arms (knob TEXT PRIMARY KEY, base REAL, candidate REAL, since_ts REAL)",
    """CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, knob TEXT,
       old_value REAL, new_value REAL, evidence TEXT, status TEXT)""",
    "CREATE TABLE IF NOT EXISTS tuned (knob TEXT PRIMARY KEY, value REAL, updated_ts REAL)",
    """CREATE TABLE IF NOT EXISTS experiment (id INTEGER PRIMARY KEY CHECK (id = 1),
       active_knob TEXT, stalls INTEGER, started_ts REAL)""",
)


def registry_entry(knob, registry=TUNER_REGISTRY):
    return next((e for e in registry if e["knob"] == knob), None)


def knob_default(knob):
    return KNOB_DEFAULTS[knob]


def session_means(summary):
    """One reward mean per canonical session per arm, the unit of the z test."""
    means = {"base": [], "cand": []}
    for trials in summary["sessions"].values():
        per = {"base": [], "cand": []}
        for trial in trials:
            per[trial["arm"]].append(trial["reward"])
        for arm, rewards in per.items():
            if rewards:
                means[arm].append(sum(rewards) / len(rewards))
    return means


def entropy(trials):
    """Shannon entropy of the category mix an arm actually served."""
    counts = {}
    for trial in trials:
        counts[trial["category"]] = counts.get(trial["category"], 0) + 1
    total = sum(counts.values())
    return -sum((c / total) * math.log(c / total) for c in counts.values()) if total else 0.0


def decide(summary, *, now, window_started):
    """Pure promotion decision from one evaluation report; touches nothing."""
    if now - window_started > TUNER_WINDOW_MAX_S:
        return "rotate", {"reason": "window_aged_out"}
    if summary.get("status") != "ok" or not summary.get("valid"):
        return "wait", {"reason": "evidence_invalid", "validity_reasons": list(summary.get("validity_reasons", []))}
    trials = {arm: [t for t in summary["trials"] if t["arm"] == arm] for arm in ("base", "cand")}
    means = session_means(summary)
    counts = {arm: len(trials[arm]) for arm in trials}
    sessions = {arm: len(means[arm]) for arm in means}
    if min(counts.values()) < TUNER_MIN_TRIALS or min(sessions.values()) < TUNER_MIN_SESSIONS:
        return "wait", {"counts": counts, "sessions": sessions}
    stats = {}
    for arm, xs in means.items():
        n = len(xs)
        mean = sum(xs) / n
        stats[arm] = (n, mean, sum((x - mean) ** 2 for x in xs) / (n - 1))
    error = math.sqrt(stats["base"][2] / stats["base"][0] + stats["cand"][2] / stats["cand"][0])
    z = ((stats["cand"][1] - stats["base"][1]) / error) if error > 0 else 0.0
    detail = {"counts": counts, "sessions": sessions, "z": round(z, 3), "means": {arm: round(stats[arm][1], 4) for arm in stats}}
    if abs(z) < TUNER_Z:
        return "stall", detail
    winner, loser = ("cand", "base") if z > 0 else ("base", "cand")
    if entropy(trials[winner]) < TUNER_ENTROPY_VETO * entropy(trials[loser]):
        return "stall", {**detail, "reason": "entropy_veto", "winner_blocked": winner}
    return "promote", {**detail, "winner": winner}


class Tuner:
    """One tuner over one store path. ``cumulative_facts(items, cutoff_ts)`` is the owning
    engine's current Catalog/Signals reader labelled with the exact evidence cutoff;
    ``read_evidence`` defaults to the ledger's and is injectable for tests."""

    def __init__(self, store_path: str, *, ledger_path: str, cumulative_facts: Callable, registry=TUNER_REGISTRY,
                 clock: Callable[[], float] = time.time, read_evidence=None, summarize_trials=None,
                 verdict_fn=verdict, reward_fn=trial_reward, automatic: bool = True):
        self.path, self.ledger_path = str(store_path), str(ledger_path)
        self.registry = tuple(dict(e) for e in registry)
        self.cumulative_facts = cumulative_facts
        self.clock = clock
        self.read_evidence = read_evidence or ledger.read_evidence
        self.summarize_trials = summarize_trials or ledger.summarize_trials
        self.verdict, self.reward = verdict_fn, reward_fn
        self.automatic = automatic
        self.lock = threading.RLock()
        self.revision = [0]
        self.tick_ts = [0.0]
        self.arms_cache = TTLCache("tuner_arms", 60.0)
        self.tuned_cache = TTLCache("tuner_snapshot", 60.0)
        self.listeners = []

    # ------------------------------------------------------------- state
    def invalidate(self):
        """Advance shared in-process state under the same lock as cache publication."""
        with self.lock:
            self.revision[0] += 1
            self.arms_cache.clear()
            self.tuned_cache.clear()
            for listener in list(self.listeners):
                listener()

    def db(self):
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        for sql in _SCHEMA:
            conn.execute(sql)
        return conn

    @contextmanager
    def transaction(self, expected_revision=None):
        """Commit one current decision, or yield None when its evidence is obsolete."""
        with self.lock:
            if expected_revision is not None and expected_revision != self.revision[0]:
                yield None
                return
            conn = self.db()
            try:
                conn.execute("BEGIN IMMEDIATE")
                changes = conn.total_changes
                yield conn
                conn.commit()
                if conn.total_changes != changes:
                    self.invalidate()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    def registry_entry(self, knob):
        return registry_entry(knob, self.registry)

    def _read_only(self):
        return sqlite3.connect(f"file:{os.path.abspath(self.path)}?mode=ro", uri=True, timeout=5)

    def snapshot(self):
        """Read the existing experiment and tuned values; never initialize on read."""
        while True:
            now = self.clock()
            with self.lock:
                revision = self.revision[0]
                if self.tuned_cache.fresh(now):
                    return self.tuned_cache.value
            conn = None
            try:
                conn = self._read_only()
                row = conn.execute("SELECT active_knob, stalls, started_ts FROM experiment WHERE id=1").fetchone()
                vals = {k: v for k, v in conn.execute("SELECT knob, value FROM tuned")}
                with self.lock:
                    if revision != self.revision[0]:
                        continue
                    return self.tuned_cache.set((row, vals), now=now)
            except Exception:
                return None, {}
            finally:
                if conn is not None:
                    conn.close()

    def arms(self, knob=None):
        """Read existing arms for a knob; missing arms stay absent until maintenance."""
        requested = knob
        while True:
            now = self.clock()
            with self.lock:
                revision = self.revision[0]
            if requested is None:
                state, _vals = self.snapshot()
                if state is None:
                    return None
                knob = state[0]
            with self.lock:
                if revision != self.revision[0]:
                    continue
                if self.arms_cache.fresh(now) and self.arms_cache.sig == knob:
                    return self.arms_cache.value
            conn = None
            try:
                conn = self._read_only()
                row = conn.execute("SELECT base, candidate, since_ts FROM arms WHERE knob=?", (knob,)).fetchone()
                with self.lock:
                    if revision != self.revision[0]:
                        continue
                    return self.arms_cache.set(row, sig=knob, now=now)
            except Exception:
                return None
            finally:
                if conn is not None:
                    conn.close()

    def resolve_knobs(self, supplied: Mapping[str, Any]):
        """Explicitly supplied knob values (including zero or the default) win and
        suppress experimentation on that knob; others read their settled value."""
        with self.lock:
            revision = self.revision[0]
            state, tuned = self.snapshot()
            active = state[0] if state else None
            arms = self.arms(active) if active else None
        values, user_set = {}, {}
        for knob in KNOB_DEFAULTS:
            user_set[knob] = knob in supplied
            values[knob] = float(supplied[knob]) if user_set[knob] else float(tuned.get(knob, knob_default(knob)))
        active_now = arms is not None and active is not None and not user_set.get(active, False)
        if active_now:
            values[active] = float(arms[0])
        experiment = None
        if active_now and abs(float(arms[1]) - float(arms[0])) > 1e-9:
            experiment = {"knob": active, "base": float(arms[0]), "candidate": float(arms[1]),
                          "since_ts": arms[2], "revision": revision}
        return {"values": values, "user_set": user_set, "experiment": experiment, "revision": revision,
                "state": (state, tuned), "arms": arms}

    def stable(self, resolved):
        """Whether the tuner state a generation was built from is unchanged."""
        with self.lock:
            state, tuned = self.snapshot()
            active = state[0] if state else None
            return (resolved["revision"] == self.revision[0] and (state, tuned) == resolved["state"]
                    and (not active or self.arms(active) == resolved["arms"]))

    # ---------------------------------------------------------- evaluation
    def evaluate(self, *, since_ts=None, through_ts=None, experiment_id=None, cumulative_at_cutoff=None):
        """Explicit evidence report only; no automatic promotion or cumulative rewards."""
        if since_ts is None and through_ts is None:
            return {"status": "disabled", "reason": "attribution_contract_required"}
        if since_ts is None or through_ts is None:
            raise ValueError("both evidence cutoffs are required")
        evidence = self.read_evidence(self.ledger_path, since_ts=since_ts, through_ts=through_ts, experiment_id=experiment_id)
        report = self.summarize_trials(evidence, verdict=self.verdict, trial_reward=self.reward,
                                       cumulative_at_cutoff=cumulative_at_cutoff)
        return {**report, "promotion_enabled": False}

    def promote(self, detail, expected_revision=None):
        """Adopt the winning arm as the tuned value: one revertible, ledgered step, then rotate."""
        with self.transaction(expected_revision) as conn:
            if conn is None:
                return False
            state = conn.execute("SELECT active_knob FROM experiment WHERE id=1").fetchone()
            if not state:
                return False
            knob = state[0]
            row = conn.execute("SELECT base, candidate FROM arms WHERE knob=?", (knob,)).fetchone()
            if not row:
                return False
            new_value = float(row[0] if detail["winner"] == "base" else row[1])
            old = conn.execute("SELECT value FROM tuned WHERE knob=?", (knob,)).fetchone()
            old_value = float(old[0]) if old else knob_default(knob)
            now = self.clock()
            conn.execute("INSERT INTO tuned VALUES (?,?,?) ON CONFLICT(knob) DO UPDATE SET "
                         "value=excluded.value, updated_ts=excluded.updated_ts", (knob, new_value, now))
            conn.execute("INSERT INTO ledger (ts, knob, old_value, new_value, evidence, status) VALUES (?,?,?,?,?,?)",
                         (now, knob, old_value, new_value, json.dumps(detail), "applied"))
            conn.execute("UPDATE experiment SET stalls=0 WHERE id=1")
            next_revision = self.revision[0] + 1
        self.rotate("promoted", expected_revision=next_revision)
        return True

    def tick(self, *, now=None):
        """One autonomous evaluation of the active experiment, end to end."""
        now = self.clock() if now is None else now
        with self.lock:
            revision = self.revision[0]
            state, _values = self.snapshot()
            active_knob = state[0] if state else None
            arms = self.arms(active_knob) if active_knob else None
        if not active_knob or not arms:
            return {"action": "disabled"}
        window_started = float(arms[2] or 0.0)
        through = now - TUNER_RIPEN_S
        if through <= window_started:
            return {"action": "wait", "reason": "window_unripe"}
        evidence = self.read_evidence(self.ledger_path, since_ts=window_started, through_ts=through)
        completed = evidence.get("attribution_run")
        if completed and window_started < float(completed["through_ts"]) < through:
            through = float(completed["through_ts"])
            evidence = self.read_evidence(self.ledger_path, since_ts=window_started, through_ts=through)
        if evidence.get("status") != "ok":
            return {"action": "wait", "reason": evidence.get("status", "unavailable")}
        viewed = set(evidence.get("viewed_ids") or [])
        items = {(row["kind"], row["item_id"]) for row in evidence.get("events", []) if row["event_id"] in viewed}
        report = self.summarize_trials(evidence, verdict=self.verdict, trial_reward=self.reward,
                                       cumulative_at_cutoff=self.cumulative_facts(items, evidence["through_ts"]))
        action, detail = decide(report, now=now, window_started=window_started)
        if action == "promote":
            self.promote(detail, expected_revision=revision)
        elif action == "stall":
            self.stall(json.dumps(detail), expected_revision=revision)
        elif action == "rotate":
            self.rotate(detail.get("reason", "window_aged_out"), expected_revision=revision)
        return {"action": action, "detail": detail}

    def maybe_tick(self):
        """Schedule the autonomous evaluation off the serve path; never block a feed."""
        if not self.automatic:
            return None
        now = self.clock()
        if now - self.tick_ts[0] < TUNER_EVAL_EVERY_S:
            return None
        self.tick_ts[0] = now

        def run():
            try:
                self.tick()
            except Exception:
                pass
        thread = threading.Thread(target=run, name="tuner-tick", daemon=True)
        thread.start()
        return thread

    def stall(self, evidence: str, expected_revision=None) -> None:
        """Volume met, no significance: count the stall, restart the window, rotate after TUNER_STALL_EVALS."""
        with self.transaction(expected_revision) as conn:
            if conn is None:
                return
            now = self.clock()
            state = conn.execute("SELECT active_knob, stalls FROM experiment WHERE id=1").fetchone()
            if state is None:
                return
            knob, stalls = state[0], state[1] + 1
            conn.execute("INSERT INTO ledger (ts, knob, old_value, new_value, evidence, status) VALUES (?,?,?,?,?,?)",
                         (now, knob, None, None, evidence, "no_verdict"))
            conn.execute("UPDATE experiment SET stalls=? WHERE id=1", (stalls,))
            conn.execute("UPDATE arms SET since_ts=? WHERE knob=?", (now, knob))
            next_revision = self.revision[0] + 1
        if stalls >= TUNER_STALL_EVALS:
            self.rotate("stalled_no_effect", expected_revision=next_revision)

    def rotate(self, reason: str, expected_revision=None) -> None:
        """Advance to the next registry knob; the outgoing one keeps its tuned value.
        The incoming knob's arms are deleted; ``initialize`` must seed them explicitly."""
        with self.transaction(expected_revision) as conn:
            if conn is None:
                return
            now = self.clock()
            state = conn.execute("SELECT active_knob FROM experiment WHERE id=1").fetchone()
            order = [e["knob"] for e in self.registry]
            current = state[0] if state else order[0]
            nxt = order[(order.index(current) + 1) % len(order)] if current in order else order[0]
            conn.execute("INSERT INTO ledger (ts, knob, old_value, new_value, evidence, status) VALUES (?,?,?,?,?,?)",
                         (now, current, None, None, json.dumps({"reason": reason, "next": nxt}), "rotated"))
            conn.execute("UPDATE experiment SET active_knob=?, stalls=0, started_ts=? WHERE id=1", (nxt, now))
            conn.execute("DELETE FROM arms WHERE knob=?", (nxt,))

    def initialize(self):
        """Explicit maintenance entry: create the store, the experiment row and the active knob's arms."""
        with self.transaction() as conn:
            now = self.clock()
            conn.execute("INSERT OR IGNORE INTO experiment VALUES (1,?,?,?)", (self.registry[0]["knob"], 0, now))
            knob = conn.execute("SELECT active_knob FROM experiment WHERE id=1").fetchone()[0]
            entry = self.registry_entry(knob)
            if entry is None:
                raise ValueError("unknown active tuner knob")
            row = conn.execute("SELECT value FROM tuned WHERE knob=?", (knob,)).fetchone()
            base = float(row[0]) if row else knob_default(knob)
            candidate = round(min(entry["max"], max(entry["min"], base + entry["step"])), 6)
            conn.execute("INSERT OR IGNORE INTO arms VALUES (?,?,?,?)", (knob, base, candidate, now))

    # ------------------------------------------------------------ control
    def write(self, action: str, ledger_id: int | None = None) -> dict:
        """Restore authoritative knob values atomically, rejecting stale ledger undo."""
        if action not in ("reset", "revert"):
            return {"ok": False, "error": "unknown action"}
        try:
            with self.lock:
                with self.transaction() as conn:
                    now = self.clock()
                    entries = {e["knob"]: e for e in self.registry}
                    if action == "reset":
                        restored = {knob: knob_default(knob) for knob in entries}
                    else:
                        row = conn.execute("SELECT knob, old_value, new_value, status FROM ledger WHERE id=?", (ledger_id,)).fetchone()
                        if row is None or row[3] != "applied" or row[0] not in entries:
                            return {"ok": False, "error": "ledger row not found or not an applied move"}
                        knob, old_v, new_v, _status = row
                        settled = conn.execute("SELECT value FROM tuned WHERE knob=?", (knob,)).fetchone()
                        newer = conn.execute("SELECT 1 FROM ledger WHERE knob=? AND id>? AND status IN "
                                             "('applied','revert','reset_to_standard') LIMIT 1", (knob, ledger_id)).fetchone()
                        if newer or not settled or settled[0] != new_v or old_v is None:
                            return {"ok": False, "error": "knob changed since this move; undo rejected"}
                        restored = {knob: float(old_v)}
                        conn.execute("UPDATE ledger SET status='reverted' WHERE id=?", (ledger_id,))
                    for knob, value in restored.items():
                        entry = entries[knob]
                        if not math.isfinite(value) or not entry["min"] <= value <= entry["max"]:
                            raise ValueError("restored knob value outside registry bounds")
                        settled = conn.execute("SELECT value FROM tuned WHERE knob=?", (knob,)).fetchone()
                        old = settled[0] if settled else knob_default(knob)
                        conn.execute("INSERT INTO tuned VALUES (?,?,?) ON CONFLICT(knob) DO UPDATE SET "
                                     "value=excluded.value, updated_ts=excluded.updated_ts", (knob, value, now))
                        conn.execute("INSERT INTO arms VALUES (?,?,?,?) ON CONFLICT(knob) DO UPDATE SET "
                                     "base=excluded.base, candidate=excluded.candidate, since_ts=excluded.since_ts",
                                     (knob, value, round(min(entry["max"], max(entry["min"], value + entry["step"])), 6), now))
                        conn.execute("UPDATE experiment SET stalls=0, started_ts=? WHERE active_knob=?", (now, knob))
                        if action == "reset":
                            conn.execute("UPDATE ledger SET status='superseded_by_reset' WHERE knob=? AND status='applied'", (knob,))
                        conn.execute("INSERT INTO ledger (ts, knob, old_value, new_value, evidence, status) VALUES (?,?,?,?,?,?)",
                                     (now, knob, old, value, json.dumps({"reverted_ledger_id": ledger_id}) if action == "revert" else "{}",
                                      "revert" if action == "revert" else "reset_to_standard"))
            return {"ok": True, "action": action, "restored": restored}
        except (sqlite3.Error, ValueError):
            return {"ok": False, "error": "tuner change failed; no changes committed"}

    def display(self):
        """Arms and the last 30 ledger rows for the scorecard, read only; missing store reads empty."""
        conn = None
        try:
            conn = self._read_only()
            arms = {k: (b, c, ts) for k, b, c, ts in conn.execute("SELECT knob, base, candidate, since_ts FROM arms")}
            rows = conn.execute("SELECT id, ts, knob, old_value, new_value, evidence, status FROM ledger ORDER BY id DESC LIMIT 30").fetchall()
        except sqlite3.Error:
            arms, rows = {}, []
        finally:
            if conn is not None:
                conn.close()
        entries = []
        for lid, ts, knob, old, new, evidence, status in rows:
            try:
                evidence = json.loads(evidence) if evidence else {}
            except (ValueError, TypeError):
                evidence = {"status": "unavailable"}
            entries.append({"id": lid, "ts": ts, "knob": knob, "old": old, "new": new, "evidence": evidence, "status": status})
        return arms, entries


__all__ = ["TUNER_MIN_TRIALS", "TUNER_MIN_SESSIONS", "TUNER_RIPEN_S", "TUNER_EVAL_EVERY_S", "TUNER_Z", "TUNER_ENTROPY_VETO",
           "TUNER_STALL_EVALS", "TUNER_WINDOW_MAX_S", "TUNER_REGISTRY", "KNOB_DEFAULTS", "registry_entry", "knob_default",
           "session_means", "entropy", "decide", "Tuner"]

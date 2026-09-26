"""Contracts for the norm precompute, the lazy fallback, the fenced window memo and stage timing."""
import random
import unittest

import numpy as np

import feedloop
from feedloop import engine, ranking, serving
from feedloop.profiles import TTLCache


def vector(rng):
    return {rng.randrange(40): rng.choice([0.0, rng.random(), -rng.random()]) for _ in range(rng.randrange(0, 12))}


class NormedCosine(unittest.TestCase):
    def test_cosine_normed_is_bit_identical_to_cosine(self):
        rng = random.Random(7)
        for _ in range(2000):
            a, b = vector(rng), vector(rng)
            self.assertEqual(ranking.cosine_normed(a, b, ranking.norm(a), ranking.norm(b)).hex(), ranking.cosine(a, b).hex())

    def test_select_reduces_each_norm_once(self):
        rng = random.Random(3)
        scored = [(rng.random(), i, vector(rng), rng.choice("abc")) for i in range(30)]
        calls = []
        original = ranking.norm
        ranking.norm = lambda vec: calls.append(1) or original(vec)
        try:
            ranking.select(scored, want=10, diversity=0.35, calibration=0.25, target_shares={"a": 0.5})
        finally:
            ranking.norm = original
        self.assertEqual(len(calls), 30)

    def test_exports(self):
        for name in ("norm", "cosine_normed", "cosine", "fallback_ids", "random_control", "best_windows",
                     "stage_add", "stage_total", "staged", "stage_line", "STAGE_TIMINGS"):
            self.assertIn(name, feedloop.__all__)


class Fallback(unittest.TestCase):
    def test_enumeration_is_lazy(self):
        def boom():
            raise AssertionError("catalog enumerated")
        self.assertEqual(engine.fallback_ids({2}, 3, enumerate_ids=boom, eligible_ids=[5, 2, 1, 4]), [1, 4, 5])

    def test_seeded_shuffle_and_ascending(self):
        ids = list(range(20))
        expected = sorted(set(ids) - {3})
        random.Random(9).shuffle(expected)
        self.assertEqual(engine.fallback_ids({3}, 5, enumerate_ids=lambda: ids, seed=9), expected[:5])
        self.assertEqual(engine.fallback_ids({3}, 5, enumerate_ids=lambda: ids), [0, 1, 2, 4, 5])

    def test_random_control(self):
        pick = engine.random_control(range(50), {1}, seed=4)
        self.assertEqual(pick, engine.random_control(reversed(range(50)), {1}, seed=4))
        self.assertIsNone(engine.random_control(range(5), set(), eligible_ids=[]))
        self.assertIsNone(engine.random_control([1], {1}))
        self.assertEqual(engine.random_control(range(50), set(), eligible_ids=[7], seed=4), 7)


class Windows(unittest.TestCase):
    def setUp(self):
        self.cache = TTLCache("windows", 3600.0)
        self.revision = ["r1"]
        self.reads = []

    def read(self, query, ids):
        self.reads.append(list(ids))
        return {i: float(i) for i in ids if i != 3}

    def call(self, ids, **kw):
        return engine.best_windows(np.ones(4, dtype=np.float32), ids, read=kw.get("read", self.read),
                                   revision=kw.get("revision", lambda: self.revision[0]), cache=self.cache)

    def test_memo_serves_known_and_reads_only_missing(self):
        self.assertEqual(self.call([1, 2, 3]), {1: 1.0, 2: 2.0})
        self.assertEqual(self.call([2, 3, 4]), {2: 2.0, 4: 4.0})
        self.assertEqual(self.reads, [[1, 2, 3], [4]])

    def test_revision_change_mid_read_serves_and_publishes_nothing(self):
        def read(query, ids):
            self.revision[0] = "r2"
            return self.read(query, ids)
        self.assertEqual(self.call([1], read=read), {})
        self.assertIsNone(self.cache.value)

    def test_reset_mid_read_publishes_nothing(self):
        def read(query, ids):
            self.cache.clear()
            return self.read(query, ids)
        self.assertEqual(self.call([1], read=read), {})
        self.assertIsNone(self.cache.value)

    def test_failed_read_publishes_nothing_and_unknown_revision_reads_through(self):
        def fail(query, ids):
            raise OSError("down")
        self.assertEqual(self.call([1], read=fail), {})
        self.assertIsNone(self.cache.value)
        self.assertEqual(self.call([1, 2], revision=lambda: None), {1: 1.0, 2: 2.0})
        self.assertIsNone(self.cache.value)
        self.assertEqual(engine.best_windows(None, [1], read=self.read, revision=lambda: "r1", cache=self.cache), {})


class Stages(unittest.TestCase):
    def test_noop_without_request_dict(self):
        serving.stage_add("x", 1.0)
        self.assertEqual(serving.stage_total("x"), 0)

    def test_staged_is_exclusive_of_matrices_and_line_partitions(self):
        stages = {}
        token = serving.STAGE_TIMINGS.set(stages)
        try:
            with serving.staged("selection"):
                serving.stage_add("matrix_visual", 5.0)
            serving.stage_add("ranker", 7.0)
        finally:
            serving.STAGE_TIMINGS.reset(token)
        self.assertLess(stages["selection"], 0.0)
        self.assertEqual(serving.stage_total("matrix_"), 0)
        line = serving.stage_line("feed", 0, "ok", {"ranker": 7.0, "matrix_visual": 5.0, "selection": 1.0}, 8.0)
        self.assertIn("matrices=5.00 matrix_visual=5.00", line)
        self.assertIn("selection=1.00", line)
        self.assertIn("ranking=1.00", line)


if __name__ == "__main__":
    unittest.main()

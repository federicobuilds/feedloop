"""v0.4.2 performance changes keep the old semantics: set membership in keyed reads, the
fingerprint pin memo, the per-feed view read reuse, and the vectorized discovery means; v0.4.3:
the evidence decode reuse across reads and the batched select similarities."""
from unittest.mock import Mock

import numpy as np
import pytest

from feedloop import catalog, discovery, ledger, ranking
from fl2_helpers import MemoryCatalog, catalog_row, md5_file
from test_engine import REQUEST, clock, make_engine  # noqa: F401  (clock is a fixture)


def old_means(index, matrix, keys=None):
    """The v0.4.1 per-row loop, kept verbatim as the reference."""
    wanted = None if keys is None else set(keys)
    out = {}
    for row, key in enumerate(index):
        if wanted is not None and key not in wanted:
            continue
        vector = np.asarray(matrix[row], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.ndim == 1 and np.isfinite(vector).all() and norm > 0:
            out[key] = vector / norm
    return out


class Spaces:
    def __init__(self, loaded):
        self.loaded = loaded

    def matrix(self, space):
        return self.loaded


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vectorized_means_equal_the_per_row_loop_bit_for_bit(dtype):
    rng = np.random.default_rng(11)
    index = [("video", i) for i in range(1, 401)] + [("image", i) for i in range(1, 201)] + [("video", 5)]
    matrix = (rng.standard_normal((len(index), 384)) * 4).astype(dtype)
    matrix[3] = 0.0
    matrix[7, 1] = np.nan
    matrix[9, 2] = np.inf
    matrix[11] = 3e38
    sources = discovery.Sources.__new__(discovery.Sources)
    sources.spaces = Spaces((index, matrix))
    subset = index[::3] + [("video", 9999)]
    for keys in (None, subset, []):
        new, old = sources.means("visual", keys), old_means(index, matrix, keys)
        assert new.keys() == old.keys()
        for key in old:
            assert new[key].dtype == old[key].dtype == np.float32
            assert np.array_equal(new[key], old[key]), key


def test_empty_space_has_no_means():
    sources = discovery.Sources.__new__(discovery.Sources)
    sources.spaces = Spaces(([], np.zeros((0, 8), dtype=np.float32)))
    assert sources.means("visual") == {}


def test_keyed_read_still_rejects_an_unrequested_id():
    source = MemoryCatalog([catalog_row("video", i) for i in (1, 2, 3)])
    original = source.fetch
    def extra(keys, *, timeout_s=30):
        payload = original([k for k in keys if k != ("video", 2)] + [("video", 3)], timeout_s=timeout_s)
        payload["total"] = len(keys)
        return payload
    source.fetch = extra
    with pytest.raises(RuntimeError, match="catalog_enumeration_partial"):
        catalog.read_catalog(source, [("video", 1), ("video", 2)], kinds=("video",))


def test_snapshot_memo_reuses_the_pin_only_while_identity_is_equal():
    source = MemoryCatalog([catalog_row("video", i, files=[md5_file("a")]) for i in (1, 2)] + [catalog_row("image", 4)])
    keys = sorted(source.rows)
    memo = catalog.SnapshotMemo()
    first = catalog.fingerprint_snapshot(source, keys, memo=memo)
    assert first == catalog.fingerprint_snapshot(source, keys)
    source.rows[("video", 1)]["updated"] = "t2"
    assert catalog.fingerprint_snapshot(source, keys, memo=memo) is first, "an update token is not ranking identity"
    source.rows[("video", 2)]["files"] = [md5_file("b")]
    changed = catalog.fingerprint_snapshot(source, keys, memo=memo)
    assert changed is not first and changed == catalog.fingerprint_snapshot(source, keys)
    assert changed["revision"] != first["revision"]
    assert catalog.fingerprint_snapshot(source, keys[:2], memo=memo) == catalog.fingerprint_snapshot(source, keys[:2])
    memo.clear()
    assert memo.entry is None


def test_views_read_the_ledger_once_per_feed_and_again_after_a_write(tmp_path, clock, monkeypatch):  # noqa: F811
    now, fake = clock
    now[0] = 90.0
    eng, _signals, _spaces = make_engine(tmp_path, fake, config={"impression_discount": 0.5})
    now[0] = 100.0
    reads = Mock(wraps=ledger.read_qualified_views)
    monkeypatch.setattr(ledger, "read_qualified_views", reads)
    page = eng.feed(REQUEST, record_delivery=False)
    assert page["status"] == "ok", page
    assert reads.call_count == 1, "the post-build fence reuses the pre-build read"
    before = ledger.generation(eng.ledger_path)
    eng.feed({**REQUEST, "request_id": "req-w", "client_request_id": "client-w"})
    assert ledger.generation(eng.ledger_path) != before, "a committed write moves the generation"
    reads.reset_mock()
    eng._views(100.0)
    assert reads.call_count == 1, "a write invalidates the reuse"
    eng.reset_caches()
    eng._views(100.0)
    assert reads.call_count == 2


def _fresh_evidence(path, **window):
    ledger._DECODED.clear()
    return ledger.read_evidence(path, **window)


def test_evidence_decode_is_reused_across_reads_until_a_write(tmp_path, clock, monkeypatch):  # noqa: F811
    now, fake = clock
    now[0] = 90.0
    eng, _signals, _spaces = make_engine(tmp_path, fake, config={"impression_discount": 0.5})
    now[0] = 100.0
    eng.feed(REQUEST)
    path = eng.ledger_path
    loads = Mock(wraps=ledger.json.loads)
    monkeypatch.setattr(ledger.json, "loads", loads)
    first = _fresh_evidence(path, since_ts=0.0, through_ts=150.0)
    decoded = loads.call_count
    assert first["status"] == "ok" and decoded > 0
    loads.reset_mock()
    before = ledger.generation(path)
    again = ledger.read_evidence(path, since_ts=0.0, through_ts=160.0)
    assert loads.call_count == 0, "a later window over the same generation decodes nothing"
    assert ledger.generation(path) == before, "the read writes nothing"
    assert {**again, "through_ts": 150.0} == first
    narrow = ledger.read_evidence(path, since_ts=0.0, through_ts=95.0)
    assert narrow == _fresh_evidence(path, since_ts=0.0, through_ts=95.0), "the window filters after the reuse"
    ledger.read_evidence(path, since_ts=0.0, through_ts=150.0)
    now[0] = 110.0
    eng.feed({**REQUEST, "request_id": "req-w", "client_request_id": "client-w"})
    assert ledger.generation(path) != before
    loads.reset_mock()
    after = ledger.read_evidence(path, since_ts=0.0, through_ts=150.0)
    assert loads.call_count >= decoded, "a committed write starts a fresh decode"
    assert after == _fresh_evidence(path, since_ts=0.0, through_ts=150.0)


def test_a_stale_memo_entry_is_never_served(tmp_path, clock):  # noqa: F811
    now, fake = clock
    now[0] = 90.0
    eng, _signals, _spaces = make_engine(tmp_path, fake, config={"impression_discount": 0.5})
    now[0] = 100.0
    eng.feed(REQUEST)
    path = eng.ledger_path
    truth = _fresh_evidence(path, since_ts=0.0, through_ts=150.0)
    requests, _events = ledger._decoded(path, ledger.generation(path))
    for request_id, hit in list(requests.items()):
        requests[request_id] = (("{}", "{}"), {"stale": True}, {})
    assert ledger.read_evidence(path, since_ts=0.0, through_ts=150.0) == truth


def old_select(scored, *, want, diversity, calibration, target_shares, details=None):
    """The v0.4.2 select with the per-pair generator, kept verbatim as the reference."""
    from collections import defaultdict
    from feedloop.ranking import cosine_normed, norm
    chosen = []
    last_vector, last_norm = {}, 0.0
    counts = defaultdict(int)
    pool = scored[:]
    norms = [norm(vec) for _rel, _sid, vec, _cat in pool]
    max_sims = [0.0] * len(pool)
    scale = abs(scored[0][0] if scored else 1.0) or 1.0
    while pool and len(chosen) < want:
        best_idx, best_val, best_trace = 0, -1e18, None
        for idx, (rel, sid, vec, cat) in enumerate(pool):
            if chosen:
                sim = cosine_normed(vec, last_vector, norms[idx], last_norm)
                if sim > max_sims[idx]:
                    max_sims[idx] = sim
            sim = max_sims[idx]
            have = counts[cat] / (len(chosen) or 1)
            deficit = max(0.0, target_shares.get(cat, 0.0) - have)
            val = rel / scale - diversity * sim + calibration * deficit
            if val > best_val:
                best_idx, best_val = idx, val
                if details is not None:
                    best_trace = {"relevance_normalized": rel / scale, "scale": scale,
                                  "maximum_similarity": sim, "diversity_penalty": diversity * sim,
                                  "category_deficit": deficit, "calibration_bonus": calibration * deficit,
                                  "value": val, "ranked_position": len(chosen)}
        rel, sid, vec, cat = pool.pop(best_idx)
        last_norm = norms.pop(best_idx)
        max_sims.pop(best_idx)
        if details is not None:
            details.setdefault(sid, {})["selection"] = best_trace
        chosen.append(sid)
        last_vector = dict(vec)
        counts[cat] += 1
    return chosen


def _scored(seed, n=160, tags=90):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        size = int(rng.integers(0, 40))
        keys = rng.choice(tags, size=size, replace=False).tolist()
        # seconds-like magnitudes spread over many orders, a few negatives, exact ties and tiny values
        vec = {int(k): float(rng.choice([rng.uniform(0.5, 4000.0), 1e-300 * rng.uniform(), -rng.uniform(0, 50), 120.0]))
               for k in keys}
        rows.append((float(rng.choice([rng.uniform(0.1, 3.0), 1.0])), ("video", i), vec, f"c{i % 4}"))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows


def _edge_rows():
    big = {k: 1e300 for k in range(8)}
    return [(2.0, ("video", 1), {}, "a"), (1.9, ("video", 2), {1: 3.0, 2: 4.0}, "a"),
            (1.8, ("video", 3), {2: 4.0, 1: 3.0}, "b"), (1.7, ("video", 4), {1: 0.0, 2: -0.0}, "b"),
            (1.6, ("video", 5), {1: 2, 2: 5}, "a"), (1.5, ("video", 6), {1: float("inf"), 3: 1.0}, "a"),
            (1.4, ("video", 7), {1: float("nan"), 2: 1.0}, "b"), (1.3, ("video", 8), big, "b"),
            (1.2, ("video", 9), dict(big), "a"), (1.1, ("video", 10), {"x": 1.0, 2: 7.0, 1: 1e-320}, "a"),
            (1.0, ("video", 11), {k: 1.0 + k * 1e-16 for k in range(60)}, "b"),
            (0.9, ("video", 12), {k: -1.0 for k in range(5)}, "a")]


@pytest.mark.parametrize("seed", range(6))
def test_batched_select_equals_the_generator_bit_for_bit(seed):
    scored = _scored(seed)
    for diversity, calibration, shares in ((0.35, 0.25, {"c0": 0.5, "c1": 0.2}), (0.9, 0.0, {}), (0.0, 0.5, {"c3": 1.0})):
        got, want = {}, {}
        assert ranking.select(scored, want=len(scored), diversity=diversity, calibration=calibration,
                              target_shares=shares, details=got) == \
            old_select(scored, want=len(scored), diversity=diversity, calibration=calibration,
                       target_shares=shares, details=want)
        assert repr(got) == repr(want)


def test_batched_select_equals_the_generator_on_edge_rows():
    scored = _edge_rows()
    got, want = {}, {}
    with np.errstate(all="ignore"):
        assert ranking.select(scored, want=len(scored), diversity=0.6, calibration=0.1, target_shares={"a": 0.5},
                              details=got) == \
            old_select(scored, want=len(scored), diversity=0.6, calibration=0.1, target_shares={"a": 0.5}, details=want)
    assert repr(got) == repr(want)
    assert ranking.select([], want=3, diversity=0.5, calibration=0.1, target_shares={}) == []


def test_similarity_batch_matches_cosine_normed_per_pair():
    rows = _scored(7, n=120) + _edge_rows()
    vectors = [vec for _rel, _sid, vec, _cat in rows]
    norms = [ranking.norm(v) for v in vectors]
    batch = ranking._CosineBatch(vectors)
    every = np.arange(len(vectors))
    with np.errstate(all="ignore"):
        for last in range(len(vectors)):
            got = batch.similarities(every, last, norms, norms[last])
            want = [ranking.cosine_normed(v, dict(vectors[last]), norms[i], norms[last]) if vectors[last] else 0.0
                    for i, v in enumerate(vectors)]
            assert [x.hex() if isinstance(x, float) else x for x in got] == [float(x).hex() for x in want], last

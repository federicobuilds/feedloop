"""v0.4.2 performance changes keep the old semantics: set membership in keyed reads, the
fingerprint pin memo, the per-feed view read reuse, and the vectorized discovery means."""
from unittest.mock import Mock

import numpy as np
import pytest

from feedloop import catalog, discovery, ledger
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
    reads = Mock(wraps=ledger.read_evidence)
    monkeypatch.setattr(ledger, "read_evidence", reads)
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

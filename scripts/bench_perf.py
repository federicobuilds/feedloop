"""Microbenchmark of the per-feed hot paths on production-sized synthetic data.

Usage: python scripts/bench_perf.py [src_dir]
src_dir defaults to this checkout's src; pass a vendored copy's parent to time it.
Prints one line per path: read_catalog keyed pass, fingerprint_snapshot, means."""
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from feedloop import catalog, discovery  # noqa: E402

SCENES, IMAGES, DIM = 17_500, 28_700, 512


def row(kind, id_):
    return {"kind": kind, "id": id_, "updated": f"2026-09-{id_ % 28 + 1:02d}", "files": [
        {"fingerprints": [{"type": "phash", "value": f"{id_:016x}"}, {"type": "oshash", "value": f"{id_ * 7:016x}"}]}]}


class Source:
    def __init__(self):
        self.rows = {("video", i): row("video", i) for i in range(1, SCENES + 1)}
        self.rows.update({("image", i): row("image", i) for i in range(1, IMAGES + 1)})

    def fetch(self, keys, *, timeout_s=30):
        items = [dict(self.rows[k]) for k in keys]
        return {"items": items, "total": len(items), "change_token": "t"}


class Spaces:
    def __init__(self, index, matrix):
        self.loaded = (index, matrix)

    def matrix(self, space):
        return self.loaded


def timed(label, call):
    started = time.perf_counter()
    call()
    print(f"{label} {time.perf_counter() - started:.3f}s", flush=True)


source = Source()
keys = sorted(source.rows)
timed("read_catalog_keyed_pass", lambda: catalog.read_catalog(source, keys, kinds=("video", "image")))
timed("fingerprint_snapshot", lambda: catalog.fingerprint_snapshot(source, keys, kinds=("video", "image")))
memo = catalog.SnapshotMemo()
catalog.fingerprint_snapshot(source, keys, kinds=("video", "image"), memo=memo)
timed("fingerprint_snapshot_memo_hit", lambda: catalog.fingerprint_snapshot(source, keys, kinds=("video", "image"), memo=memo))
rng = np.random.default_rng(7)
matrix = rng.standard_normal((len(keys), DIM)).astype(np.float32)
matrix[3] = 0.0
matrix[5, 2] = np.nan
sources = discovery.Sources.__new__(discovery.Sources)
sources.spaces = Spaces(keys, matrix)
timed("means_all_keys", lambda: sources.means("visual", keys))

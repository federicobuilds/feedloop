"""Neutral in-memory slot fixtures shared by the FL-2 contract tests."""
from __future__ import annotations

import numpy as np

from feedloop.slots import MissingKeys


class MemoryCatalog:
    """Catalog slot over plain rows: {"kind","id","title","duration_s","media_url","tags",
    "contributor_ids","updated","files"}, with weighted features and tag names."""

    def __init__(self, rows, features=None, names=None, *, page_size=500):
        self.rows = {(r["kind"], r["id"]): dict(r) for r in rows}
        self.feature_rows = dict(features or {})
        self.names = dict(names or {})
        self.page_size = page_size
        self.calls = []
        self.change_token = "catalog-1"

    def enumerate(self, kinds, page, page_size, *, timeout_s=30):
        self.calls.append(("enumerate", tuple(kinds), page, timeout_s))
        rows = sorted((r for r in self.rows.values() if r["kind"] in kinds), key=lambda r: (r["kind"], r["id"]))
        start = (page - 1) * page_size
        return {"items": [dict(r) for r in rows[start:start + page_size]], "total": len(rows), "change_token": self.change_token}

    def fetch(self, keys, *, timeout_s=30):
        self.calls.append(("fetch", tuple(keys), timeout_s))
        missing = [k for k in keys if k not in self.rows]
        if missing:
            raise MissingKeys(missing)
        items = [dict(self.rows[k]) for k in sorted(set(keys))]
        return {"items": items, "total": len(items), "change_token": self.change_token}

    def features(self, keys):
        return {k: self.feature_rows[k] for k in keys if k in self.feature_rows}

    def tag_names(self):
        return dict(self.names)


class MemorySignals:
    def __init__(self, rows=None, *, observed_at=0.0):
        self.rows = dict(rows or {})
        self.observed_at = observed_at

    def read(self, keys=None):
        rows = self.rows if keys is None else {k: self.rows[k] for k in keys if k in self.rows}
        return {"observed_at": self.observed_at, "rows": {k: dict(v) for k, v in rows.items()}}

    def current(self, key):
        row = self.rows.get(key, {})
        return {"status": "ok", "rating100": row.get("rating"), "engagement_count": int(row.get("engagement_count", 0))}

    def apply(self, key, change):
        row = self.rows.setdefault(key, {"rating": None, "engagement_count": 0})
        if change["action"] == "rating":
            row["rating"] = change["rating100"]
        else:
            row["engagement_count"] = row.get("engagement_count", 0) + change["delta"]
        return {"status": "confirmed", **{k: v for k, v in self.current(key).items() if k != "status"}}


class MemorySpaces:
    def __init__(self, matrices, revisions=None, windows=None):
        self.matrices = dict(matrices)
        self.revisions = revisions if revisions is not None else {name: 1 for name in matrices}
        self.window_rows = dict(windows or {})

    def spaces(self):
        return sorted(set(self.matrices) | set(self.window_rows))

    def matrix(self, space):
        return self.matrices.get(space)

    def revision(self, space):
        return self.revisions.get(space)

    def windows(self, space):
        return self.window_rows.get(space)


def unit_rows(count, dim, seed):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(count, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def md5_file(char):
    return {"fingerprints": [{"type": "md5", "value": char * 32}]}


def catalog_row(kind, item_id, *, files=None, duration=600.0, tags=(), contributors=(), updated="t1"):
    return {"kind": kind, "id": item_id, "title": f"item {kind} {item_id}", "duration_s": duration,
            "media_url": f"/media/{kind}/{item_id}", "tags": list(tags), "contributor_ids": list(contributors),
            "updated": updated, "files": [] if files is None else files}

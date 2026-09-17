"""Catalog, Signals and FeatureSpaces over a media folder, sidecar JSON and one local SQLite file.

Identity: every media file under the folder becomes one item. Its kind follows the
file extension (video or image), its positive integer id is assigned once and kept
in the state database, so the same path keeps the same key across reopenings. A
file that disappears is no longer present; a file that comes back keeps its id.

Metadata: an optional sidecar ``<file>.json`` next to the media supplies ``title``,
``tags`` (strings or ``{"name", "category", "seconds"}``), ``contributors``,
``duration_s`` and ``tag_seconds``. Nothing is decoded from the media itself, so a
duration is known only when the sidecar states it or a committed watch batch
reported it from the player. Tags without seconds cover the whole item.

Fingerprints are content hashes (md5 for duplicate groups, sha256) cached by size
and mtime; the ``updated`` token combines the sidecar mtime and the signals
revision, so a rating never moves the ranking identity.

Signals are authoritative rows in the state database: rating, engagement count and
the actual watch intervals committed through ``commit_watch``. Reads never write.
Ratings and engagement change only through ``read_current``/``apply_change``.

Feature spaces are ``<state>/spaces/<space>.npz`` files with ``keys``, ``matrix``,
optional timed ``window_*`` rows (only with real timestamps) and a ``meta`` JSON string naming
their provenance. A space produced by ``build_text_hash_space`` encodes sidecar
tag words, never pixels or audio, and says so in its meta.
"""
from __future__ import annotations

import errno
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Sequence

import numpy as np

from feedloop.slots import MissingKeys
from feedloop.taste import DEFAULT_KINDS

VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mkv", ".mov", ".m4v"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})
SIDECAR_SUFFIX = ".json"
GENERATED_SUFFIX = ".generated.json"
DEFAULT_TAG_CATEGORY = "general"
TEXT_HASH_DIM = 64
SCAN_TTL_S = 1.0

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS items(kind TEXT NOT NULL, id INTEGER NOT NULL, relpath TEXT NOT NULL UNIQUE,"
    " first_seen_ts REAL NOT NULL, PRIMARY KEY(kind, id))",
    "CREATE TABLE IF NOT EXISTS fingerprints(relpath TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,"
    " md5 TEXT NOT NULL, sha256 TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS tags(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, category TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS signals(kind TEXT NOT NULL, id INTEGER NOT NULL, rating INTEGER,"
    " engagement_count INTEGER NOT NULL DEFAULT 0, updated_ts REAL NOT NULL, PRIMARY KEY(kind, id))",
    "CREATE TABLE IF NOT EXISTS meta(name TEXT PRIMARY KEY, value REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS watch_steps(event_id TEXT PRIMARY KEY, stream_session_id TEXT NOT NULL, item_id INTEGER NOT NULL, type TEXT NOT NULL,"
    " position REAL NOT NULL, occurred_at REAL NOT NULL, duration REAL NOT NULL, playback_rate REAL NOT NULL, previous_event_id TEXT)",
    "CREATE TABLE IF NOT EXISTS watch(step_id TEXT PRIMARY KEY, kind TEXT NOT NULL, id INTEGER NOT NULL,"
    " stream_session_id TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL, at_ts REAL NOT NULL, duration_s REAL NOT NULL)",
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _kind_of(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return None


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _utc_day(ts: float) -> int:
    return int(ts // 86400)


def sidecar_path_reason(path: Path) -> str | None:
    """None when ``path`` can be probed; otherwise the named reason it cannot be, so a
    media name near the filesystem's limit never aborts a scan or a sidecar write."""
    try:
        path.is_file()
        return None
    except OSError as exc:
        return "sidecar_path_too_long" if exc.errno == errno.ENAMETOOLONG else "sidecar_unreadable"


class TextHashEncoder:
    """A deterministic bag-of-words hashing encoder. It measures nothing about media: a
    vector is the normalized sum of hashed word signatures, so text queries match items
    whose sidecar tags share words. Compatible only with spaces built by
    ``build_text_hash_space`` with the same dimension."""

    def __init__(self, spaces: Sequence[str], dim: int = TEXT_HASH_DIM):
        self.spaces_supported = frozenset(spaces)
        self.dim = int(dim)

    def encode(self, space: str, text: str):
        if space not in self.spaces_supported:
            return None
        vector = text_hash_vector(text, self.dim)
        return None if vector is None else vector


def text_hash_vector(text: str, dim: int = TEXT_HASH_DIM):
    words = [w for w in "".join(c.lower() if c.isalnum() else " " for c in str(text)).split() if w]
    if not words:
        return None
    out = np.zeros(dim, dtype=np.float32)
    for word in words:
        seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        out += rng.normal(size=dim).astype(np.float32)
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else None


class FilesystemSource:
    """One object filling the Catalog, Signals and FeatureSpaces slots over a folder."""

    def __init__(self, folder, state_dir, *, kinds=DEFAULT_KINDS, media_prefix="/media/", clock=time.time, scan_ttl_s=SCAN_TTL_S):
        self.folder = Path(folder).resolve()
        if not self.folder.is_dir():
            raise ValueError("media folder does not exist")
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "source.sqlite"
        self.spaces_dir = self.state_dir / "spaces"
        self.kinds = tuple(kinds)
        self.media_prefix = media_prefix
        self.clock = clock
        self.scan_ttl_s = scan_ttl_s
        self._scan: dict | None = None
        self._scanned_at = -math.inf
        with self._connection(write=True) as conn:
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.execute("INSERT OR IGNORE INTO meta VALUES('signals_written_ts', ?)", (self.clock(),))

    # --------------------------------------------------------------- storage
    @contextmanager
    def _connection(self, write=False):
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write:
                conn.execute("COMMIT")
        except BaseException:
            if write:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _mark_written(self, conn):
        conn.execute("UPDATE meta SET value=? WHERE name='signals_written_ts'", (self.clock(),))

    def _signals_revision(self, conn) -> str:
        row = conn.execute("SELECT count(*), coalesce(max(updated_ts), 0) FROM signals").fetchone()
        watch = conn.execute("SELECT count(*), coalesce(max(at_ts), 0) FROM watch").fetchone()
        return _digest([tuple(row), tuple(watch)])

    # ------------------------------------------------------------------ scan
    def _walk(self):
        found = []
        for root, dirs, files in os.walk(self.folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not Path(root, d).is_symlink())
            for name in sorted(files):
                path = Path(root, name)
                if name.startswith(".") or path.is_symlink() or _kind_of(path) is None:
                    continue
                found.append(path.relative_to(self.folder).as_posix())
        return found

    def _fingerprint(self, conn, relpath: str, stat: os.stat_result):
        row = conn.execute("SELECT * FROM fingerprints WHERE relpath=?", (relpath,)).fetchone()
        if row and row["size"] == stat.st_size and row["mtime_ns"] == stat.st_mtime_ns:
            return row["md5"], row["sha256"]
        md5, sha = hashlib.md5(), hashlib.sha256()
        with open(self.folder / relpath, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                md5.update(chunk)
                sha.update(chunk)
        conn.execute("INSERT INTO fingerprints VALUES(?,?,?,?,?) ON CONFLICT(relpath) DO UPDATE SET size=excluded.size,"
                     " mtime_ns=excluded.mtime_ns, md5=excluded.md5, sha256=excluded.sha256",
                     (relpath, stat.st_size, stat.st_mtime_ns, md5.hexdigest(), sha.hexdigest()))
        return md5.hexdigest(), sha.hexdigest()

    def refresh(self) -> dict:
        """Scan the folder, assign ids to new paths, refresh fingerprints, read sidecars."""
        relpaths = self._walk()
        with self._connection(write=True) as conn:
            known = {r["relpath"]: (r["kind"], r["id"]) for r in conn.execute("SELECT kind, id, relpath FROM items")}
            next_id = {kind: (conn.execute("SELECT coalesce(max(id), 0) FROM items WHERE kind=?", (kind,)).fetchone()[0] + 1)
                       for kind in self.kinds}
            now = self.clock()
            items = {}
            for relpath in relpaths:
                path = self.folder / relpath
                kind = _kind_of(path)
                if kind not in self.kinds:
                    continue
                if relpath not in known:
                    conn.execute("INSERT INTO items VALUES(?,?,?,?)", (kind, next_id[kind], relpath, now))
                    known[relpath] = (kind, next_id[kind])
                    next_id[kind] += 1
                key = known[relpath]
                stat = path.stat()
                md5, sha = self._fingerprint(conn, relpath, stat)
                sidecar, sidecar_mtime = self._sidecar(path)
                items[key] = {"relpath": relpath, "size": stat.st_size, "md5": md5, "sha256": sha,
                              "sidecar": sidecar, "sidecar_mtime_ns": sidecar_mtime}
            tag_ids = self._register_tags(conn, items)
            revision = self._signals_revision(conn)
            names = {int(r["id"]): (r["name"], r["category"]) for r in conn.execute("SELECT id, name, category FROM tags")}
        self._scan = {"items": items, "tag_ids": tag_ids, "tag_rows": names, "signals_revision": revision,
                      "change_token": _digest([sorted((f"{k[0]}:{k[1]}", v["sha256"], v["sidecar_mtime_ns"]) for k, v in items.items()), revision])}
        self._scanned_at = time.monotonic()
        return self._scan

    def _sidecar(self, path: Path):
        """The user's sidecar, plus tags from an extractor's separate generated file. The
        generated file is read only, never merged back into the user's own sidecar. A
        sidecar path the filesystem refuses (the media name is near the length limit)
        reads as absent and carries its named reason; the item itself stays catalogued."""
        data, mtime = {}, 0
        sidecar, generated = Path(str(path) + SIDECAR_SUFFIX), Path(str(path) + GENERATED_SUFFIX)
        reason = sidecar_path_reason(sidecar) or sidecar_path_reason(generated)
        if reason:
            return {"_reason": reason}, None
        if sidecar.is_file():
            mtime = sidecar.stat().st_mtime_ns
            try:
                loaded = json.loads(sidecar.read_text(encoding="utf-8"))
                data = loaded if isinstance(loaded, dict) else {"_invalid": True}
            except (ValueError, OSError):
                data = {"_invalid": True}
        if generated.is_file():
            mtime = max(mtime, generated.stat().st_mtime_ns)
            try:
                extra = json.loads(generated.read_text(encoding="utf-8"))
                if isinstance(extra, dict) and isinstance(extra.get("tags"), list):
                    data = {**data, "tags": list(data.get("tags") or []) + [t for t in extra["tags"] if isinstance(t, dict)]}
            except (ValueError, OSError):
                pass
        return data, (mtime or None)

    @staticmethod
    def _sidecar_tags(sidecar) -> list[tuple[str, str, float | None]]:
        rows = []
        seconds_map = sidecar.get("tag_seconds") if isinstance(sidecar.get("tag_seconds"), dict) else {}
        for entry in sidecar.get("tags") or []:
            if isinstance(entry, str):
                name, category, seconds = entry, DEFAULT_TAG_CATEGORY, seconds_map.get(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
                name = entry["name"]
                category = entry.get("category") if isinstance(entry.get("category"), str) else DEFAULT_TAG_CATEGORY
                seconds = entry.get("seconds", seconds_map.get(name))
            else:
                continue
            name = name.strip()
            if not name:
                continue
            value = float(seconds) if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds > 0 else None
            rows.append((name, category.strip().lower() or DEFAULT_TAG_CATEGORY, value))
        return rows

    @staticmethod
    def _sidecar_segments(sidecar) -> list[tuple[float, list[str]]]:
        """Author-supplied timed segments: ``[{"start_s", "tags": [...]}]``; only finite,
        non-negative starts with at least one tag count. These are the only real window
        timestamps the source knows."""
        out = []
        for entry in sidecar.get("segments") or []:
            if not isinstance(entry, dict):
                continue
            start = entry.get("start_s")
            tags = [t.strip() for t in entry.get("tags") or [] if isinstance(t, str) and t.strip()]
            if isinstance(start, (int, float)) and not isinstance(start, bool) and math.isfinite(start) and start >= 0 and tags:
                out.append((float(start), tags))
        return sorted(out)

    def _register_tags(self, conn, items):
        ids = {r["name"]: int(r["id"]) for r in conn.execute("SELECT id, name FROM tags")}
        for entry in items.values():
            for name, category, _ in self._sidecar_tags(entry["sidecar"]):
                if name not in ids:
                    cursor = conn.execute("INSERT INTO tags(name, category) VALUES(?,?)", (name, category))
                    ids[name] = int(cursor.lastrowid)
        return ids

    def _scanned(self):
        if self._scan is None or time.monotonic() - self._scanned_at > self.scan_ttl_s:
            self.refresh()
        return self._scan

    # --------------------------------------------------------------- catalog
    def _duration(self, key, sidecar, conn=None) -> tuple[float | None, str | None]:
        value = sidecar.get("duration_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            return float(value), "sidecar"
        if key[0] != self.kinds[0]:
            return None, None
        query = "SELECT duration_s FROM watch WHERE kind=? AND id=? ORDER BY at_ts DESC LIMIT 1"
        if conn is not None:
            row = conn.execute(query, key).fetchone()
        else:
            with self._connection() as own:
                row = own.execute(query, key).fetchone()
        return (float(row[0]), "player") if row and row[0] and row[0] > 0 else (None, None)

    def _item(self, key, entry, signals_row, conn) -> dict:
        sidecar = entry["sidecar"]
        duration, duration_source = self._duration(key, sidecar, conn)
        tags = self._sidecar_tags(sidecar)
        title = sidecar.get("title") if isinstance(sidecar.get("title"), str) and sidecar["title"].strip() else Path(entry["relpath"]).stem
        contributors = [str(c) for c in sidecar.get("contributors") or [] if isinstance(c, str) and c.strip()]
        updated = f"{entry['sidecar_mtime_ns'] or 0}:{signals_row['updated_ts'] if signals_row else 0}"
        return {"kind": key[0], "id": key[1], "title": title, "duration_s": duration, "duration_source": duration_source,
                "media_url": f"{self.media_prefix}{key[0]}/{key[1]}", "tags": [name for name, _, _ in tags],
                "contributor_ids": contributors, "updated": updated, "path": entry["relpath"],
                "files": [{"fingerprints": [{"type": "md5", "value": entry["md5"]}, {"type": "sha256", "value": entry["sha256"]}]}],
                "sidecar_valid": not sidecar.get("_invalid", False),
                "sidecar_reason": "sidecar_invalid" if sidecar.get("_invalid") else sidecar.get("_reason")}

    def sidecar_reasons(self, keys) -> dict:
        """Per catalogued item, the named reason its sidecar could not be read (None when it
        was read or does not exist); the honest status the served card shows."""
        scan = self._scanned()
        return {key: ("sidecar_invalid" if scan["items"][key]["sidecar"].get("_invalid") else scan["items"][key]["sidecar"].get("_reason"))
                for key in keys if key in scan["items"]}

    def _rows(self, keys):
        scan = self._scanned()
        with self._connection() as conn:
            signals = {(r["kind"], r["id"]): dict(r) for r in conn.execute("SELECT * FROM signals")}
            return [self._item(key, scan["items"][key], signals.get(key), conn) for key in keys], scan["change_token"]

    def enumerate(self, kinds, page, page_size, *, timeout_s=30):
        scan = self._scanned()
        keys = sorted(key for key in scan["items"] if key[0] in kinds)
        start = (int(page) - 1) * int(page_size)
        items, token = self._rows(keys[start:start + int(page_size)])
        return {"items": items, "total": len(keys), "change_token": token}

    def fetch(self, keys, *, timeout_s=30):
        scan = self._scanned()
        wanted = sorted(set(keys))
        missing = [key for key in wanted if key not in scan["items"]]
        if missing:
            raise MissingKeys(missing)
        items, token = self._rows(wanted)
        return {"items": items, "total": len(items), "change_token": token}

    def features(self, keys):
        scan = self._scanned()
        out = {}
        for key in keys:
            entry = scan["items"].get(key)
            if entry is None:
                continue
            tags = self._sidecar_tags(entry["sidecar"])
            if not tags:
                continue
            duration, _ = self._duration(key, entry["sidecar"])
            whole = float(duration) if key[0] == self.kinds[0] and duration else 1.0
            seconds, categories = {}, {}
            for name, category, value in tags:
                tag_id = scan["tag_ids"][name]
                seconds[tag_id] = seconds.get(tag_id, 0.0) + (value if value is not None else whole)
                categories[tag_id] = scan["tag_rows"][tag_id][1]
            out[key] = {"tag_seconds": seconds, "watched_tag_seconds": None, "tag_categories": categories}
        return out

    def tag_names(self):
        scan = self._scanned()
        return {tag_id: name for tag_id, (name, _category) in scan["tag_rows"].items()}

    # --------------------------------------------------------------- signals
    def read(self, keys=None):
        """Authoritative rows as of the last write to them; ``observed_at`` is that write
        time, never the moment of reading."""
        scan = self._scanned()
        with self._connection() as conn:
            observed_at = float(conn.execute("SELECT value FROM meta WHERE name='signals_written_ts'").fetchone()[0])
            wanted = None if keys is None else set(keys)
            rows = {}
            for row in conn.execute("SELECT * FROM signals"):
                key = (row["kind"], row["id"])
                if wanted is None or key in wanted:
                    rows[key] = {"rating": row["rating"], "engagement_count": int(row["engagement_count"])}
            watch = {}
            for row in conn.execute("SELECT * FROM watch ORDER BY at_ts"):
                key = (row["kind"], row["id"])
                if (wanted is None or key in wanted) and key[0] == self.kinds[0]:
                    watch.setdefault(key, []).append(row)
        for key, steps in watch.items():
            intervals = _merge([(float(s["start_s"]), float(s["end_s"])) for s in steps if s["end_s"] > s["start_s"]])
            if not intervals:
                continue
            rows.setdefault(key, {"rating": None, "engagement_count": 0})["watch"] = {
                "watched_s": float(sum(b - a for a, b in intervals)), "last_at": float(max(s["at_ts"] for s in steps)),
                "visit_days": sorted({_utc_day(float(s["at_ts"])) for s in steps}), "intervals": intervals}
        return {"observed_at": observed_at, "rows": rows}

    def read_current(self, key):
        """Feedback authority read: the persisted rating and engagement count."""
        if key[0] not in self.kinds or key not in self._scanned()["items"]:
            return {"status": "error", "error_code": "unknown_item"}
        with self._connection() as conn:
            row = conn.execute("SELECT rating, engagement_count FROM signals WHERE kind=? AND id=?", key).fetchone()
        return {"status": "ok", "rating100": row["rating"] if row else None, "engagement_count": int(row["engagement_count"]) if row else 0}

    def apply_change(self, key, change):
        """Feedback authority write, applied exactly once per call: a rating value or a count delta."""
        if key[0] not in self.kinds or key not in self._scanned()["items"]:
            return {"status": "conflict"}
        with self._connection(write=True) as conn:
            row = conn.execute("SELECT rating, engagement_count FROM signals WHERE kind=? AND id=?", key).fetchone()
            rating = row["rating"] if row else None
            count = int(row["engagement_count"]) if row else 0
            if change["action"] == "rating":
                value = change["rating100"]
                if value is not None and (type(value) is not int or not 0 <= value <= 100):
                    return {"status": "conflict"}
                rating = value
            else:
                delta = change["delta"]
                if delta not in (-1, 1) or count + delta < 0:
                    return {"status": "conflict"}
                count += delta
            conn.execute("INSERT INTO signals VALUES(?,?,?,?,?) ON CONFLICT(kind, id) DO UPDATE SET rating=excluded.rating,"
                         " engagement_count=excluded.engagement_count, updated_ts=excluded.updated_ts", (*key, rating, count, self.clock()))
            self._mark_written(conn)
        self._scan = None
        return {"status": "confirmed", "rating100": rating, "engagement_count": count}

    def commit_watch(self, rows: Sequence[Mapping[str, Any]], *, received_at: float, dry_run: bool = False) -> dict:
        """Persist a player batch as authoritative watch history, all or nothing.

        Every row is validated against the persisted stream before anything is written:
        shape, clock and position bounds, continuity with the stored previous step of the
        same stream and item (no fork, no regression), a bounded elapsed time (0 < elapsed
        <= 15 s) and a position advance no faster than playback allows. A batch with any
        invalid row is rejected and writes nothing. ``view_start`` and ``view_seek`` are
        anchors; ``view_progress``/``view_pause``/``view_complete`` credit the interval
        since the previous step. Steps are keyed by event id, so a replay stores nothing.
        ``dry_run`` validates and reports without writing (status ``valid``)."""
        ordered = sorted(rows, key=lambda r: (r.get("occurred_at", 0) if isinstance(r.get("occurred_at"), (int, float)) else 0, str(r.get("id", ""))))
        with self._connection(write=True) as conn:
            pending: dict[str, dict] = {}
            intervals = []
            for row in ordered:
                verdict = self._validate_step(conn, row, pending, received_at)
                if verdict[0] != "ok":
                    return {"status": "rejected", "error_code": verdict[1], "stored": 0}
                if verdict[1] == "duplicate":
                    continue
                step = verdict[2]
                pending[step["event_id"]] = step
                if step["credit"] is not None:
                    intervals.append((step["event_id"], self.kinds[0], step["item_id"], step["stream_session_id"], step["credit"][0], step["credit"][1],
                                      step["occurred_at"], step["duration"]))
            if dry_run:
                return {"status": "valid", "stored": len(pending), "credited": len(intervals), "received_at": received_at}
            for step in pending.values():
                conn.execute("INSERT INTO watch_steps VALUES(?,?,?,?,?,?,?,?,?)", (step["event_id"], step["stream_session_id"], step["item_id"], step["type"],
                             step["position"], step["occurred_at"], step["duration"], step["playback_rate"], step["previous_event_id"]))
            for interval in intervals:
                conn.execute("INSERT INTO watch VALUES(?,?,?,?,?,?,?,?)", interval)
            if pending:
                self._mark_written(conn)
        self._scan = None
        return {"status": "committed", "stored": len(pending), "credited": len(intervals), "received_at": received_at}

    def _validate_step(self, conn, row, pending, received_at):
        """('ok', 'new', step) | ('ok', 'duplicate') | ('rejected', code)."""
        def number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or not row["id"] or not isinstance(row.get("stream_session_id"), str):
            return ("rejected", "watch_row_invalid")
        event_id, stream, item_id = row["id"], row["stream_session_id"], row.get("item_id")
        kind, position, duration, at = row.get("type"), row.get("position"), row.get("duration"), row.get("occurred_at")
        rate = row.get("playback_rate", 1)
        previous_id = row.get("previous_event_id")
        if kind not in ("view_start", "view_seek", "view_progress", "view_pause", "view_complete"):
            return ("rejected", "watch_type_invalid")
        if type(item_id) is not int or item_id <= 0 or not all(number(v) for v in (position, duration, at, rate)):
            return ("rejected", "watch_row_invalid")
        if not (duration > 0 and 0 <= position <= duration and 0 < rate <= 4 and at <= received_at):
            return ("rejected", "watch_clock_or_position_invalid")
        if previous_id is not None and not isinstance(previous_id, str):
            return ("rejected", "watch_row_invalid")
        existing = conn.execute("SELECT 1 FROM watch_steps WHERE event_id=?", (event_id,)).fetchone()
        if existing or event_id in pending:
            return ("ok", "duplicate")
        latest = conn.execute("SELECT event_id, occurred_at FROM watch_steps WHERE stream_session_id=? AND item_id=? ORDER BY occurred_at DESC LIMIT 1",
                              (stream, item_id)).fetchone()
        latest_id, latest_at = (latest[0], latest[1]) if latest else (None, None)
        for step in pending.values():
            if step["stream_session_id"] == stream and step["item_id"] == item_id and (latest_at is None or step["occurred_at"] > latest_at):
                latest_id, latest_at = step["event_id"], step["occurred_at"]
        if latest_at is not None and at <= latest_at:
            return ("rejected", "watch_stream_clock_regressed")
        step = {"event_id": event_id, "stream_session_id": stream, "item_id": item_id, "type": kind, "position": float(position), "occurred_at": float(at),
                "duration": float(duration), "playback_rate": float(rate), "previous_event_id": previous_id, "credit": None}
        if kind == "view_start":
            return ("ok", "new", step)
        if previous_id is None or previous_id != latest_id:
            return ("rejected", "watch_chain_gap" if previous_id is None or latest_id is None else "watch_chain_fork")
        previous = pending.get(previous_id)
        if previous is None:
            found = conn.execute("SELECT type, position, occurred_at, duration, playback_rate FROM watch_steps WHERE event_id=?", (previous_id,)).fetchone()
            previous = {"type": found[0], "position": found[1], "occurred_at": found[2], "duration": found[3], "playback_rate": found[4]}
        if previous["duration"] != step["duration"] or previous["playback_rate"] != step["playback_rate"]:
            return ("rejected", "watch_chain_scope_changed")
        elapsed = step["occurred_at"] - previous["occurred_at"]
        if not 0 < elapsed <= 15:
            return ("rejected", "watch_chain_clock_gap")
        if kind == "view_seek":
            return ("ok", "new", step)
        if previous["type"] not in ("view_start", "view_progress", "view_seek"):
            return ("rejected", "watch_not_playing")
        delta = step["position"] - previous["position"]
        if not 0 <= delta <= elapsed * step["playback_rate"] + .5:
            return ("rejected", "watch_seek_or_rate_unproven")
        if delta > 0:
            step["credit"] = (previous["position"], step["position"])
        return ("ok", "new", step)

    def media_path(self, key) -> Path | None:
        """The catalogued file for one present item key, or None. Only catalogued media is
        ever served; sidecars, state files and anything else are not items."""
        entry = self._scanned()["items"].get(key)
        return None if entry is None else self.folder / entry["relpath"]

    # --------------------------------------------------------- feature spaces
    def _space_path(self, space: str) -> Path:
        if not space or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in space):
            raise ValueError("invalid space name")
        return self.spaces_dir / (space + ".npz")

    def _load_space(self, space):
        path = self._space_path(space)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                loaded = {name: data[name] for name in data.files}
            meta = json.loads(str(loaded["meta"][0])) if "meta" in loaded else {}
        except (OSError, ValueError, KeyError, IndexError):
            return None
        return loaded, meta

    def spaces(self):
        if not self.spaces_dir.is_dir():
            return []
        return sorted(p.stem for p in self.spaces_dir.glob("*.npz"))

    def matrix(self, space):
        loaded = self._load_space(space)
        if loaded is None:
            return None
        data, _meta = loaded
        present = self._scanned()["items"]
        keys = [(k, int(i)) for k, _, i in (str(v).partition(":") for v in data["keys"])]
        rows = [n for n, key in enumerate(keys) if key in present]
        matrix = np.asarray(data["matrix"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(keys) or not np.isfinite(matrix).all():
            return None
        return [keys[n] for n in rows], matrix[rows]

    def revision(self, space):
        loaded = self._load_space(space)
        if loaded is None:
            return None
        _data, meta = loaded
        return meta.get("revision")

    def windows(self, space):
        loaded = self._load_space(space)
        if loaded is None or "window_keys" not in loaded[0]:
            return None
        data, _meta = loaded
        present = self._scanned()["items"]
        keys = [(k, int(i)) for k, _, i in (str(v).partition(":") for v in data["window_keys"])]
        rows = [n for n, key in enumerate(keys) if key in present]
        return [keys[n] for n in rows], np.asarray(data["window_times"], dtype=np.float32)[rows], np.asarray(data["window_matrix"], dtype=np.float32)[rows]

    def space_meta(self, space):
        loaded = self._load_space(space)
        return None if loaded is None else loaded[1]

    def write_space(self, space: str, keys, matrix, *, meta: Mapping[str, Any], windows=None):
        """Store one feature space. ``meta`` must name its ``provenance`` and ``window_scope``
        (``"timed"`` with real per-window timestamps, else ``"none"``: means only, no
        invented window rows); the revision is the digest of the stored arrays and meta."""
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(keys) or not np.isfinite(matrix).all():
            raise ValueError("invalid feature matrix")
        if not isinstance(meta.get("provenance"), str) or meta.get("window_scope") not in ("none", "timed"):
            raise ValueError("space meta needs provenance and window_scope")
        if (meta["window_scope"] == "timed") != (windows is not None):
            raise ValueError("timed windows need real timestamps; a means-only space has none")
        arrays = {"keys": np.array([f"{k}:{i}" for k, i in keys]), "matrix": matrix}
        if windows is not None:
            window_keys, times, window_matrix = windows
            window_matrix = np.asarray(window_matrix, dtype=np.float32)
            times = np.asarray(times, dtype=np.float32)
            if window_matrix.ndim != 2 or window_matrix.shape[0] != len(window_keys) != len(times) or window_matrix.shape[1] != matrix.shape[1]:
                raise ValueError("invalid window rows")
            arrays.update(window_keys=np.array([f"{k}:{i}" for k, i in window_keys]), window_times=times, window_matrix=window_matrix)
        stored = {**meta, "dim": int(matrix.shape[1]), "items": int(matrix.shape[0])}
        stored["revision"] = _digest([stored.get("provenance"), stored.get("window_scope"), [a.tobytes().hex() if a.dtype != object else a.tolist() for a in arrays.values()]])
        arrays["meta"] = np.array([json.dumps(stored, sort_keys=True)])
        self.spaces_dir.mkdir(parents=True, exist_ok=True)
        path = self._space_path(space)
        temporary = path.with_suffix(".npz.tmp")
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
        return stored["revision"]


def build_text_hash_space(source: FilesystemSource, space: str, *, dim: int = TEXT_HASH_DIM) -> dict:
    """Write a space whose vectors hash each item's sidecar tag words. It is labelled as
    text-derived: it describes the sidecar, not the pixels or audio of the media. Window
    rows exist only for items whose sidecar supplies timed ``segments``; their timestamps
    are the author's segment starts, never a measured moment. Without segments the space
    has means only and text search reports no-feature."""
    scan = source.refresh()
    keys, vectors, window_keys, window_times, window_vectors = [], [], [], [], []
    for key in sorted(scan["items"]):
        sidecar = scan["items"][key]["sidecar"]
        tags = [name for name, _, _ in source._sidecar_tags(sidecar)]
        vector = text_hash_vector(" ".join(tags), dim)
        if vector is None:
            continue
        keys.append(key)
        vectors.append(vector)
        if key[0] == source.kinds[0]:
            for start, segment_tags in source._sidecar_segments(sidecar):
                window = text_hash_vector(" ".join(segment_tags), dim)
                if window is not None:
                    window_keys.append(key)
                    window_times.append(start)
                    window_vectors.append(window)
    matrix = np.stack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)
    timed = bool(window_keys)
    meta = {"provenance": "text-hash of sidecar tag words; not a measurement of the media"
            + ("; window times are the sidecar's own segment starts" if timed else ""), "encoder": "TextHashEncoder",
            "window_scope": "timed" if timed else "none", "built_at": source.clock()}
    windows = (window_keys, np.asarray(window_times, dtype=np.float32), np.stack(window_vectors)) if timed else None
    revision = source.write_space(space, keys, matrix, meta=meta, windows=windows)
    return {"space": space, "items": len(keys), "windows": len(window_keys), "revision": revision, "provenance": meta["provenance"]}


__all__ = ["FilesystemSource", "TextHashEncoder", "text_hash_vector", "build_text_hash_space", "sidecar_path_reason", "VIDEO_EXTENSIONS", "IMAGE_EXTENSIONS",
           "TEXT_HASH_DIM", "DEFAULT_TAG_CATEGORY", "SIDECAR_SUFFIX", "GENERATED_SUFFIX"]

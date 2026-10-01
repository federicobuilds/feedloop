"""The six host protocols. Every other module consumes these and nothing host-specific.

Identity: an item is ``ItemKey = (kind, id)``; kind is one of the configured
kinds (primary first) and id a positive integer, never a boolean. An id of 1 in
two kinds names two different items. Wire items carry separate ``kind`` and
``id`` fields; cursors carry ``after = "<kind>:<id>"``. Conversion happens at
the serving boundary; no unqualified string ids exist inside the package.

Reads are complete or unavailable. A page or keyed read either describes every
item it claims (exact ``total``) or raises; a partial page is a failure, never
a shorter catalog. Missing requested keys are reported with ``MissingKeys``,
whose ``keys`` must be a nonempty subset of the request; anything else is a
transport (``TransportError``, retried) or authority/validation failure
(``AuthorityError``/``ResponseError``, never retried).

Revisions: ``change_token`` and per-item ``updated`` tokens may move on any
metadata touch (a rating, a view). Ranking identity is derived by
``catalog.fingerprint_snapshot`` from item existence plus file fingerprints
only. ``FeatureSpaces.revision(space)`` is the committed feature revision;
``None`` means unavailable and disables dependent cache publication.

Rows: ``matrix(space)`` has one row per item (means); ``windows(space)``
repeats item keys, one row per time window, distinguished by ``times_s``.

Mutation: no read protocol changes anything. ``Annotator`` writes tags/links
only; ratings and engagement change through the feedback callbacks the Engine
is constructed with (``read_current``/``apply_change``), never through a slot.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

ItemKey = tuple[str, int]

DEFAULT_SPACE_ROLES = {"visual": "visual", "semantic": "semvisual", "voice": "audioembed", "sound": "audiomix"}


class TransportError(Exception):
    """The slot could not reach its authority (stall, connection loss). Retried."""


class AuthorityError(Exception):
    """The authority answered with a status or permission failure. Not retried."""


class ResponseError(Exception):
    """The authority answered with a malformed body, page or count. Not retried."""


class MissingKeys(Exception):
    """Some explicitly requested keys do not exist. ``keys`` is a frozenset of ItemKey."""

    def __init__(self, keys):
        super().__init__(frozenset(keys))
        self.keys = frozenset(keys)


@runtime_checkable
class Catalog(Protocol):
    """Required. Items with ``id``, ``kind``, ``title``, ``duration_s``, ``media_url``,
    ``tags`` (display strings), ``contributor_ids``, ``updated`` (opaque token) and
    ``files``: a list, ``[]`` for fileless items, each ``{"fingerprints": [{"type", "value"}, ...]}``.
    Weighted feature metadata is separate: ``features(keys)`` returns per key
    ``{"tag_seconds": {tag_id: s}, "watched_tag_seconds": {...} | None,
    "tag_categories": {tag_id: category}}`` with positive-integer tag ids.
    ``tag_names()`` maps known tag ids to display names."""

    def enumerate(self, kinds: Sequence[str], page: int, page_size: int, *, timeout_s: float = 30) -> Mapping[str, Any]:
        """{"items": [...], "total": int, "change_token": str} for one page (1-based)."""

    def fetch(self, keys: Sequence[ItemKey], *, timeout_s: float = 30) -> Mapping[str, Any]:
        """Complete keyed read, not paginated, any size: {"items", "total", "change_token"}."""

    def features(self, keys: Sequence[ItemKey]) -> Mapping[ItemKey, Mapping[str, Any]]:
        """Weighted tag coverage and categories by key; absent keys have no features."""

    def tag_names(self) -> Mapping[int, str]:
        """Known tag id -> name. Unknown ids stay unnamed."""


@runtime_checkable
class Signals(Protocol):
    """Required, read only. ``read(keys)`` returns ``{"observed_at": float, "rows": {key: row}}``.
    A row: ``rating`` (0-100 or None), ``engagement_count`` (int >= 0), and, only
    when the item was actually watched, ``watch = {"watched_s", "last_at",
    "visit_days": [utc_day_int, ...], "intervals": [(start, end), ...]}``.
    No watch history means no ``watch`` key, never a zero-second visit.

    Each row should also carry ``updated_at``: the epoch seconds of that item's last
    rating, engagement or watch change, never later than ``observed_at``. The tuner
    judges a past trial from current rows, so it needs to know which rows changed
    after the trial's evidence cutoff. When any returned row lacks ``updated_at`` the
    whole read falls back to the global ``observed_at`` as the cutoff: while the host
    keeps writing, ``observed_at`` stays newer than every cutoff and tuner evidence is
    unavailable. ``check_signals`` reports rows that break this contract."""

    def read(self, keys: Sequence[ItemKey] | None = None) -> Mapping[str, Any]:
        """Current authoritative rows and their observation time. ``None`` reads every row."""


@runtime_checkable
class FeatureSpaces(Protocol):
    """Required. Item mean vectors per space plus optional timestamped windows."""

    def spaces(self) -> Sequence[str]: ...

    def matrix(self, space: str) -> tuple[Sequence[ItemKey], np.ndarray] | None:
        """(id_index, matrix): rows aligned with keys; finite, compatible dimensions."""

    def revision(self, space: str) -> Any | None:
        """Committed feature revision, or None when unavailable."""

    def windows(self, space: str) -> tuple[Sequence[ItemKey], np.ndarray, np.ndarray] | None:
        """(item_keys, times_s, matrix) with repeated keys per window, or None."""


@runtime_checkable
class TextEncoder(Protocol):
    """Optional. ``encode(space, text)`` returns a vector compatible with that space or None."""

    def encode(self, space: str, text: str) -> np.ndarray | None: ...


@runtime_checkable
class IdentityLinks(Protocol):
    """Optional. Trusted contributor ids by key; these, not display ids, drive affinity."""

    def links(self, keys: Sequence[ItemKey] | None = None) -> Mapping[ItemKey, Sequence[str]]: ...


@runtime_checkable
class Annotator(Protocol):
    """Optional write-back. Never invoked by read, ranking, search or similar paths."""

    def write_tags(self, key: ItemKey, tags: Sequence[str]) -> Mapping[str, Any]: ...

    def write_links(self, key: ItemKey, contributor_ids: Sequence[str]) -> Mapping[str, Any]: ...


def item_key(kind, item_id, kinds) -> ItemKey:
    """Validate a qualified key against the configured kinds."""
    if kind not in kinds or type(item_id) is not int or item_id <= 0:
        raise ValueError("invalid item key")
    return kind, item_id


def wire_key(key: ItemKey) -> str:
    return f"{key[0]}:{key[1]}"


def parse_wire_key(value: str, kinds) -> ItemKey:
    kind, _, rest = str(value).partition(":")
    if kind not in kinds or not rest.isdigit():
        raise ValueError("invalid item key")
    return item_key(kind, int(rest), kinds)


def check_signals(signals: Signals, keys: Sequence[ItemKey] | None = None) -> list[str]:
    """Conformance problems in one ``signals.read(keys)``; an empty list means it conforms."""
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    current = signals.read(keys)
    observed_at = current.get("observed_at")
    problems = [] if finite(observed_at) else ["observed_at is not a finite number"]
    for key, row in sorted(current.get("rows", {}).items()):
        name = wire_key(key)
        if "updated_at" not in row:
            problems.append(f"{name}: missing updated_at")
        elif not finite(row["updated_at"]):
            problems.append(f"{name}: updated_at is not a finite number")
        elif finite(observed_at) and row["updated_at"] > observed_at:
            problems.append(f"{name}: updated_at is after observed_at")
        watch = row.get("watch") or {}
        values = {"rating": row.get("rating"), "engagement_count": row.get("engagement_count"),
                  "watch.watched_s": watch.get("watched_s"), "watch.last_at": watch.get("last_at")}
        problems += [f"{name}: {field} is not finite" for field, value in values.items()
                     if value is not None and not finite(value)]
    return problems


__all__ = [
    "ItemKey", "DEFAULT_SPACE_ROLES", "TransportError", "AuthorityError", "ResponseError", "MissingKeys",
    "Catalog", "Signals", "FeatureSpaces", "TextEncoder", "IdentityLinks", "Annotator",
    "item_key", "wire_key", "parse_wire_key", "check_signals",
]

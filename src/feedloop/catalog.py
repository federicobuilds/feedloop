"""Complete catalog reads and derived ranking identity over the Catalog slot.

Every page and count is validated; fileless items are included with
``files=[]``; a partial page is a failure, never a shorter catalog. Only the
transport failure class is retried (three attempts, one 30 s timeout each);
authority and malformed-response failures propagate on the first attempt.
Missing requested keys reported by the slot are removed and that keyed read
restarts; a report during unrestricted enumeration, an unexpected key, or a
mixed failure is fatal.
"""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

from feedloop.ranking import fingerprint_groups
from feedloop.slots import AuthorityError, MissingKeys, ResponseError, TransportError
from feedloop.taste import DEFAULT_KINDS

TRANSPORT_ATTEMPTS = 3
TRANSPORT_TIMEOUT_S = 30
PAGE_SIZE = 500


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def host_links(row):
    """The optional ``preview_url`` and ``open_url`` of a catalog row, kept only when each is an
    http(s) URL or a root-relative path; anything else is dropped."""
    links = {}
    for field in ("preview_url", "open_url"):
        value = row.get(field)
        if not isinstance(value, str):
            continue
        parts = urlsplit(value)
        absolute = parts.scheme.lower() in ("http", "https") and parts.netloc
        root_relative = not parts.scheme and not parts.netloc and value.startswith("/") and not value.startswith(("//", "/\\"))
        if absolute or root_relative:
            links[field] = value
    return links


def media_flags(row):
    """The optional ``animated_image`` flag of a catalog row, kept only when it is exactly
    boolean ``True``; any other value is dropped, so the client never trusts a truthy string."""
    return {"animated_image": True} if row.get("animated_image") is True else {}


class CatalogMissing(Exception):
    """Some explicitly requested keys are absent, nothing else. args[0] is the frozenset of ItemKey."""


def _attempt(call):
    """Retry only a stalled transport; status and body failures fail at once."""
    for attempt in range(TRANSPORT_ATTEMPTS):
        try:
            result = call()
            break
        except TransportError:
            if attempt + 1 == TRANSPORT_ATTEMPTS:
                raise
        except MissingKeys as absent:
            raise CatalogMissing(frozenset(absent.keys)) from None
        except AuthorityError:
            raise
        except ResponseError:
            raise RuntimeError("catalog_enumeration_failed") from None
    if not isinstance(result, dict):
        raise RuntimeError("catalog_enumeration_failed")
    return result


def read_catalog(source, keys=None, *, kinds=DEFAULT_KINDS):
    """Validate every page and count; include fileless items, never hydration stubs.

    keys=None enumerates every configured kind through ``source.enumerate``;
    otherwise ``source.fetch`` performs one complete keyed read per kind."""
    result = []
    for kind in kinds:
        ids = None if keys is None else sorted({id_ for k, id_ in keys if k == kind})
        wanted = None if ids is None else set(ids)
        if ids == []:
            continue
        seen, expected, page, rows = set(), None, 1, []
        while True:
            try:
                if ids is None:
                    payload = _attempt(lambda: source.enumerate((kind,), page, PAGE_SIZE, timeout_s=TRANSPORT_TIMEOUT_S))
                else:
                    payload = _attempt(lambda: source.fetch([(kind, id_) for id_ in ids], timeout_s=TRANSPORT_TIMEOUT_S))
            except CatalogMissing as absent:
                missing = {id_ for k, id_ in (absent.args[0] if absent.args else ()) if k == kind}
                reported = set(absent.args[0]) if absent.args else set()
                if ids is None or not reported or any(k != kind for k, _ in reported) or not missing <= set(ids):
                    raise RuntimeError("catalog_enumeration_failed") from None
                ids = [id_ for id_ in ids if id_ not in missing]
                wanted = set(ids)
                seen, expected, page, rows = set(), None, 1, []
                if not ids:
                    break
                continue
            if type(payload.get("total")) is not int or payload["total"] < 0:
                raise RuntimeError("catalog_enumeration_unknown")
            count, batch = payload["total"], payload.get("items")
            if expected is not None and expected != count:
                raise RuntimeError("catalog_enumeration_changed")
            expected = count
            expected_size = count if ids is not None else min(PAGE_SIZE, max(0, count - len(seen)))
            if not isinstance(batch, list) or len(batch) != expected_size or (ids is not None and count != len(ids)):
                raise RuntimeError("catalog_enumeration_partial")
            for row in batch:
                if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0 or row.get("kind") != kind:
                    raise RuntimeError("catalog_item_invalid")
                id_, attached = row["id"], row.get("files")
                if (id_ in seen or (wanted is not None and id_ not in wanted) or not row.get("updated")
                        or not isinstance(attached, list) or any(not isinstance(f, dict) or
                        not isinstance(f.get("fingerprints"), list) for f in attached)):
                    raise RuntimeError("catalog_enumeration_partial")
                seen.add(id_)
                rows.append({"kind": kind, "id": id_, "updated": row["updated"], "files": attached})
            if len(seen) == expected:
                break
            page += 1
        result.extend(rows)
    return sorted(result, key=lambda row: (row["kind"], row["id"]))


def ranking_identity(rows):
    """Which items exist and which files they carry; update tokens are deliberately excluded."""
    return [{"kind": row["kind"], "id": row["id"], "files": row["files"]} for row in rows]


class SnapshotMemo:
    """The last pin fingerprint_snapshot derived, reused while the ranking identity is equal.

    Invalidation contract: every call still makes both validated reads; only the derivation
    (revision digest, duplicate groups, present set) is reused, and only when the agreed
    identity, keys and kinds equal the memo's. Any item, file or fingerprint change is a
    different identity and derives a new pin. change_token is not trusted as a fence: the
    slot contract lets it stay put across changes. The pin returned is shared: read it, never
    mutate it. clear() drops it (Engine.reset_caches)."""

    def __init__(self):
        self.entry = None

    def clear(self):
        self.entry = None


def fingerprint_snapshot(source, keys, *, attempts=3, kinds=DEFAULT_KINDS, memo=None):
    """Two consecutive reads of the requested keys that agree on ranking identity.

    A read whose items or files differ is retried a bounded number of times;
    only a set that keeps changing is reported as changed."""
    keys = tuple(sorted(set(keys)))
    previous = ranking_identity(read_catalog(source, keys, kinds=kinds))
    for _ in range(max(1, attempts)):
        current = ranking_identity(read_catalog(source, keys, kinds=kinds))
        if current == previous:
            entry = memo.entry if memo is not None else None
            if entry is not None and entry[0] == (keys, tuple(kinds)) and entry[1] == current:
                return entry[2]
            text = json.dumps(current, sort_keys=True, separators=(",", ":"), allow_nan=False)
            pin = {"revision": hashlib.sha256(text.encode()).hexdigest(), "groups": fingerprint_groups(current, kinds=kinds),
                   "present": {(r["kind"], r["id"]) for r in current}, "keys": keys}
            if memo is not None:
                # a decoded copy: slots may hand out rows that share file dicts with their own state
                memo.entry = ((keys, tuple(kinds)), json.loads(text), pin)
            return pin
        previous = current
    raise RuntimeError("catalog_enumeration_changed")


__all__ = ["TRANSPORT_ATTEMPTS", "TRANSPORT_TIMEOUT_S", "PAGE_SIZE", "digest", "CatalogMissing",
           "read_catalog", "ranking_identity", "SnapshotMemo", "fingerprint_snapshot"]

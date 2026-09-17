"""Complete keyed reads, stale-key exclusion, transport retry and derived ranking identity."""
import pytest
from unittest.mock import Mock

from feedloop import catalog
from feedloop.slots import AuthorityError, MissingKeys, ResponseError, TransportError
from fl2_helpers import MemoryCatalog, catalog_row, md5_file


class Source:
    """Scripted Catalog slot: ``fetch`` is a callable (keys) -> payload or raising."""

    def __init__(self, fetch=None, enumerate=None):
        self._fetch, self._enumerate = fetch, enumerate
        self.calls = []

    def fetch(self, keys, *, timeout_s=30):
        self.calls.append(("fetch", list(keys), timeout_s))
        return self._fetch(list(keys))

    def enumerate(self, kinds, page, page_size, *, timeout_s=30):
        self.calls.append(("enumerate", tuple(kinds), page, timeout_s))
        return self._enumerate(kinds, page, page_size)

    def features(self, keys):
        return {}

    def tag_names(self):
        return {}


def complete(keys, updated="t"):
    return {"items": [catalog_row(k, i, updated=updated) for k, i in keys], "total": len(keys), "change_token": "c"}


@pytest.mark.parametrize("kind", ["video", "image"])
@pytest.mark.parametrize("count", [499, 501, 2339])
def test_complete_keyed_reads_accept_any_size(kind, count):
    keys = [(kind, index) for index in range(1, count + 1)]
    source = Source(fetch=lambda ks: complete(ks) if ks == keys else pytest.fail("unexpected keys"))
    rows = catalog.read_catalog(source, keys, kinds=(kind,))
    assert [(row["kind"], row["id"]) for row in rows] == keys
    assert source.calls == [("fetch", keys, 30)]


@pytest.mark.parametrize("kind", ["video", "image"])
def test_stale_ids_are_reported_absent_not_fatal(kind):
    keys = [(kind, index) for index in range(1, 8)]
    calls = []

    def fetch(ks):
        calls.append([i for _, i in ks])
        if (kind, 4) in ks:
            raise MissingKeys({(kind, 4)})
        return complete(ks)

    snapshot = catalog.fingerprint_snapshot(Source(fetch=fetch), keys, kinds=(kind,))
    assert snapshot["present"] == {(kind, i) for i in (1, 2, 3, 5, 6, 7)}
    assert snapshot["keys"] == tuple(sorted(keys))
    # One retry per read with the missing key removed; the second read is clean.
    assert calls[:2] == [list(range(1, 8)), [1, 2, 3, 5, 6, 7]]


def test_missing_ids_outside_the_request_are_a_real_failure():
    keys = [("video", i) for i in (1, 2, 3)]
    def fetch(ks):
        raise MissingKeys({("video", 99)})
    with pytest.raises(RuntimeError, match="catalog_enumeration_failed"):
        catalog.read_catalog(Source(fetch=fetch), keys, kinds=("video",))


def test_every_requested_id_missing_yields_an_empty_present_set():
    keys = [("video", 1), ("video", 2)]
    calls = []
    def fetch(ks):
        calls.append([i for _, i in ks])
        raise MissingKeys(set(ks))
    assert catalog.read_catalog(Source(fetch=fetch), keys, kinds=("video",)) == []
    assert calls == [[1, 2]]


def test_missing_keys_are_validated_strictly():
    keys = [("video", 41), ("video", 7)]
    # a requested subset is excluded and the read finishes
    def subset(ks):
        if ("video", 41) in ks:
            raise MissingKeys({("video", 41)})
        return complete(ks)
    assert [r["id"] for r in catalog.read_catalog(Source(fetch=subset), keys, kinds=("video",))] == [7]
    # an empty report is not a missing case
    def empty(ks):
        raise MissingKeys(set())
    with pytest.raises(RuntimeError, match="catalog_enumeration_failed"):
        catalog.read_catalog(Source(fetch=empty), keys, kinds=("video",))
    # an unrelated failure stays fatal on the first attempt
    def denied(ks):
        raise AuthorityError("permission denied")
    with pytest.raises(AuthorityError):
        catalog.read_catalog(Source(fetch=denied), keys, kinds=("video",))
    # a report mixing an unrequested key is fatal
    def mixed(ks):
        raise MissingKeys({("video", 41), ("video", 99)})
    with pytest.raises(RuntimeError, match="catalog_enumeration_failed"):
        catalog.read_catalog(Source(fetch=mixed), keys, kinds=("video",))
    # a report of another kind is fatal
    def other_kind(ks):
        raise MissingKeys({("image", 41)})
    with pytest.raises(RuntimeError, match="catalog_enumeration_failed"):
        catalog.read_catalog(Source(fetch=other_kind), keys, kinds=("video",))
    # a report during unrestricted enumeration is fatal
    def enumerate_missing(kinds, page, size):
        raise MissingKeys({("video", 1)})
    with pytest.raises(RuntimeError, match="catalog_enumeration_failed"):
        catalog.read_catalog(Source(enumerate=enumerate_missing), kinds=("video",))


def test_fingerprint_snapshot_tolerates_one_file_change_that_then_settles():
    keys = [("video", 1), ("video", 2)]
    files = iter([md5_file("a"), md5_file("b"), md5_file("b"), md5_file("b")])
    calls = []
    def fetch(ks):
        f = next(files)
        calls.append(f["fingerprints"][0]["value"][0])
        return {"items": [catalog_row("video", 1, files=[f]), catalog_row("video", 2)], "total": 2, "change_token": "c"}
    snapshot = catalog.fingerprint_snapshot(Source(fetch=fetch), keys)
    assert snapshot["groups"] == {("video", 1): "md5:" + "b" * 32}
    assert calls == ["a", "b", "b"]


def test_fingerprint_snapshot_still_refuses_a_catalog_whose_files_keep_changing():
    counter = iter("abcdefghij")
    def fetch(ks):
        return {"items": [catalog_row("video", 1, files=[md5_file(next(counter))])], "total": 1, "change_token": "c"}
    with pytest.raises(RuntimeError, match="catalog_enumeration_changed"):
        catalog.fingerprint_snapshot(Source(fetch=fetch), [("video", 1)], attempts=3)


def test_update_token_is_not_part_of_ranking_catalog_identity():
    keys = [("video", 1), ("video", 2)]
    md5 = md5_file("a")
    def reader(stamp):
        return lambda ks: {"items": [catalog_row("video", 1, files=[md5], updated=stamp),
                                     catalog_row("video", 2, updated="fixed")], "total": 2, "change_token": stamp}
    before = catalog.fingerprint_snapshot(Source(fetch=reader("t1")), keys)
    after = catalog.fingerprint_snapshot(Source(fetch=reader("t2")), keys)
    assert before == after
    assert before["groups"] == {("video", 1): "md5:" + "a" * 32}
    md5["fingerprints"][0]["value"] = "b" * 32
    changed = catalog.fingerprint_snapshot(Source(fetch=reader("t2")), keys)
    assert changed["revision"] != after["revision"] and changed["groups"] != after["groups"]
    stamps = iter(["t1", "t2"])
    churn = lambda ks: reader(next(stamps))(ks)
    assert catalog.fingerprint_snapshot(Source(fetch=churn), keys, attempts=1)["present"] == {("video", 1), ("video", 2)}


def test_catalog_retries_a_stalled_transport_but_not_a_status_or_body_error():
    replies = []
    def fetch(ks):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    source = Source(fetch=fetch)
    keys = [("video", 1)]
    replies[:] = [TransportError("timed out"), TransportError("timed out"), complete(keys)]
    assert [r["id"] for r in catalog.read_catalog(source, keys, kinds=("video",))] == [1]
    assert [c[2] for c in source.calls] == [30, 30, 30], "a stalled read is retried with the same bounded timeout"
    replies[:] = [TransportError("timed out")] * (catalog.TRANSPORT_ATTEMPTS + 2)
    with pytest.raises(TransportError):
        catalog.read_catalog(source, keys, kinds=("video",))
    assert len(source.calls) == 3 + catalog.TRANSPORT_ATTEMPTS, "a persistently stalled transport still fails, bounded"
    replies[:] = [AuthorityError("500"), complete(keys)]
    with pytest.raises(AuthorityError):
        catalog.read_catalog(source, keys, kinds=("video",))
    replies[:] = [ResponseError("malformed"), complete(keys)]
    with pytest.raises(RuntimeError, match="catalog_enumeration_failed"):
        catalog.read_catalog(source, keys, kinds=("video",))
    assert len(replies) == 1, "a status or malformed body is never retried"


def test_fingerprint_enumeration_refuses_same_count_changes_and_missing_pages():
    bad = Source(fetch=lambda ks: {"items": [catalog_row("video", 1)], "total": 501, "change_token": "c"})
    with pytest.raises(RuntimeError, match="partial"):
        catalog.fingerprint_snapshot(bad, [("video", 1)])
    reads = [0]
    def churning(ks):
        reads[0] += 1
        return {"items": [catalog_row("video", 1, files=[{"fingerprints": [{"type": "md5", "value": f"{reads[0]:032x}"}]}])],
                "total": 1, "change_token": "c"}
    with pytest.raises(RuntimeError, match="changed"):
        catalog.fingerprint_snapshot(Source(fetch=churning), [("video", 1)])
    assert reads[0] == 4   # initial read + 3 bounded attempts, then refused
    # unrestricted enumeration: a count that moves between pages is refused, duplicates too
    pages = iter([{"items": [catalog_row("video", i) for i in range(1, 501)], "total": 502, "change_token": "c"},
                  {"items": [catalog_row("video", 501)], "total": 503, "change_token": "c"}])
    with pytest.raises(RuntimeError, match="changed"):
        catalog.read_catalog(Source(enumerate=lambda k, p, s: next(pages)), kinds=("video",))
    duplicate = Source(enumerate=lambda k, p, s: {"items": [catalog_row("video", 1), catalog_row("video", 1)], "total": 2, "change_token": "c"})
    with pytest.raises(RuntimeError, match="partial"):
        catalog.read_catalog(duplicate, kinds=("video",))


def test_memory_catalog_enumeration_pages_and_orders_kinds():
    rows = [catalog_row(kind, i) for kind in ("image", "video") for i in range(1, 4)]
    source = MemoryCatalog(rows)
    out = catalog.read_catalog(source, kinds=("video", "image"))
    assert [(r["kind"], r["id"]) for r in out] == [("image", i) for i in (1, 2, 3)] + [("video", i) for i in (1, 2, 3)]


def test_supplier_reads_real_md5_rows_and_serving_final_pass_excludes_seed_duplicate():
    from feedloop import discovery
    from fl2_helpers import MemorySignals, MemorySpaces
    rows = [catalog_row("video", i, files=[md5_file(h)]) for i, h in ((1, "a"), (2, "a"), (3, "b"))]
    features = {("video", i): {"tag_seconds": {10: 100.0}, "watched_tag_seconds": None, "tag_categories": {10: "acts"}} for i in (1, 2, 3)}
    source = MemoryCatalog(rows, features)
    sources = discovery.Sources(catalog=source, spaces=MemorySpaces({}), signals=MemorySignals(), clock=lambda: 1.0)
    result = discovery.similar(sources, seed_ids=[1], config={"cooldown_days": 0, "contributor_weight": 0, "diversity": 0})
    assert [row["id"] for row in result["items"]] == [3], "item 2 shares the seed's md5 and is excluded in the final pass"
    assert len(result["provenance"]["revisions"]["catalog_fingerprints"]) == 64
    fetches = [call for call in source.calls if call[0] == "fetch"]
    assert len(fetches) >= 4, "two agreeing reads per fingerprint snapshot, at least two snapshots"
    snapshot = catalog.fingerprint_snapshot(source, [("video", 1), ("video", 2), ("video", 3)])
    assert snapshot["groups"] == {("video", 1): "md5:" + "a" * 32, ("video", 2): "md5:" + "a" * 32, ("video", 3): "md5:" + "b" * 32}


def test_same_count_fingerprint_change_invalidates_cached_content(tmp_path, monkeypatch):
    import random
    from types import SimpleNamespace
    from feedloop import engine as engine_module, ledger, tuning
    from feedloop.engine import Engine, initialize_stores
    from fl2_helpers import MemorySignals, MemorySpaces
    now = [2000000.0]
    for module in (ledger, tuning, engine_module):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], perf_counter=lambda: 0.0))
    initialize_stores(ledger_path=str(tmp_path / "e.sqlite"), tuner_path=str(tmp_path / "t.sqlite"), cutover_ts=1.0, clock=lambda: now[0])
    md5 = md5_file("a")
    source = MemoryCatalog([catalog_row("video", 2, files=[md5])],
                           {("video", 2): {"tag_seconds": {10: 100.0}, "watched_tag_seconds": None, "tag_categories": {10: "acts"}}})
    eng = Engine(catalog=source, signals=MemorySignals(observed_at=1.0), spaces=MemorySpaces({}), ledger_path=str(tmp_path / "e.sqlite"),
                 tuner_path=str(tmp_path / "t.sqlite"), config={"explore_slots": 0, "control_rate": 0.0}, clock=lambda: now[0])
    cfg = {"intent": '{"tag_ids":[10]}', "eligibility": "", "filter_identity": "intent:0", "session_id": "s", "client_request_id": "c",
           "request_id": "r", "cursor": ""}
    with monkeypatch.context() as patched:
        patched.setattr(random, "getrandbits", Mock(side_effect=[31, 32]))
        first = eng._rank(cfg, limit=20, offset=0, include_secondary=False)
        again = eng._rank(cfg, limit=20, offset=0, include_secondary=False)
        assert again["ranking"]["seed"] == 31, "an unchanged catalog serves the frozen generation again"
        md5["fingerprints"][0]["value"] = "b" * 32
        second = eng._rank(cfg, limit=20, offset=0, include_secondary=False)
    assert first["ranking"]["revisions"]["catalog_fingerprints"] != second["ranking"]["revisions"]["catalog_fingerprints"]
    assert second["ranking"]["seed"] == 32

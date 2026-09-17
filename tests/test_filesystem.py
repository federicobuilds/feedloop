"""FilesystemSource: identity, fingerprints, sidecars, signals, spaces, and the Engine over it."""
import json
import os

import numpy as np
import pytest

from feedloop import Engine, initialize_stores
from feedloop.catalog import read_catalog, fingerprint_snapshot
from feedloop.slots import MissingKeys
from feedloop.sources.filesystem import FilesystemSource, TextHashEncoder, build_text_hash_space, text_hash_vector
from fl3_helpers import Clock, make_folder


def source_for(tmp_path, **kwargs):
    media = make_folder(tmp_path, **kwargs)
    clock = Clock()
    return FilesystemSource(media, tmp_path / "state", clock=clock), media, clock


def test_identity_is_stable_and_files_are_fingerprinted(tmp_path):
    source, media, clock = source_for(tmp_path)
    rows = read_catalog(source)
    assert [(r["kind"], r["id"]) for r in rows] == [("image", i) for i in (1, 2, 3)] + [("video", i) for i in range(1, 7)]
    item = source.fetch([("video", 1)])["items"][0]
    assert item["path"] == "sample-00.mp4" and item["title"] == "Sample 0" and item["duration_s"] == 600.0 and item["duration_source"] == "sidecar"
    assert item["media_url"] == "/media/video/1" and source.media_path(("video", 1)) == media / "sample-00.mp4" and source.media_path(("video", 99)) is None
    fingerprints = {fp["type"]: fp["value"] for fp in item["files"][0]["fingerprints"]}
    assert len(fingerprints["md5"]) == 32 and len(fingerprints["sha256"]) == 64
    # reopening keeps ids; a removed file disappears from enumeration but its id is never reused
    os.remove(media / "sample-02.mp4")
    (media / "sample-99.mp4").write_bytes(b"new content")
    reopened = FilesystemSource(media, tmp_path / "state", clock=clock)
    keys = {(r["kind"], r["id"], r["path"]) for r in reopened.enumerate(("video",), 1, 500)["items"]}
    assert ("video", 3, "sample-02.mp4") not in keys and ("video", 1, "sample-00.mp4") in keys and ("video", 7, "sample-99.mp4") in keys
    with pytest.raises(MissingKeys):
        reopened.fetch([("video", 3)])


def test_metadata_token_moves_without_ranking_identity(tmp_path):
    source, media, clock = source_for(tmp_path)
    before = fingerprint_snapshot(source, [("video", 1)])
    token = source.fetch([("video", 1)])["items"][0]["updated"]
    clock.advance(5)
    assert source.apply_change(("video", 1), {"action": "rating", "rating100": 80})["status"] == "confirmed"
    assert source.fetch([("video", 1)])["items"][0]["updated"] != token
    assert fingerprint_snapshot(source, [("video", 1)])["revision"] == before["revision"]
    # a changed file moves the ranking identity
    (media / "sample-00.mp4").write_bytes(b"replaced bytes")
    source.refresh()
    assert fingerprint_snapshot(source, [("video", 1)])["revision"] != before["revision"]


def test_duplicate_content_shares_a_group(tmp_path):
    source, media, _ = source_for(tmp_path)
    source.refresh()
    (media / "copy-of-00.mp4").write_bytes((media / "sample-00.mp4").read_bytes())
    source.refresh()
    snapshot = fingerprint_snapshot(source, [("video", 1), ("video", 7)])
    assert snapshot["groups"][("video", 1)] == snapshot["groups"][("video", 7)]


def test_sidecar_tags_features_and_invalid_sidecar(tmp_path):
    source, media, _ = source_for(tmp_path)
    features = source.features([("video", 1), ("image", 1), ("video", 2)])
    names = source.tag_names()
    assert set(names.values()) >= {"amber", "linen", "umber"}
    assert all(v == 600.0 for v in features[("video", 1)]["tag_seconds"].values()) and features[("video", 1)]["watched_tag_seconds"] is None
    assert all(v == 1.0 for v in features[("image", 1)]["tag_seconds"].values())
    (media / "sample-01.mp4.json").write_text("{not json")
    source.refresh()
    broken = source.fetch([("video", 2)])["items"][0]
    assert broken["sidecar_valid"] is False and broken["tags"] == [] and broken["title"] == "sample-01"
    assert ("video", 2) not in source.features([("video", 2)])
    # duration is never invented
    (media / "sample-03.mp4.json").write_text(json.dumps({"tags": ["moss"]}))
    source.refresh()
    assert source.fetch([("video", 4)])["items"][0]["duration_s"] is None
    assert source.features([("video", 4)])[("video", 4)]["tag_seconds"] == {names_inverse(names)["moss"]: 1.0}


def names_inverse(names):
    return {v: k for k, v in names.items()}


def test_generated_tags_merge_without_touching_user_sidecar(tmp_path):
    source, media, _ = source_for(tmp_path)
    original = (media / "sample-00.mp4.json").read_text()
    (media / "sample-00.mp4.generated.json").write_text(json.dumps({"model": "x", "tags": [{"name": "harbor", "category": "zero_shot", "probability": 0.5}]}))
    source.refresh()
    assert "harbor" in source.fetch([("video", 1)])["items"][0]["tags"]
    assert (media / "sample-00.mp4.json").read_text() == original


def test_signals_read_current_apply_change_and_watch(tmp_path):
    source, media, clock = source_for(tmp_path)
    assert source.read_current(("video", 1)) == {"status": "ok", "rating100": None, "engagement_count": 0}
    assert source.read_current(("video", 99))["status"] == "error"
    assert source.apply_change(("video", 1), {"action": "engagement", "delta": -1})["status"] == "conflict"
    assert source.apply_change(("video", 1), {"action": "engagement", "delta": 1}) == {"status": "confirmed", "rating100": None, "engagement_count": 1}
    assert source.apply_change(("video", 1), {"action": "rating", "rating100": 101})["status"] == "conflict"
    written = clock.advance(10)
    assert source.apply_change(("video", 1), {"action": "rating", "rating100": 40})["status"] == "confirmed"
    def step(event_id, kind, at, position, previous, **extra):
        return {"id": event_id, "stream_session_id": "st", "type": kind, "item_id": 1, "occurred_at": at, "position": position, "duration": 600.0,
                "playback_rate": 1, "previous_event_id": previous, **extra}
    rows = [step("s", "view_start", written, 0, None), step("p", "view_progress", written + 4, 4, "s"), step("k", "view_seek", written + 5, 100, "p"),
            step("q", "view_progress", written + 9, 104, "k")]
    assert source.commit_watch(rows, received_at=written + 9) == {"status": "committed", "stored": 4, "credited": 2, "received_at": written + 9}
    assert source.commit_watch(rows, received_at=written + 9)["stored"] == 0, "replay stores nothing"
    # a batch with one invalid row writes nothing authoritative: too fast an advance, a fork, a clock gap, a regression, a bad duration
    before = source.read()["rows"][("video", 1)]["watch"]
    for bad, code in [(step("z1", "view_progress", written + 14, 600, "ok"), "watch_seek_or_rate_unproven"),
                      (step("z2", "view_progress", written + 14, 109, "k"), "watch_chain_fork"),
                      (step("z3", "view_progress", written + 30, 109, "ok"), "watch_chain_clock_gap"),
                      (step("z4", "view_progress", written + 13, 105, "q"), "watch_stream_clock_regressed"),
                      (step("z5", "view_progress", written + 14, 109, None), "watch_chain_gap"),
                      ({"id": "z6", "type": "view_start", "item_id": 1, "occurred_at": written, "position": 5, "duration": 0, "stream_session_id": "st"}, "watch_clock_or_position_invalid")]:
        result = source.commit_watch([step("ok", "view_progress", written + 13, 108, "q"), bad], received_at=written + 30)
        assert (result["status"], result["error_code"], result["stored"]) == ("rejected", code, 0), code
        assert source.read()["rows"][("video", 1)]["watch"] == before, code
    read = source.read()
    row = read["rows"][("video", 1)]
    assert read["observed_at"] == written and row["rating"] == 40 and row["engagement_count"] == 1
    assert row["watch"]["watched_s"] == 8.0 and row["watch"]["intervals"] == [(0.0, 4.0), (100.0, 104.0)] and row["watch"]["visit_days"] == [int((written + 9) // 86400)]
    # the client's singleton batching: one batch per player event, anchors persisted across calls
    later = written + 100
    assert source.commit_watch([step("a", "view_start", later, 200, None, stream_session_id="st2")], received_at=later)["credited"] == 0
    assert source.commit_watch([step("b", "view_progress", later + 4, 204, "a", stream_session_id="st2")], received_at=later + 4)["credited"] == 1
    assert source.read()["rows"][("video", 1)]["watch"]["watched_s"] == 12.0, "coverage intervals merge; a new segment adds"
    # a player-reported duration is labelled, not measured
    assert source.fetch([("video", 1)])["items"][0]["duration_source"] == "sidecar"
    (media / "sample-00.mp4.json").write_text(json.dumps({"tags": ["amber"]}))
    source.refresh()
    assert source.fetch([("video", 1)])["items"][0]["duration_source"] == "player"


def test_spaces_write_load_windows_and_text_hash(tmp_path):
    source, media, _ = source_for(tmp_path)
    assert source.spaces() == [] and source.revision("visual") is None and source.matrix("visual") is None and source.windows("visual") is None
    report = build_text_hash_space(source, "sidecar_text")
    assert report["items"] == 9 and report["windows"] == 0 and "not a measurement" in report["provenance"]
    assert source.spaces() == ["sidecar_text"] and source.revision("sidecar_text") == report["revision"]
    keys, matrix = source.matrix("sidecar_text")
    assert len(keys) == 9 and matrix.shape == (9, 64) and np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    assert source.windows("sidecar_text") is None and source.space_meta("sidecar_text")["window_scope"] == "none", "no invented timestamps for mean vectors"
    with pytest.raises(ValueError):
        source.write_space("timed", keys, matrix, meta={"provenance": "x", "window_scope": "timed"})
    with pytest.raises(ValueError):
        source.write_space("nowin", keys, matrix, meta={"provenance": "x", "window_scope": "none"}, windows=(keys, np.zeros(len(keys)), matrix))
    source.write_space("timed", keys[:1], matrix[:1], meta={"provenance": "x", "window_scope": "timed"}, windows=(keys[:1] * 2, np.array([1.5, 7.0]), np.stack([matrix[0], matrix[0]])))
    assert list(source.windows("timed")[1]) == [1.5, 7.0]
    encoder = TextHashEncoder(["sidecar_text"])
    assert encoder.encode("visual", "amber") is None and encoder.encode("sidecar_text", "") is None
    query = encoder.encode("sidecar_text", "amber")
    sims = matrix @ query
    tagged = {k for k, item in zip(keys, source.fetch(keys)["items"]) if "amber" in item["tags"]}
    assert {keys[i] for i in np.argsort(-sims)[:len(tagged)]} == tagged
    # removed items drop out of the aligned matrix; a corrupt file reads as absent
    os.remove(media / "sample-00.mp4")
    source.refresh()
    assert ("video", 1) not in source.matrix("sidecar_text")[0]
    (source.spaces_dir / "sidecar_text.npz").write_bytes(b"garbage")
    assert source.revision("sidecar_text") is None and source.matrix("sidecar_text") is None
    with pytest.raises(ValueError):
        source.write_space("bad name", [], np.zeros((0, 4)), meta={"provenance": "x", "window_scope": "none"})
    with pytest.raises(ValueError):
        source.write_space("ok", [("video", 1)], np.zeros((1, 4)), meta={"window_scope": "none"})


def test_engine_feed_search_similar_over_folder(tmp_path):
    source, media, clock = source_for(tmp_path)
    build_text_hash_space(source, "sidecar_text")
    initialize_stores(ledger_path=tmp_path / "state" / "ledger.sqlite", tuner_path=tmp_path / "state" / "tuner.sqlite", cutover_ts=clock(), clock=clock)
    engine = Engine(catalog=source, signals=source, spaces=source, encoder=TextHashEncoder(["sidecar_text"]), ledger_path=tmp_path / "state" / "ledger.sqlite",
                    tuner_path=tmp_path / "state" / "tuner.sqlite", read_current=source.read_current, apply_change=source.apply_change, clock=clock,
                    space_roles={"visual": "visual", "semantic": "sidecar_text", "voice": "audioembed", "sound": "audiomix"},
                    config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0})
    clock.advance(60)
    page = engine.feed({"limit": 8, "images": True, "surface": "feed", "session_id": "s", "request_id": "r", "client_request_id": "c"})
    assert page["status"] == "ok" and len(page["items"]) == 6 and {i["kind"] for i in page["items"]} == {"video"}, "no positive evidence yet: the fallback lane serves only primary items"
    assert all(i["media_url"] == f"/media/{i['kind']}/{i['id']}" for i in page["items"])
    assert engine.search("amber", "look")["status"] == "no-feature", "discovery.search needs window rows; the means-only space has none (the server route supplies the means fallback)"
    assert engine.search("amber", "sound")["status"] == "no-feature"
    alike = engine.similar(("video", 1), limit=5)
    assert alike["status"] == "ok" and alike["items"]
    clock.advance(1)
    result = engine.feedback(("video", 1), operation_id="op-1", session_id="s", rating=90)
    assert result["status"] == "confirmed" and source.read_current(("video", 1))["rating100"] == 90
    clock.advance(1)
    assert engine.undo(result, operation_id="op-2")["status"] == "confirmed" and source.read_current(("video", 1))["rating100"] is None


def test_timed_segments_become_the_only_window_rows(tmp_path):
    media = make_folder(tmp_path, videos=2, images=1, segments=True)
    (media / "sample-01.mp4.json").write_text(json.dumps({"tags": ["moss"], "segments": [{"start_s": -1, "tags": ["moss"]}, {"start_s": "x", "tags": ["moss"]}, {"tags": ["moss"]}]}))
    source = FilesystemSource(media, tmp_path / "state", clock=Clock())
    report = build_text_hash_space(source, "sidecar_text")
    keys, times, matrix = source.windows("sidecar_text")
    assert report["windows"] == 3 and keys == [("video", 1)] * 3 and list(times) == [0.0, 30.0, 60.0] and matrix.shape == (3, 64)
    assert source.space_meta("sidecar_text")["window_scope"] == "timed" and "segment starts" in source.space_meta("sidecar_text")["provenance"]
    assert len(source.matrix("sidecar_text")[0]) == 3, "every tagged item keeps its mean; only segmented videos have windows"


def test_text_hash_vector_is_deterministic_and_word_based():
    a, b = text_hash_vector("Amber, cobalt"), text_hash_vector("cobalt amber")
    assert a is not None and np.allclose(a, b) and text_hash_vector("  ") is None and np.isclose(np.linalg.norm(a), 1.0)


def test_long_media_name_keeps_the_item_and_names_the_missing_sidecar(tmp_path):
    """A media name near the filesystem limit cannot take ``.json`` or ``.generated.json``:
    the scan must not abort, the item stays catalogued with a named reason, and no writer
    touches the impossible sidecar path."""
    from feedloop.cli import write_fixture_sidecars
    from feedloop.extractors.visual import write_generated_tags
    media = make_folder(tmp_path, videos=1, images=1, sidecars=False)
    long_stem = "n" * 250
    (media / f"{long_stem}.png").write_bytes(b"\x89PNG" + bytes(range(200)))
    (media / f"{long_stem}.mp4").write_bytes(bytes(range(256)) * 4)
    with pytest.raises(OSError):
        (media / f"{long_stem}.png.json").is_file()  # the failure the source must absorb
    source = FilesystemSource(media, tmp_path / "state", clock=Clock())
    scan = source.refresh()
    items = source.enumerate(("video", "image"), 1, 50)["items"]
    long_items = {i["kind"]: i for i in items if i["title"] == long_stem}
    assert len(scan["items"]) == 4 and set(long_items) == {"video", "image"}
    for item in long_items.values():
        assert item["sidecar_reason"] == "sidecar_path_too_long" and item["sidecar_valid"] is True and item["tags"] == [] and item["duration_s"] is None
        assert len(item["files"][0]["fingerprints"]) == 2 and item["media_url"] == f"/media/{item['kind']}/{item['id']}"
        assert source.media_path((item["kind"], item["id"])) == media / f"{long_stem}.{'mp4' if item['kind'] == 'video' else 'png'}"
    short = next(i for i in items if i["title"] == "sample-00")
    assert short["sidecar_reason"] is None
    assert source.sidecar_reasons([(i["kind"], i["id"]) for i in items]) == {(i["kind"], i["id"]): i["sidecar_reason"] for i in items}
    # the fixture writer skips the impossible paths and writes the others
    assert write_fixture_sidecars(media) == 2
    assert (media / "sample-00.mp4.json").is_file() and (media / "still-00.jpg.json").is_file()
    # the extractor's generated-tag writer refuses the impossible path without raising
    assert write_generated_tags(source, ("image", long_items["image"]["id"]), [("alpha", 0.9)], model="fake") is None
    assert write_generated_tags(source, ("image", short_image_id := next(i["id"] for i in items if i["title"] == "still-00")), [("alpha", 0.9)], model="fake") is not None
    assert (media / "still-00.jpg.generated.json").is_file() and short_image_id
    # a second scan sees the written sidecars and still names the reason for the long names
    after = {i["title"]: i for i in source.refresh() and source.enumerate(("video", "image"), 1, 50)["items"]}
    assert after["sample-00"]["tags"] and after[long_stem]["sidecar_reason"] == "sidecar_path_too_long"
    assert sorted(p.name for p in media.iterdir() if p.suffix == ".json") == ["sample-00.mp4.json", "still-00.jpg.generated.json", "still-00.jpg.json"]

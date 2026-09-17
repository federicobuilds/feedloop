"""HTTP boundary: authorization, bounded bodies, delivery versus observation, view and watch
capture, feedback and undo, discovery, scorecard, tuner controls, static and media files."""
import json
import threading
import urllib.request

import pytest

from feedloop import ledger
from feedloop.server.app import MAX_BODY_BYTES, run_server
from fl3_helpers import feed_request, make_app, request


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    return make_app(tmp_path, monkeypatch)


def test_health_config_and_static_files(ctx):
    assert request(ctx, "GET", "/api/health") == (200, {"ok": True})
    status, config = request(ctx, "GET", "/api/config")
    assert status == 200 and config["primary"] == "video" and config["attribution"]["window_s"] == 5.0 and config["encoder"] is True
    assert "not a measurement" in config["space_meta"]["sidecar_text"]["provenance"]
    assert request(ctx, "GET", "/")[0] == 200 and request(ctx, "GET", "/app.js")[0] == 200 and request(ctx, "GET", "/feed")[0] == 200
    assert request(ctx, "GET", "/missing.js")[0] == 404 and request(ctx, "GET", "/../pyproject.toml")[0] == 404
    assert request(ctx, "POST", "/index.html")[0] == 405 and request(ctx, "GET", "/api/nothing")[0] == 404


def test_media_serves_catalogued_items_only(ctx, tmp_path):
    item = ctx.source.fetch([("video", 1)])["items"][0]
    assert item["media_url"] == "/media/video/1"
    status, headers, body = ctx.app.handle("GET", "/media/video/1", {}, b"")
    assert status == 200 and body == (ctx.media / "sample-00.mp4").read_bytes() and ("Accept-Ranges", "bytes") in headers
    status, headers, body = ctx.app.handle("GET", "/media/video/1", {"range": "bytes=10-19"}, b"")
    assert status == 206 and body == (ctx.media / "sample-00.mp4").read_bytes()[10:20] and dict(headers)["Content-Range"].startswith("bytes 10-19/")
    assert ctx.app.handle("GET", "/media/video/1", {"range": "bytes=999999-"}, b"")[0] == 416
    assert ctx.app.handle("GET", "/media/image/1", {}, b"")[0] == 200
    # nothing that is not a catalogued item: paths, sidecars, the state directory, dotfiles, unknown ids or kinds
    (ctx.media / ".feedloop").mkdir(exist_ok=True)
    (ctx.media / ".feedloop" / "source.sqlite").write_bytes(b"not served")
    (ctx.media / ".hidden.mp4").write_bytes(b"dotfile")
    ctx.source.refresh()
    for target in ("/media/sample-00.mp4", "/media/sample-00.mp4.json", "/media/.feedloop/source.sqlite", "/media/../state/source.sqlite",
                   "/media/video/999", "/media/other/1", "/media/video/x", "/media/.hidden.mp4", "/media/", f"/media/{(tmp_path / 'state' / 'source.sqlite').as_posix()}"):
        assert ctx.app.handle("GET", target, {}, b"")[0] == 404, target
    assert (tmp_path / "state" / "source.sqlite").exists()
    assert ("video", 7) not in {(i["kind"], i["id"]) for i in ctx.source.enumerate(("video",), 1, 500)["items"]} or ctx.source.media_path(("video", 7)).name != ".hidden.mp4"


def test_mutations_require_key_and_exact_origin(ctx):
    body = feed_request()
    assert request(ctx, "POST", "/api/feed", body, auth=False) == (401, {"detail": "mutation_credential_required"})
    assert request(ctx, "POST", "/api/feed", body, origin="http://evil.example:8765") == (403, {"detail": "mutation_origin_denied"})
    assert request(ctx, "POST", "/api/feed", body, headers={"x-ai-api-key": "wrong", "origin": ctx.origin}, auth=False)[0] == 401
    assert ledger.read_evidence(ctx.engine.ledger_path, since_ts=0, through_ts=ctx.clock() + 1)["requests"] == []
    for route in ("view", "watch", "feedback", "feedback/reconcile", "tick", "tuner/reset", "tuner/revert?ledger_id=1"):
        assert request(ctx, "POST", "/api/" + route, {}, auth=False)[0] == 401, route
    ctx.app.api_key = None
    assert request(ctx, "POST", "/api/feed", body) == (503, {"detail": "mutation_auth_unconfigured"})


def test_bounded_and_malformed_bodies(ctx):
    assert request(ctx, "POST", "/api/feed", b"{not json") == (400, {"detail": "invalid_json"})
    status, _, _ = ctx.app.handle("POST", "/api/feed", {"x-ai-api-key": ctx.key, "origin": ctx.origin}, None)
    assert status == 413
    status, payload = request(ctx, "POST", "/api/feed", {"limit": "eight"})
    assert status == 200 and payload["status"] == "error" and payload["error_code"] == "invalid_feed_request"


def test_delivery_then_observation_strips_identities(ctx):
    ctx.clock.advance(30)
    status, page = request(ctx, "POST", "/api/feed", feed_request())
    assert status == 200 and page["status"] == "ok" and page["items"] and page["request_id"] == "req-1"
    assert all(item["served_item_id"] and item["request_id"] == "req-1" for item in page["items"])
    status, observed = request(ctx, "GET", "/api/feed?limit=8&images=1&surface=feed&session_id=session-1", auth=False)
    assert status == 200 and observed["status"] == "ok" and observed["request_id"] is None and observed["client_request_id"] is None
    assert [item["id"] for item in observed["items"]] == [item["id"] for item in page["items"]]
    assert all(item["served_item_id"] is None and item["request_id"] is None and item["viewed_event_id"] is None for item in observed["items"])
    assert ctx.app.observe_feed({"limit": ["8"], "session_id": ["someone-else"]})[2] == json.dumps(
        {"status": "unavailable", "error_code": "feed_observation_unavailable", "items": [], "request_id": None}, separators=(",", ":")).encode()
    assert request(ctx, "GET", "/api/feed?surface=nope", auth=False)[1]["error_code"] == "invalid_surface"
    requests = ledger.read_evidence(ctx.engine.ledger_path, since_ts=0, through_ts=ctx.clock() + 1)["requests"]
    assert [r["request_id"] for r in requests] == ["req-1"], "observation never records a delivery"
    # a stale cursor is explicit and never records
    ctx.clock.advance(1)
    cursor = dict(page["pagination"]["next_cursor"] or {"offset": len(page["items"]), "after": "video:1"}, generation_id="0" * 64)
    status, stale = request(ctx, "POST", "/api/feed", feed_request(request_id="req-2", client_request_id="client-2", offset=cursor["offset"], cursor=cursor))
    assert (stale["status"], stale["error_code"], stale["items"]) == ("error", "stale_ranking_cursor", [])
    assert len(ledger.read_evidence(ctx.engine.ledger_path, since_ts=0, through_ts=ctx.clock() + 1)["requests"]) == 1


def deliver_and_view(ctx):
    ctx.clock.advance(30)
    _, page = request(ctx, "POST", "/api/feed", feed_request())
    item = next(i for i in page["items"] if i["kind"] == "video")
    ctx.clock.advance(2)
    view = {"client_event_id": "view-1", "session_id": "session-1", "request_id": "req-1", "served_item_id": item["served_item_id"], "surface": "feed",
            "position": item["source_rank"], "dwell_ms": 1200, "visible_fraction": 0.6, "visibility_policy": "foreground-60pct-1200ms-v1",
            "kind": "video", "item_id": item["id"], "occurred_at": ctx.clock()}
    status, viewed = request(ctx, "POST", "/api/view", view)
    assert status == 200 and viewed["status"] == "confirmed"
    return item, viewed


def watch_batch(item, viewed, at, *, capture="capture-1"):
    return {"capture_id": capture, "stream_session_id": "stream-1", "session_id": "session-1", "events": [
        {"id": capture + "-start", "type": "view_start", "item_id": item["id"], "occurred_at": at - 4, "position": 0, "duration": 600.0, "viewed_event_id": viewed["event_id"], "previous_event_id": None, "playback_rate": 1},
        {"id": capture + "-progress", "type": "view_progress", "item_id": item["id"], "occurred_at": at, "position": 4, "duration": 600.0, "viewed_event_id": viewed["event_id"], "previous_event_id": capture + "-start", "playback_rate": 1}]}


def test_observation_never_ranks_opens_stores_or_starts_threads(ctx, monkeypatch):
    import sqlite3
    import threading
    ctx.clock.advance(30)
    request(ctx, "POST", "/api/feed", feed_request())
    tripped = []
    monkeypatch.setattr(ctx.engine, "feed", lambda payload: tripped.append("feed") or {"status": "error", "items": []})
    monkeypatch.setattr(ctx.engine, "_rank", lambda *a, **k: tripped.append("rank"))
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: tripped.append("sqlite") or (_ for _ in ()).throw(AssertionError("sqlite opened")))
    monkeypatch.setattr(threading.Thread, "start", lambda self: tripped.append("thread") or (_ for _ in ()).throw(AssertionError("thread started")))
    status, warm = request(ctx, "GET", "/api/feed?limit=8&images=1&surface=feed&session_id=session-1", auth=False)
    assert status == 200 and warm["status"] == "ok" and warm["items"] and warm["request_id"] is None
    status, cold = request(ctx, "GET", "/api/feed?limit=8&images=1&surface=feed&session_id=nobody", auth=False)
    assert status == 200 and cold["error_code"] == "feed_observation_unavailable" and cold["items"] == []
    assert tripped == [], tripped


def test_view_watch_attribution_and_scorecard(ctx):
    item, viewed = deliver_and_view(ctx)
    status, unqualified = request(ctx, "POST", "/api/view", {"client_event_id": "view-2", "session_id": "session-1", "request_id": "req-1", "served_item_id": item["served_item_id"],
                                                             "surface": "feed", "position": item["source_rank"], "dwell_ms": 300, "visible_fraction": 0.6,
                                                             "visibility_policy": "foreground-60pct-1200ms-v1", "kind": "video", "item_id": item["id"], "occurred_at": ctx.clock()})
    assert unqualified["status"] == "error"
    at = ctx.clock.advance(4)
    status, imported = request(ctx, "POST", "/api/watch", watch_batch(item, viewed, at))
    assert status == 200 and imported["status"] == "imported" and imported["outcomes"] == 1 and imported["stored_steps"] == 2
    assert request(ctx, "POST", "/api/watch", watch_batch(item, viewed, at))[1]["status"] == "duplicate"
    assert ctx.source.read([("video", item["id"])])["rows"][("video", item["id"])]["watch"]["watched_s"] == 4.0
    assert request(ctx, "POST", "/api/watch", {"capture_id": "x"})[1]["error_code"] == "invalid_watch_batch"
    assert request(ctx, "POST", "/api/tick")[1]["attributed"] == 0
    ctx.clock.advance(6)
    assert request(ctx, "POST", "/api/tick")[1]["attributed"] == 1
    status, card = request(ctx, "GET", "/api/scorecard", auth=False)
    assert card["capture"]["verified"] is True and card["capture"]["attributed_outcomes"] == 1 and card["tuner"]["knob"] == "embedding_weight"
    # an invalid batch is validated before anything authoritative is written and never reaches the ledger
    history = ctx.source.read([("video", item["id"])])["rows"][("video", item["id"])]["watch"]
    bad = watch_batch(item, viewed, ctx.clock.advance(10), capture="capture-bad")
    bad["events"][1]["position"] = 500  # 500 s advanced in 4 s of playback
    status, refused = request(ctx, "POST", "/api/watch", bad)
    assert status == 200 and refused["status"] == "rejected" and refused["error_code"] == "watch_seek_or_rate_unproven" and refused["stored_steps"] == 0
    assert ctx.source.read([("video", item["id"])])["rows"][("video", item["id"])]["watch"] == history
    assert ledger.read_capture_readiness(ctx.engine.ledger_path)["receipts"] == {"imported": 1}, "a rejected batch never becomes a receipt"
    # an unproven view reference is refused by the ledger before any source write
    history = ctx.source.read([("video", item["id"])])["rows"][("video", item["id"])]["watch"]
    unproven = watch_batch(item, {"event_id": "not-a-view"}, ctx.clock.advance(10), capture="capture-unproven")
    status, refused = request(ctx, "POST", "/api/watch", unproven)
    assert status == 200 and refused["status"] == "rejected" and refused["error_code"] == "watch_view_or_session_unproven" and refused["quarantined"] == 2 and refused["stored_steps"] == 0
    assert ctx.source.read([("video", item["id"])])["rows"][("video", item["id"])]["watch"] == history
    other_session = watch_batch(item, viewed, ctx.clock.advance(10), capture="capture-session")
    other_session["session_id"] = "someone-else"
    status, refused = request(ctx, "POST", "/api/watch", other_session)
    assert refused["status"] == "rejected" and refused["stored_steps"] == 0 and ctx.source.read([("video", item["id"])])["rows"][("video", item["id"])]["watch"] == history
    # the client's singleton batches: one event per batch on the view's one stream, anchored across calls
    start_at = ctx.clock.advance(20)
    single_start = {"capture_id": "c-start", "stream_session_id": "stream-1", "session_id": "session-1", "events": [watch_batch(item, viewed, start_at + 4, capture="c2")["events"][0]]}
    single_progress = {"capture_id": "c-progress", "stream_session_id": "stream-1", "session_id": "session-1", "events": [watch_batch(item, viewed, start_at + 4, capture="c2")["events"][1]]}
    assert request(ctx, "POST", "/api/watch", single_start)[1]["outcomes"] == 0
    ctx.clock.advance(4)
    progressed = request(ctx, "POST", "/api/watch", single_progress)[1]
    assert progressed["status"] == "imported" and progressed["outcomes"] == 1
    assert ctx.source.read([("video", item["id"])])["rows"][("video", item["id"])]["watch"]["watched_s"] == 4.0


def test_feedback_undo_reconcile_and_conflicts(ctx):
    item, viewed = deliver_and_view(ctx)
    ctx.clock.advance(1)
    op = {"operation_id": "op-1", "session_id": "session-1", "kind": "video", "item_id": item["id"], "action": "rating", "rating100": 90, "request_id": "req-1", "viewed_event_id": viewed["event_id"]}
    status, result = request(ctx, "POST", "/api/feedback", op)
    assert result["status"] == "confirmed" and result["after"]["rating100"] == 90 and ctx.source.read_current(("video", item["id"]))["rating100"] == 90
    assert request(ctx, "POST", "/api/feedback", op)[1]["status"] == "confirmed", "replay is idempotent"
    assert request(ctx, "POST", "/api/feedback", {**op, "rating100": 10})[1]["error_code"] == "operation_conflict"
    assert request(ctx, "POST", "/api/feedback", "text")[1]["status"] == "indeterminate"
    ctx.clock.advance(1)
    undo = {"operation_id": "op-2", "session_id": "session-1", "kind": "video", "item_id": item["id"], "action": "undo", "undo_of": "op-1"}
    status, undone = request(ctx, "POST", "/api/feedback", undo)
    assert undone["status"] == "confirmed" and ctx.source.read_current(("video", item["id"])) == {"status": "ok", "rating100": None, "engagement_count": 0}
    assert request(ctx, "POST", "/api/feedback/reconcile", {"operation_id": "op-1"})[1]["status"] == "confirmed"
    assert request(ctx, "POST", "/api/feedback/reconcile", {"operation_id": "nope"})[1]["error_code"] == "unknown_operation"
    ctx.clock.advance(1)
    status, counted = request(ctx, "POST", "/api/feedback", {"operation_id": "op-3", "session_id": "session-1", "kind": "video", "item_id": item["id"], "action": "engagement"})
    assert counted["status"] == "confirmed" and counted["after"]["engagement_count"] == 1


def test_search_similar_and_whole_item_positions(ctx):
    ctx.clock.advance(30)
    facade = ctx.engine.search("amber", "look")
    status, found = request(ctx, "GET", "/api/search?q=amber", auth=False)
    assert facade["status"] == "no-feature" and found == json.loads(json.dumps(facade)), "no window rows: the facade says no-feature and the route says exactly that"
    assert request(ctx, "GET", "/api/search?q=", auth=False)[1]["status"] == "empty"
    assert request(ctx, "GET", "/api/search?q=amber&mode=sound", auth=False)[1]["status"] == "no-feature"
    assert request(ctx, "GET", "/api/search?q=amber&limit=zz", auth=False)[1]["error_code"] == "invalid_search_request"
    status, alike = request(ctx, "GET", "/api/similar?kind=video&id=1", auth=False)
    assert alike["status"] == "ok" and alike["items"] and all(i["kind"] == "video" and i["id"] != 1 for i in alike["items"])
    assert request(ctx, "GET", "/api/similar?kind=video&id=x", auth=False)[1]["error_code"] == "invalid_similar_request"
    assert request(ctx, "GET", "/api/similar?kind=other&id=1", auth=False)[1]["error_code"] == "invalid_similar_request"


def test_search_over_timed_fixture_windows_matches_the_facade(tmp_path, monkeypatch):
    ctx = make_app(tmp_path, monkeypatch, segments=True)
    ctx.clock.advance(30)
    meta = ctx.source.space_meta("sidecar_text")
    assert meta["window_scope"] == "timed" and "segment starts" in meta["provenance"] and len(ctx.source.windows("sidecar_text")[0]) == 18
    facade = ctx.engine.search("amber", "look")
    status, found = request(ctx, "GET", "/api/search?q=amber", auth=False)
    assert found == json.loads(json.dumps(facade)) and found["status"] == "ok" and found["items"]
    assert all(i["kind"] == "video" and i["search"]["best_t"] in (0.0, 30.0, 60.0) for i in found["items"]), "positions are the fixture's own segment starts"
    tagged = {i["id"] for i in ctx.source.enumerate(("video",), 1, 500)["items"] if "amber" in i["tags"]}
    assert found["items"][0]["id"] in tagged


def test_media_refuses_a_catalogued_path_turned_symlink(ctx, tmp_path):
    import os
    target = ctx.media / "sample-00.mp4"
    assert ctx.app.handle("GET", "/media/video/1", {}, b"")[0] == 200
    os.remove(target)
    os.symlink(tmp_path / "state" / "source.sqlite", target)
    assert target.is_symlink() and ctx.source.media_path(("video", 1)) == target, "the scan cache still points at the catalogued path"
    status, _, body = ctx.app.handle("GET", "/media/video/1", {}, b"")
    assert status == 404 and body == b'{"detail":"not_found"}'
    os.remove(target)
    (ctx.media / "linkdir").symlink_to(tmp_path / "state", target_is_directory=True)
    ctx.source.refresh()
    assert not any(i["path"].startswith("linkdir") for i in ctx.source.enumerate(("video", "image"), 1, 500)["items"]), "symlinked directories are not scanned"
    assert ctx.app.handle("GET", "/linkdir/source.sqlite", {}, b"")[0] == 404


def test_tuner_controls_and_scorecard_cache(ctx):
    status, card = request(ctx, "GET", "/api/scorecard", auth=False)
    assert card["tuner"]["ledger"] == [] and card["automation"]["enabled"] is True
    with ctx.engine.tuner.lock:
        revision = ctx.engine.tuner.revision[0]
    assert ctx.engine.tuner.promote({"winner": "cand", "demonstration": True}, expected_revision=revision) is True
    status, card = request(ctx, "GET", "/api/scorecard", auth=False)
    applied = [row for row in card["tuner"]["ledger"] if row["status"] == "applied"]
    assert len(applied) == 1 and applied[0]["evidence"]["demonstration"] is True and card["tuner"]["tuned_values"]["embedding_weight"] == applied[0]["new"]
    assert request(ctx, "POST", "/api/tuner/revert?ledger_id=abc")[0] == 400
    status, reverted = request(ctx, "POST", f"/api/tuner/revert?ledger_id={applied[0]['id']}")
    assert reverted["ok"] is True
    status, card = request(ctx, "GET", "/api/scorecard", auth=False)
    assert any(row["id"] == applied[0]["id"] and row["status"] == "reverted" for row in card["tuner"]["ledger"])
    status, reset = request(ctx, "POST", "/api/tuner/reset")
    tuned = request(ctx, "GET", "/api/scorecard", auth=False)[1]["tuner"]["tuned_values"]
    assert reset["ok"] is True and tuned == reset["restored"] and tuned["embedding_weight"] == 0.35


def test_internal_failures_hide_details(ctx, monkeypatch):
    monkeypatch.setattr(ctx.engine, "scorecard", lambda: 1 / 0)
    assert request(ctx, "GET", "/api/scorecard", auth=False) == (500, {"detail": "internal_error"})


def test_real_socket_roundtrip(ctx):
    server = run_server(ctx.app, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
            assert json.loads(response.read()) == {"ok": True}
        big = urllib.request.Request(f"http://127.0.0.1:{port}/api/feed", data=b"x" * (MAX_BODY_BYTES + 1), method="POST",
                                     headers={"x-ai-api-key": ctx.key, "Origin": ctx.origin, "Content-Type": "application/json"})
        try:
            urllib.request.urlopen(big, timeout=5)
            assert False, "oversized body accepted"
        except urllib.error.HTTPError as error:
            assert error.code == 413
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/media/video/1", timeout=5) as response:
            assert response.read() == (ctx.media / "sample-00.mp4").read_bytes()
    finally:
        server.shutdown()
        server.server_close()


def test_served_items_carry_the_sidecar_reason(ctx):
    long_stem = "n" * 250
    (ctx.media / f"{long_stem}.mp4").write_bytes(bytes(range(256)) * 4)
    ctx.source.refresh()
    ctx.clock.advance(30)
    status, feed = request(ctx, "POST", "/api/feed", feed_request(limit=40))
    assert status == 200 and feed["status"] == "ok"
    by_title = {item["title"]: item for item in feed["items"]}
    assert long_stem in by_title and by_title[long_stem]["sidecar_reason"] == "sidecar_path_too_long"
    assert by_title["Sample 0"]["sidecar_reason"] is None
    _, observed = request(ctx, "GET", "/api/feed?limit=40&images=1&surface=feed&session_id=session-1", auth=False)
    assert {i["title"]: i["sidecar_reason"] for i in observed["items"]}[long_stem] == "sidecar_path_too_long"

#!/usr/bin/env python3
"""Portable demonstration runner: the real filesystem source and HTTP server over one
copied media folder, on loopback, with a fake clock for the shortened attribution
window and labelled synthetic fixtures for tags, the watch batch and tuner trials.

Usage: verify_demo.py --folder MEDIA --state-dir STATE --report REPORT.json

Real checks read and serve the copied files. Synthetic checks are labelled as such in
the report and never describe the media: the sidecar tags, the watch batch positions
and duration, and the tuner trials are fixture values. The report is
{"status": "PASS", ...} only when every required assertion holds; the exit code is
nonzero otherwise. State is written only under --state-dir plus sidecar JSON files
next to the copied media inside --folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import threading
import time
import traceback
import types
import urllib.error
import urllib.request

import numpy as np

from feedloop import engine as engine_module, ledger, serving, tuning
from feedloop.server.app import DEMO_ATTRIBUTION, DEMO_SPACE, FeedloopApp, build_engine, origins_for, run_server
from feedloop.discovery import D_MIN_SIMILARITY
from feedloop.sources.filesystem import FilesystemSource, text_hash_vector

from feedloop.cli import FIXTURE_NOTE as SYNTHETIC_NOTE, FIXTURE_TAGS as DEMO_TAGS, write_fixture_sidecars as write_demo_sidecars


def fixture_segment_starts(folder, items) -> dict:
    """Per video item: {tag: start_s} read back from the labelled sidecar the runner wrote."""
    starts = {}
    for item in items:
        if item["kind"] != "video":
            continue
        sidecar = json.loads((Path(folder) / (item["path"] + ".json")).read_text(encoding="utf-8"))
        starts[(item["kind"], item["id"])] = {tag: float(segment["start_s"]) for segment in sidecar.get("segments", []) for tag in segment["tags"]}
    return starts


def expected_best_t(query, segment_starts):
    """The start of the segment whose hashed tag is most similar to the query: the same
    argmax the timed text space yields, computed from the fixture alone."""
    qvec = text_hash_vector(query)
    sims = {tag: float(qvec @ text_hash_vector(tag)) for tag in segment_starts}
    best = max(sims, key=sims.get)
    return segment_starts[best], sims[best]


def choose_query(segment_starts, candidates, floor):
    """The first fixture tag carried by at least one video whose every plausible match
    (floor minus a rounding margin) sits at a nonzero segment start; None when no tag does."""
    for tag in candidates:
        if not any(tag in starts for starts in segment_starts.values()):
            continue
        plausible = [expected_best_t(tag, starts) for starts in segment_starts.values()]
        if all(start > 0.0 for start, sim in plausible if sim >= floor - 0.01):
            return tag
    return None


class Check:
    def __init__(self):
        self.real, self.synthetic, self.failures = {}, {}, []

    def assert_real(self, name, condition, detail=None):
        self.real[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            self.failures.append(name)

    def assert_synthetic(self, name, condition, detail=None):
        self.synthetic[name] = {"pass": bool(condition), "detail": detail, "evidence": SYNTHETIC_NOTE}
        if not condition:
            self.failures.append(name)


def run(folder, state_dir, report_path) -> dict:
    folder, state_dir, report_path = Path(folder).resolve(), Path(state_dir).resolve(), Path(report_path).resolve()
    if not folder.is_dir():
        raise SystemExit("folder does not exist")
    state_dir.mkdir(parents=True, exist_ok=True)
    if state_dir == folder or folder in state_dir.parents:
        raise SystemExit("state directory must not be inside the media folder")
    check = Check()
    report = {"status": "FAIL", "folder_files": None, "real_checks": check.real, "synthetic_checks": check.synthetic, "failures": check.failures,
              "attribution": dict(DEMO_ATTRIBUTION), "production_thresholds": {"attribution_window_s": ledger.ATTRIBUTION_WINDOW_S, "tuner_ripen_s": tuning.TUNER_RIPEN_S,
                                                                                "tuner_min_trials": tuning.TUNER_MIN_TRIALS}}
    now = [time.time()]
    clock = lambda: now[0]
    # One fake clock everywhere: the ledger and tuner stamp with time.time(), so the
    # demonstration replaces that module attribute for the duration of the run.
    fake_time = types.SimpleNamespace(time=clock, monotonic=time.monotonic, perf_counter=time.perf_counter, sleep=time.sleep)
    patched = {module: module.time for module in (ledger, tuning, engine_module, serving)}
    for module in patched:
        module.time = fake_time
    server = None
    try:
        sidecars = write_demo_sidecars(folder)
        report["synthetic_sidecars_written"] = sidecars
        source, engine = build_engine(folder, state_dir, initialize=True, attribution=DEMO_ATTRIBUTION, clock=clock,
                                      config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0})
        for path in (source.db_path, Path(engine.ledger_path), Path(engine.tuner_path), source.spaces_dir):
            check.assert_real("state_inside_state_dir:" + path.name, state_dir in Path(path).resolve().parents, str(Path(path).resolve().relative_to(state_dir)))
        # real copied-file checks
        page = source.enumerate(("video", "image"), 1, 500)
        items = page["items"]
        videos = [i for i in items if i["kind"] == "video"]
        report["folder_files"] = {"total": page["total"], "videos": len(videos), "images": len(items) - len(videos)}
        check.assert_real("at_least_two_files", page["total"] >= 2, page["total"])
        check.assert_real("at_least_one_video", len(videos) >= 1, len(videos))
        check.assert_real("every_item_has_content_fingerprint", all(len(i["files"][0]["fingerprints"]) == 2 for i in items))
        check.assert_real("durations_not_invented", all(i["duration_s"] is None for i in items), "no sidecar or player duration exists yet")
        reopened = FilesystemSource(folder, state_dir, clock=clock)
        same = {(i["kind"], i["id"], i["path"]) for i in reopened.enumerate(("video", "image"), 1, 500)["items"]}
        check.assert_real("stable_identity_across_reopen", same == {(i["kind"], i["id"], i["path"]) for i in items})
        api_key = "demo-" + hashlib.sha256(str(folder).encode()).hexdigest()[:16]
        app = FeedloopApp(engine, api_key=api_key, allowed_origins=origins_for("127.0.0.1", 0), source=source, clock=clock)
        server = run_server(app, "127.0.0.1", 0)
        port = server.server_address[1]
        app.allowed_origins = origins_for("127.0.0.1", port)
        threading.Thread(target=server.serve_forever, name="verify-demo-server", daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        report["server"] = {"bind": "127.0.0.1", "port": port, "loopback_only": True}

        def call(method, path, body=None, auth=True):
            data = None if body is None else json.dumps(body).encode()
            headers = {"Content-Type": "application/json"}
            if auth:
                headers.update({"x-ai-api-key": api_key, "Origin": base})
            request = urllib.request.Request(base + path, method=method, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()

        first = items[0]
        status, served_bytes = call("GET", first["media_url"], auth=False)
        check.assert_real("media_served_byte_exact", status == 200 and hashlib.sha256(served_bytes).hexdigest() == first["files"][0]["fingerprints"][1]["value"], first["media_url"])
        status, _ = call("GET", "/media/" + first["path"], auth=False)
        state_status, _ = call("GET", "/media/../" + state_dir.name + "/source.sqlite", auth=False)
        check.assert_real("media_route_serves_catalogued_items_only", status == 404 and state_status == 404, [status, state_status])
        status, _ = call("GET", "/", auth=False)
        check.assert_real("client_served", status == 200)
        # delivery
        now[0] += 10
        status, body = call("POST", "/api/feed", {"limit": 8, "images": True, "surface": "feed", "session_id": "demo-session", "request_id": "demo-req-1", "client_request_id": "demo-client-1"})
        feed = json.loads(body)
        check.assert_real("feed_delivers_items", status == 200 and feed.get("status") == "ok" and feed["items"], feed.get("error_code") or feed.get("error_detail"))
        requests = ledger.read_evidence(engine.ledger_path, since_ts=0, through_ts=now[0] + 1)["requests"]
        check.assert_real("delivery_recorded_in_ledger", len(requests) == 1 and requests[0]["request_id"] == "demo-req-1")
        denied, _ = call("POST", "/api/feed", {"limit": 8}, auth=False)
        check.assert_real("denied_delivery_records_nothing", denied == 401 and len(ledger.read_evidence(engine.ledger_path, since_ts=0, through_ts=now[0] + 1)["requests"]) == 1)
        _, observed = call("GET", "/api/feed?limit=8&images=1&surface=feed&session_id=demo-session", auth=False)
        observed = json.loads(observed)
        check.assert_real("warm_observation_strips_delivery", observed.get("request_id") is None and observed.get("items") and all(i["served_item_id"] is None for i in observed["items"]))
        chosen = next((i for i in feed.get("items", []) if i["kind"] == "video"), None)
        check.assert_real("primary_item_served", chosen is not None)
        if chosen is None:
            raise RuntimeError("no primary item to continue with")
        # qualified view (real HTTP path; the dwell values are the policy's own thresholds)
        now[0] += 2
        status, body = call("POST", "/api/view", {"client_event_id": "demo-view-1", "session_id": "demo-session", "request_id": "demo-req-1", "served_item_id": chosen["served_item_id"],
                                                  "surface": "feed", "position": chosen["source_rank"], "dwell_ms": 1200, "visible_fraction": 0.6,
                                                  "visibility_policy": "foreground-60pct-1200ms-v1", "kind": "video", "item_id": chosen["id"], "occurred_at": now[0]})
        viewed = json.loads(body)
        check.assert_real("qualified_view_confirmed", viewed.get("status") == "confirmed", viewed.get("error_code"))
        # committed watch batch: synthetic positions and duration
        now[0] += 4
        events = [{"id": "demo-w-start", "type": "view_start", "item_id": chosen["id"], "occurred_at": now[0] - 4, "position": 0, "duration": 600.0, "viewed_event_id": viewed.get("event_id"), "previous_event_id": None, "playback_rate": 1},
                  {"id": "demo-w-progress", "type": "view_progress", "item_id": chosen["id"], "occurred_at": now[0], "position": 4, "duration": 600.0, "viewed_event_id": viewed.get("event_id"), "previous_event_id": "demo-w-start", "playback_rate": 1}]
        batch = {"capture_id": "demo-capture-1", "stream_session_id": "demo-stream-1", "session_id": "demo-session", "events": events}
        status, body = call("POST", "/api/watch", batch)
        imported = json.loads(body)
        check.assert_synthetic("watch_batch_imported_once", imported.get("status") == "imported" and imported.get("outcomes") == 1 and imported.get("stored_steps") == 2, imported)
        status, body = call("POST", "/api/watch", batch)
        check.assert_synthetic("watch_batch_replay_is_duplicate", json.loads(body).get("status") == "duplicate")
        watch_rows = source.read([("video", chosen["id"])])["rows"].get(("video", chosen["id"]), {}).get("watch")
        check.assert_synthetic("watch_interval_persisted_in_source", bool(watch_rows) and abs(watch_rows["watched_s"] - 4.0) < 1e-6, watch_rows)
        # attribution after the shortened window
        status, body = call("POST", "/api/tick")
        early = json.loads(body)
        check.assert_synthetic("attribution_waits_inside_window", early.get("attributed") == 0, early)
        now[0] += 6
        status, body = call("POST", "/api/tick")
        late = json.loads(body)
        check.assert_synthetic("attribution_credits_after_window", late.get("attributed") == 1, late)
        card = json.loads(call("GET", "/api/scorecard", auth=False)[1])
        check.assert_synthetic("scorecard_shows_capture_verified", card["capture"]["verified"] is True and card["capture"]["attributed_outcomes"] == 1, card["capture"])
        check.assert_synthetic("production_gates_unchanged_in_scorecard", card["gate"]["min_trials_per_arm"] == tuning.TUNER_MIN_TRIALS and abs(card["gate"]["ripen_hours"] * 3600 - tuning.TUNER_RIPEN_S) < 1e-6)
        # explicit rating and exact undo through the authority callbacks
        now[0] += 2
        key = ("video", chosen["id"])
        before = source.read_current(key)
        rated = json.loads(call("POST", "/api/feedback", {"operation_id": "demo-rate-1", "session_id": "demo-session", "kind": "video", "item_id": chosen["id"], "action": "rating", "rating100": 90, "request_id": "demo-req-1", "viewed_event_id": viewed.get("event_id")})[1])
        check.assert_real("like_persisted", rated.get("status") == "confirmed" and source.read_current(key)["rating100"] == 90, rated.get("error_code"))
        now[0] += 1
        undone = json.loads(call("POST", "/api/feedback", {"operation_id": "demo-undo-1", "session_id": "demo-session", "kind": "video", "item_id": chosen["id"], "action": "undo", "undo_of": "demo-rate-1", "request_id": "demo-req-1", "viewed_event_id": viewed.get("event_id")})[1])
        check.assert_real("undo_restores_exact_original", undone.get("status") == "confirmed" and source.read_current(key) == before, undone.get("error_code"))
        # search and similar over the labelled sidecar-text space
        segment_starts = fixture_segment_starts(folder, items)
        query = choose_query(segment_starts, DEMO_TAGS, D_MIN_SIMILARITY)
        check.assert_synthetic("search_query_has_nonzero_fixture_segment_starts", query is not None, {"candidates": DEMO_TAGS, "videos_with_segments": len(segment_starts)})
        query = query or DEMO_TAGS[0]
        found = json.loads(call("GET", "/api/search?q=" + query, auth=False)[1])
        meta = source.space_meta(DEMO_SPACE) or {}
        windows = source.windows(DEMO_SPACE)
        check.assert_synthetic("search_returns_results_over_labelled_timed_text_space", found.get("status") == "ok" and found["items"] and "not a measurement" in meta.get("provenance", "")
                               and meta.get("window_scope") == "timed" and windows is not None and len(windows[0]) == 3 * len(videos)
                               and found == json.loads(json.dumps(engine.search(query, "look"))),
                               {"status": found.get("status"), "items": len(found.get("items", [])), "provenance": meta.get("provenance"), "windows": None if windows is None else len(windows[0])})
        results = []
        for item in found.get("items", []):
            starts = segment_starts.get((item["kind"], item["id"]), {})
            expected = expected_best_t(query, starts)[0] if starts else None
            results.append({"kind": item["kind"], "id": item["id"], "title": item["title"], "best_t": item["search"]["best_t"], "expected_best_t": expected,
                            "carries_query": query in starts, "query_segment_start": starts.get(query)})
        report["search"] = {"query": query, "results": results, "fixture_segment_starts": {f"{kind}:{item_id}": starts for (kind, item_id), starts in segment_starts.items()},
                            "evidence": SYNTHETIC_NOTE}
        real_positions = all(isinstance(r["best_t"], float) and np.isfinite(r["best_t"]) and r["best_t"] > 0.0 and r["best_t"] == r["expected_best_t"] for r in results)
        check.assert_synthetic("search_positions_are_nonzero_fixture_segment_starts", bool(results) and real_positions
                               and any(r["carries_query"] and r["best_t"] == r["query_segment_start"] for r in results), results)
        # an invalid player batch is refused before anything authoritative is written
        rejected = json.loads(call("POST", "/api/watch", {"capture_id": "demo-capture-bad", "stream_session_id": "demo-stream-1", "session_id": "demo-session", "events": [
            {"id": "demo-w-bad", "type": "view_progress", "item_id": chosen["id"], "occurred_at": now[0], "position": 500, "duration": 600.0, "viewed_event_id": viewed.get("event_id"),
             "previous_event_id": "demo-w-progress", "playback_rate": 1}]})[1])
        check.assert_synthetic("invalid_watch_batch_rejected_without_writes", rejected.get("status") == "rejected" and rejected.get("stored_steps") == 0
                               and source.read([key])["rows"][key]["watch"]["watched_s"] == 4.0, rejected)
        alike = json.loads(call("GET", f"/api/similar?kind=video&id={chosen['id']}", auth=False)[1])
        check.assert_synthetic("similar_returns_results", (alike.get("status") == "ok" and alike["items"]) if len(videos) >= 2 else alike.get("status") in ("empty", "ok"),
                               {"status": alike.get("status"), "items": len(alike.get("items", [])), "videos_in_folder": len(videos)})
        # stale cursor is explicit and records no delivery
        now[0] += 1
        cursor = dict(feed["pagination"]["next_cursor"] or {"generation_id": "0" * 64, "offset": len(feed["items"]), "after": "video:1"}, generation_id="0" * 64)
        stale = json.loads(call("POST", "/api/feed", {"limit": 8, "images": True, "surface": "feed", "session_id": "demo-session", "request_id": "demo-req-stale", "client_request_id": "demo-client-stale",
                                                      "offset": cursor["offset"], "cursor": cursor})[1])
        check.assert_real("stale_cursor_explicit_without_delivery", stale.get("status") == "error" and stale.get("error_code") == "stale_ranking_cursor" and
                          len(ledger.read_evidence(engine.ledger_path, since_ts=0, through_ts=now[0] + 1)["requests"]) == 1, stale.get("error_code"))
        # synthetic tuner trials: a real mix promotes, a collapsed candidate stalls; both visible and the promotion reversible
        rng = random.Random(3)

        def summary(cand_categories):
            trials, grouped = [], {}
            for s in range(20):
                for i in range(4):
                    for arm, reward in (("base", .2 + rng.uniform(-.05, .05)), ("cand", .5 + rng.uniform(-.05, .05))):
                        pool = ["acts", "other"] if arm == "base" else cand_categories
                        trial = {"arm": arm, "reward": reward, "category": pool[(s * 4 + i) % len(pool)], "viewed_id": f"{arm}-{s}-{i}", "kind": "video", "item_id": s * 10 + i}
                        trials.append(trial)
                        grouped.setdefault(f"synthetic-session-{s}", []).append(trial)
            return {"status": "ok", "valid": True, "validity_reasons": [], "trials": trials, "sessions": grouped}
        action, detail = tuning.decide(summary(["acts", "other"]), now=now[0], window_started=now[0] - 30 * 86400)
        check.assert_synthetic("synthetic_mixed_trials_promote", action == "promote", detail.get("reason"))
        stalled, stall_detail = tuning.decide(summary(["other"]), now=now[0], window_started=now[0] - 30 * 86400)
        check.assert_synthetic("synthetic_collapsed_trials_stall_entropy_veto", (stalled, stall_detail.get("reason")) == ("stall", "entropy_veto"), stall_detail.get("reason"))
        labelled = {**detail, "demonstration": True, "evidence_source": SYNTHETIC_NOTE}
        with engine.tuner.lock:
            revision = engine.tuner.revision[0]
        promoted = engine.tuner.promote(labelled, expected_revision=revision)
        engine.tuner.stall(json.dumps({**stall_detail, "demonstration": True, "evidence_source": SYNTHETIC_NOTE}))
        card = json.loads(call("GET", "/api/scorecard", auth=False)[1])
        applied = [row for row in card["tuner"]["ledger"] if row["status"] == "applied" and row["evidence"].get("demonstration") is True]
        check.assert_synthetic("promotion_visible_in_ledger_labelled", promoted is True and len(applied) == 1, [(r["knob"], r["old"], r["new"]) for r in applied])
        check.assert_synthetic("stall_visible_in_ledger", any(row["status"] == "no_verdict" for row in card["tuner"]["ledger"]))
        reverted = json.loads(call("POST", f"/api/tuner/revert?ledger_id={applied[0]['id']}")[1]) if applied else {"ok": False}
        card = json.loads(call("GET", "/api/scorecard", auth=False)[1])
        settled = card["tuner"]["tuned_values"].get(applied[0]["knob"]) if applied else None
        check.assert_synthetic("promotion_reverted_through_http_control", reverted.get("ok") is True and applied and settled == applied[0]["old"] and
                               any(row["id"] == applied[0]["id"] and row["status"] == "reverted" for row in card["tuner"]["ledger"]), reverted)
        denied_control, _ = call("POST", "/api/tuner/reset", auth=False)
        check.assert_real("denied_tuner_control_refused", denied_control == 401)
        report["ledger_rows"] = len(card["tuner"]["ledger"])
    except Exception:
        check.failures.append("exception")
        report["exception"] = traceback.format_exc().splitlines()[-1]
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        for module, original in patched.items():
            module.time = original
        report["server_stopped"] = True
    report["status"] = "PASS" if not check.failures else "FAIL"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folder", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    report = run(args.folder, args.state_dir, args.report)
    summary = {"status": report["status"], "folder_files": report.get("folder_files"), "failures": report["failures"],
               "real_checks": sum(1 for v in report["real_checks"].values() if v["pass"]), "synthetic_checks": sum(1 for v in report["synthetic_checks"].values() if v["pass"])}
    print(json.dumps(summary))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

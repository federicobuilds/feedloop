#!/usr/bin/env python3
"""Browser contract runner for the no-build client.

Two processes: the browser interpreter (with Playwright) runs this script as the
runner, and the project interpreter runs it as the harness that serves the public
client over the real server on loopback with a synthetic neutral folder.

    check_client.py --python /path/to/project/venv/python --output DIR

The harness exposes a loopback control endpoint used only by the runner: recorded
deliveries, dropping published generations (the exact condition the server reports as
``stale_ranking_cursor``), reading an item's authority state, and one labelled
synthetic tuner promotion. Failure states are forced by intercepting responses in the
browser. Screenshots and the JSON report go under --output. Exit code is nonzero on
any failed assertion.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


# ----------------------------------------------------------------- harness
def harness(output: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from fl3_helpers import make_folder
    from feedloop.server.app import DEMO_ATTRIBUTION, FeedloopApp, build_engine, origins_for, run_server
    output.mkdir(parents=True, exist_ok=True)
    media = make_folder(output / "synthetic", videos=40, images=6, segments=True)
    source, engine = build_engine(media, output / "state", initialize=True, attribution=DEMO_ATTRIBUTION,
                                  config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0})
    api_key = "browser-check-key"
    app = FeedloopApp(engine, api_key=api_key, allowed_origins=origins_for("127.0.0.1", 0), source=source)
    server = run_server(app, "127.0.0.1", 0)
    port = server.server_address[1]
    app.allowed_origins = origins_for("127.0.0.1", port)
    deliveries = []
    real_feed = engine.feed

    def recording_feed(payload):
        result = real_feed(payload)
        deliveries.append({"request": json.loads(json.dumps(payload)), "status": result.get("status"), "error_code": result.get("error_code"),
                           "pagination": result.get("pagination"), "count": len(result.get("items", []))})
        return result
    engine.feed = recording_feed
    counters = {"feed": 0, "rank": 0, "view": 0, "sqlite": 0, "search": []}
    real_rank, real_view, real_search = engine._rank, engine.view, engine.search
    import sqlite3
    real_connect = sqlite3.connect

    def counting(name, function):
        def wrapped(*args, **kwargs):
            counters[name] += 1
            return function(*args, **kwargs)
        return wrapped
    engine._rank = counting("rank", real_rank)
    engine.view = counting("view", real_view)
    sqlite3.connect = counting("sqlite", real_connect)

    def recording_search(query, mode="look", **kwargs):
        counters["search"].append({"q": query, "mode": mode})
        return real_search(query, mode, **kwargs)
    engine.search = recording_search
    original_feed_wrapper = engine.feed
    engine.feed = counting("feed", original_feed_wrapper)

    class Control(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, value):
            data = json.dumps(value, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            if parts.path == "/deliveries":
                self._send(deliveries)
            elif parts.path == "/read_current":
                self._send(source.read_current((query["kind"][0], int(query["id"][0]))))
            elif parts.path == "/ledger":
                self._send(engine.scorecard()["tuner"]["ledger"])
            elif parts.path == "/counters":
                self._send(counters)
            else:
                self._send({"error": "unknown"})

        def do_POST(self):
            if self.path == "/stale":
                with engine.lock:
                    engine._cursors.clear()
                self._send({"ok": True})
            elif self.path == "/promote":
                with engine.tuner.lock:
                    revision = engine.tuner.revision[0]
                ok = engine.tuner.promote({"winner": "cand", "demonstration": True, "evidence_source": "synthetic browser-check fixture"}, expected_revision=revision)
                self._send({"ok": ok})
            else:
                self._send({"error": "unknown"})
    control = ThreadingHTTPServer(("127.0.0.1", 0), Control)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=control.serve_forever, daemon=True).start()
    print(json.dumps({"base": f"http://127.0.0.1:{port}", "control": f"http://127.0.0.1:{control.server_address[1]}", "key": api_key}), flush=True)
    try:
        sys.stdin.read()
    finally:
        server.shutdown()
        server.server_close()
        control.shutdown()
        control.server_close()
    return 0


# ------------------------------------------------------------------ runner
class Checks:
    def __init__(self):
        self.rows, self.failures = [], []

    def check(self, name, condition, detail=None):
        self.rows.append({"name": name, "pass": bool(condition), "detail": detail})
        if not condition:
            self.failures.append(name)


def control_call(url, method="GET"):
    request = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def failure_response(kind):
    bodies = {"unavailable": (503, {"detail": "ranking_unavailable"}), "error": (200, {"status": "error", "error_code": "ranking_contract_unavailable", "items": []}),
              "empty": (200, {"status": "empty", "items": [], "pagination": {"has_more": False}}),
              "partial": (200, {"status": "partial", "items": [], "components": {"voice": {"status": "unavailable"}}, "pagination": {"has_more": False}})}
    return bodies[kind]


def runner(python: str, output: Path) -> int:
    from playwright.sync_api import sync_playwright
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    process = subprocess.Popen([python, str(Path(__file__).resolve()), "--harness", "--output", str(output)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    line = process.stdout.readline()
    if not line:
        print("harness failed to start", file=sys.stderr)
        return 1
    info = json.loads(line)
    base, control, api_key = info["base"], info["control"], info["key"]
    checks, console = Checks(), []
    report = {"status": "FAIL", "base": base, "checks": checks.rows, "failures": checks.failures, "console": console, "screenshots": []}

    def shot(page, name):
        path = output / f"client-{name}.png"
        page.screenshot(path=str(path))
        report["screenshots"].append(str(path))

    def wait_for(predicate, timeout=10.0, step=0.05):
        deadline = time.time() + timeout
        while time.time() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(step)
        return predicate()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("console", lambda m: console.append({"type": m.type, "text": m.text}) if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e: console.append({"type": "pageerror", "text": str(e)}))
            # 1. Feed building panel under a fake browser clock on a throwaway page: counter, 20 s note, live region
            probe = browser.new_page(viewport={"width": 1280, "height": 900})
            probe.clock.install()
            probe.route("**/api/feed", lambda route: None)
            probe.goto(base + "/#key=" + api_key)
            probe.wait_for_selector("[data-ai-feed] [data-ai-state='loading']", timeout=10000)
            checks.check("feed_building_panel", probe.text_content("[data-ai-building-title]") == "Building your recommendations"
                         and probe.text_content("[data-ai-state-message]") == "Loading recommendations..." and probe.text_content("[data-ai-building-elapsed]") == "0 s"
                         and probe.eval_on_selector("[data-ai-building-note]", "e => getComputedStyle(e).visibility") == "hidden")
            probe.clock.fast_forward(21000)
            checks.check("feed_building_note_after_20s", probe.text_content("[data-ai-building-elapsed]") == "21 s"
                         and probe.eval_on_selector("[data-ai-building-note]", "e => getComputedStyle(e).visibility") == "visible", probe.text_content("[data-ai-building-elapsed]"))
            checks.check("key_not_in_address", "key=" not in probe.url and probe.evaluate("sessionStorage.getItem('feedloop-key')") == api_key, probe.url)
            probe.evaluate("location.hash = '#/home'")
            probe.wait_for_selector("[data-ai-home-status][data-ai-state='loading'] [data-ai-building]", timeout=10000)
            probe.clock.fast_forward(21000)
            home_panel = probe.evaluate("""(() => { const s = document.querySelector('[data-ai-home-status]'); const p = s.querySelector('[data-ai-building]');
              const line = s.querySelector('.ai-status-line'); const r = p.getBoundingClientRect();
              return { elapsed: p.querySelector('[data-ai-building-elapsed]').textContent, note: getComputedStyle(p.querySelector('[data-ai-building-note]')).visibility,
                       live: s.querySelector('[data-ai-state-message]').textContent, lineHidden: line.getBoundingClientRect().width <= 1, centered: Math.abs((r.left + r.width / 2) - (s.getBoundingClientRect().left + s.getBoundingClientRect().width / 2)) < 2, tall: r.height >= 200 }; })()""")
            checks.check("home_building_panel_clock_and_layout", home_panel["elapsed"] == "21 s" and home_panel["note"] == "visible" and home_panel["live"] == "Loading recommendations..."
                         and home_panel["lineHidden"] and home_panel["centered"] and home_panel["tall"], home_panel)
            probe.close()
            # cold observation before any delivery: no ranking, no feed call, no store opened
            before = control_call(f"{control}/counters")
            cold = json.loads(urllib.request.urlopen(base + "/api/feed?limit=8&images=1&surface=feed&session_id=nobody", timeout=10).read())
            after = control_call(f"{control}/counters")
            checks.check("cold_observation_tripwire", cold["error_code"] == "feed_observation_unavailable" and cold["items"] == [] and after["feed"] == before["feed"]
                         and after["rank"] == before["rank"] and after["sqlite"] == before["sqlite"], [cold.get("error_code"), before, after])
            page.goto(base + "/#key=" + api_key)
            page.wait_for_selector("[data-idx='0']", timeout=15000)
            cells = page.query_selector_all("[data-idx]")
            checks.check("feed_renders_cells_and_clears_timer", len(cells) >= 20 and page.query_selector("[data-ai-feed] [data-ai-building]") is None
                         and page.evaluate("Array.from(document.querySelectorAll('[data-ai-feed] > *')).every(e => e.__buildingTimer == null)"), len(cells))
            session_id = page.evaluate("sessionStorage.getItem('feedloop-session')")
            before = control_call(f"{control}/counters")
            warm = json.loads(urllib.request.urlopen(base + "/api/feed?limit=24&images=1&surface=feed&session_id=" + session_id, timeout=10).read())
            after = control_call(f"{control}/counters")
            checks.check("warm_observation_tripwire", warm["status"] == "ok" and warm["items"] and warm["request_id"] is None and all(i["served_item_id"] is None for i in warm["items"])
                         and after["feed"] == before["feed"] and after["rank"] == before["rank"] and after["sqlite"] == before["sqlite"], [warm.get("status"), before, after])
            first = cells[0]
            body = first.query_selector("details.ai-explanation .ai-explain-body")
            raw = first.query_selector("details.ai-explanation-raw")
            text, raw_text = body.text_content() if body else "", raw.text_content() if raw else ""
            checks.check("explanation_words_and_not_measured", body is not None and "not measured" in text and "Why this item" in first.text_content(), text[:160])
            checks.check("raw_json_only_in_closed_technical_details", raw is not None and not raw.evaluate("e => e.open") and "explanation" in raw_text.lower()
                         and "{" not in text.replace(raw_text, ""))
            shot(page, "feed")
            # 2. Measured view after the real 1,200 ms dwell
            viewed = wait_for(lambda: first.get_attribute("data-ai-viewed"), timeout=8)
            checks.check("qualified_view_confirmed", viewed is not None and not viewed.startswith("unconfirmed"), viewed)
            # 2b. The qualification timer never posts for a disposed card or a hidden page
            views_before = control_call(f"{control}/counters")["view"]
            page.evaluate("location.hash = '#/engine'")
            page.wait_for_selector("[data-ai-capture-status]", timeout=15000)
            page.evaluate("location.hash = '#/feed'")
            page.wait_for_selector("[data-idx='0']", timeout=15000)
            page.evaluate("location.hash = '#/engine'")  # dispose within the 1,200 ms dwell
            page.wait_for_selector("[data-ai-capture-status]", timeout=15000)
            page.wait_for_timeout(1600)
            views_after_dispose = control_call(f"{control}/counters")["view"]
            page.evaluate("Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' })")
            page.evaluate("location.hash = '#/feed'")
            page.wait_for_selector("[data-idx='0']", timeout=15000)
            page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
            page.wait_for_timeout(1600)
            views_after_hidden = control_call(f"{control}/counters")["view"]
            page.evaluate("delete document.visibilityState")
            checks.check("view_timer_cancelled_on_dispose_and_hidden", views_after_dispose == views_before and views_after_hidden == views_before
                         and page.query_selector("[data-idx='0'][data-ai-viewed]") is None, [views_before, views_after_dispose, views_after_hidden])
            page.evaluate("location.hash = '#/engine'")
            page.wait_for_selector("[data-ai-capture-status]", timeout=15000)
            page.evaluate("location.hash = '#/feed'")
            page.wait_for_selector("[data-idx='0']", timeout=15000)
            cells = page.query_selector_all("[data-idx]")
            first = cells[0]
            wait_for(lambda: first.get_attribute("data-ai-viewed"), timeout=8)
            # 3. Feedback confirmed from the ledger result; undo restores the exact original
            kind, item_id = first.get_attribute("data-ai-key").split(":")
            before = control_call(f"{control}/read_current?kind={kind}&id={item_id}")
            first.query_selector("[data-ai-feedback='like']").click()
            page.wait_for_selector("[data-idx='0'] [data-ai-feedback-status='confirmed']", timeout=8000)
            after = control_call(f"{control}/read_current?kind={kind}&id={item_id}")
            checks.check("like_confirmed_and_persisted", after["rating100"] == 90 and first.query_selector("[data-ai-feedback='like']").get_attribute("aria-pressed") == "true", after)
            first.query_selector("[data-ai-undo]").click()
            page.wait_for_selector("[data-idx='0'] [data-ai-feedback-status='undone']", timeout=8000)
            checks.check("undo_restores_original", control_call(f"{control}/read_current?kind={kind}&id={item_id}") == before)
            # 3b. Unresolved feedback: the server confirms but the reply is lost; new actions stay blocked until Check status,
            #     which confirms and keeps the receipt so Undo still works
            page.route("**/api/feedback", lambda route: (route.fetch(), route.fulfill(status=503, content_type="application/json", body='{"detail":"lost"}'))[1])
            first.query_selector("[data-ai-feedback='like']").click()
            page.wait_for_selector("[data-idx='0'] [data-ai-feedback-status='uncertain']", timeout=8000)
            page.unroute("**/api/feedback")
            blocked = page.evaluate("Array.from(document.querySelectorAll(\"[data-idx='0'] [data-ai-feedback]\")).every(b => b.getAttribute('aria-disabled') === 'true')")
            checks.check("unresolved_feedback_blocks_new_actions", blocked and page.query_selector("[data-idx='0'] [data-ai-check-status]") is not None
                         and control_call(f"{control}/read_current?kind={kind}&id={item_id}")["rating100"] == 90)
            # reopening the view keeps the control and the block (the cell is found by key: the pending rating reorders the page)
            cell = f"[data-ai-key='{kind}:{item_id}']"
            page.evaluate("location.hash = '#/engine'")
            page.wait_for_selector("[data-ai-capture-status]", timeout=15000)
            page.evaluate("location.hash = '#/feed'")
            page.wait_for_selector(cell, timeout=15000)
            reopened_blocked = page.evaluate(f"Array.from(document.querySelectorAll(\"{cell} [data-ai-feedback]\")).every(b => b.getAttribute('aria-disabled') === 'true')")
            checks.check("reopened_unresolved_keeps_check_status", reopened_blocked and page.query_selector(f"{cell} [data-ai-check-status]") is not None
                         and page.get_attribute(f"{cell} .feedback-row", "data-ai-feedback-unresolved") == "true")
            page.click(f"{cell} [data-ai-check-status]")
            page.wait_for_selector(f"{cell} [data-ai-feedback-status='confirmed']", timeout=8000)
            first = page.query_selector(cell)
            checks.check("reconciled_receipt_enables_undo", page.get_attribute(f"{cell} .feedback-row", "data-ai-feedback-unresolved") == "false"
                         and not first.query_selector("[data-ai-undo]").evaluate("e => e.hidden") and first.query_selector("[data-ai-feedback='like']").get_attribute("aria-disabled") == "false")
            first.query_selector("[data-ai-undo]").click()
            page.wait_for_selector(f"{cell} [data-ai-feedback-status='undone']", timeout=8000)
            checks.check("undo_after_reconciliation_restores_original", control_call(f"{control}/read_current?kind={kind}&id={item_id}") == before)
            def empty_restart(route):
                response = route.fetch()
                payload = response.json()
                request_body = json.loads(route.request.post_data or "{}")
                if request_body.get("offset") == 0 and "cursor" not in request_body and payload.get("items"):
                    payload["items"] = []
                route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

            # 4. Feed stale cursor: a fresh Feed with a usable continuation, then drop generations: one cursorless restart, kept cells
            page.evaluate("location.hash = '#/engine'")
            page.wait_for_selector("[data-ai-capture-status]", timeout=15000)
            page.evaluate("location.hash = '#/feed'")
            page.wait_for_selector("[data-idx='0']", timeout=15000)
            page.wait_for_function("document.querySelector(\"[data-ai-feed] [data-ai-state='loading']\") === null", timeout=10000)
            feed_state = page.evaluate("(() => { const s = window.feedloop.current.handle.state; return { hasMore: s.hasMore, cursor: !!s.cursor, items: s.items.length }; })()")
            checks.check("feed_has_usable_continuation", feed_state["hasMore"] and feed_state["cursor"] and feed_state["items"] >= 20, feed_state)
            cells_before = len(page.query_selector_all("[data-idx]"))
            sent = len(control_call(f"{control}/deliveries"))
            control_call(f"{control}/stale", "POST")

            def scroll_feed():
                # one synthetic scroll event at the bottom; the browser may add its own, so re-arm checks count restarts, not requests
                page.evaluate("(() => { const c = document.querySelector('[data-ai-feed]'); c.scrollTop = c.scrollHeight; c.dispatchEvent(new Event('scroll')); })()")
            scroll_feed()
            new = wait_for(lambda: (lambda d: d if len(d) >= 2 else None)(control_call(f"{control}/deliveries")[sent:]), timeout=10) or control_call(f"{control}/deliveries")[sent:]
            page.wait_for_function("document.querySelector(\"[data-ai-feed] [data-ai-state='loading']\") === null", timeout=10000)
            checks.check("feed_stale_cursor_restarts_once_cursorless", len(new) >= 2 and new[0]["error_code"] == "stale_ranking_cursor" and new[0]["request"].get("cursor")
                         and new[1]["request"]["offset"] == 0 and "cursor" not in new[1]["request"] and new[1]["request"]["request_id"] != new[0]["request"]["request_id"]
                         and new[1]["request"]["session_id"] == new[0]["request"]["session_id"], [(d["request"].get("offset"), d["error_code"]) for d in new])
            checks.check("feed_keeps_cells_and_dedupes", page.query_selector("[data-idx='0']") is not None and len(page.query_selector_all("[data-idx]")) >= cells_before
                         and len(page.query_selector_all("[data-idx]")) == len({e.get_attribute("data-ai-key") for e in page.query_selector_all("[data-idx]")}))
            # 4b. A restart that adds nothing earns no second restart; manual Retry re-arms exactly one (restart response intercepted to add nothing)
            page.route("**/api/feed", empty_restart)
            control_call(f"{control}/stale", "POST")
            sent = len(control_call(f"{control}/deliveries"))
            scroll_feed()
            wait_for(lambda: len(control_call(f"{control}/deliveries")) >= sent + 2, timeout=10)
            page.wait_for_function("document.querySelector(\"[data-ai-feed] [data-ai-state='loading']\") === null", timeout=10000)
            first_round = control_call(f"{control}/deliveries")[sent:]
            page.unroute("**/api/feed")
            control_call(f"{control}/stale", "POST")
            sent = len(control_call(f"{control}/deliveries"))
            scroll_feed()
            page.wait_for_selector("[data-ai-feed] [data-ai-state='error'] [data-ai-retry]", timeout=10000)
            refused = control_call(f"{control}/deliveries")[sent:]
            checks.check("feed_second_stale_refusal_does_not_rearm", len(first_round) >= 2 and first_round[0]["error_code"] == "stale_ranking_cursor" and first_round[1]["request"]["offset"] == 0
                         and 1 <= len(refused) <= 2 and all(d["error_code"] == "stale_ranking_cursor" and d["request"].get("cursor") and d["request"]["offset"] != 0 for d in refused)
                         and page.query_selector("[data-idx='0']") is not None,
                         [[(d["request"].get("offset"), d["error_code"]) for d in first_round], [(d["request"].get("offset"), d["error_code"]) for d in refused]])
            sent = len(control_call(f"{control}/deliveries"))
            page.click("[data-ai-feed] [data-ai-retry]")
            wait_for(lambda: len(control_call(f"{control}/deliveries")) > sent, timeout=10)
            page.wait_for_function("document.querySelector(\"[data-ai-feed] [data-ai-state='loading']\") === null", timeout=10000)
            retried = control_call(f"{control}/deliveries")[sent:]
            control_call(f"{control}/stale", "POST")
            sent = len(control_call(f"{control}/deliveries"))
            scroll_feed()
            rearmed = wait_for(lambda: (lambda d: d if len(d) >= 2 else None)(control_call(f"{control}/deliveries")[sent:]), timeout=10) or control_call(f"{control}/deliveries")[sent:]
            page.wait_for_function("document.querySelector(\"[data-ai-feed] [data-ai-state='loading']\") === null", timeout=10000)
            checks.check("feed_retry_rearms_exactly_one_restart", len(retried) >= 1 and retried[0]["request"]["offset"] == 0 and "cursor" not in retried[0]["request"] and retried[0]["status"] == "ok"
                         and len(rearmed) >= 2 and rearmed[0]["error_code"] == "stale_ranking_cursor" and rearmed[1]["request"]["offset"] == 0 and "cursor" not in rearmed[1]["request"],
                         [[(d["request"].get("offset"), d["error_code"]) for d in retried], [(d["request"].get("offset"), d["error_code"]) for d in rearmed]])
            # 5. Honest failure states on Feed (intercepted responses), each with Retry
            def open_view(name):
                if page.evaluate("location.hash") == "#/" + name:
                    page.evaluate("window.feedloop.route()")
                else:
                    page.evaluate(f"location.hash = '#/{name}'")

            def fulfil(status, payload):
                return lambda route: route.fulfill(status=status, content_type="application/json", body=json.dumps(payload))
            for state in ("unavailable", "error", "empty", "partial"):
                page.route("**/api/feed", fulfil(*failure_response(state)))
                open_view("home")
                for _ in range(100):
                    page.wait_for_timeout(100)
                    shown = page.evaluate("(() => { const e = document.querySelector('[data-ai-home-status][data-ai-state]'); return e && e.getAttribute('data-ai-state'); })()")
                    if shown and shown != "loading":
                        break
                message = page.evaluate("(() => { const e = document.querySelector('[data-ai-home-status] [data-ai-state-message]'); return e && e.textContent; })()")
                checks.check(f"home_state_{state}", shown == state and bool(message) and page.query_selector("[data-ai-home-status] [data-ai-retry]") is not None
                             and page.query_selector("[data-ai-home-status] [data-ai-building]") is None, [shown, message])
                page.unroute("**/api/feed")
            page.route("**/api/search**", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "no-feature", "items": [], "components": {"look": {"status": "no-feature"}}})))
            page.evaluate("location.hash = '#/search?q=amber'")
            page.wait_for_selector("[data-ai-search-status][data-ai-state='no-feature']", timeout=10000)
            checks.check("search_state_no_feature", "No usable feature" in page.text_content("[data-ai-search-status] [data-ai-state-message]"))
            page.unroute("**/api/search**")
            # 6. Home: shelves, skip-aware offset forwarding, stale restart keeps card identity
            open_view("home")
            page.wait_for_selector("[data-ai-home-key]", timeout=15000)
            page.evaluate("window.__keptCard = document.querySelector('[data-ai-home-key]')")
            kept = page.evaluate("window.__keptCard.getAttribute('data-ai-home-key')")
            count_before = len(page.query_selector_all("[data-ai-home-key]"))
            home_first = control_call(f"{control}/deliveries")[-1]
            control_call(f"{control}/stale", "POST")
            sent = len(control_call(f"{control}/deliveries"))
            page.click("[data-ai-load-more]")
            new = wait_for(lambda: (lambda d: d if len(d) >= 2 else None)(control_call(f"{control}/deliveries")[sent:]), timeout=10) or control_call(f"{control}/deliveries")[sent:]
            page.wait_for_function("document.querySelector('[data-ai-home-status][data-ai-state=\"loading\"]') === null", timeout=10000)
            checks.check("home_forwards_next_offset", home_first["pagination"]["has_more"] is True and new and new[0]["request"]["offset"] == home_first["pagination"]["next_offset"]
                         and new[0]["request"]["cursor"] == home_first["pagination"]["next_cursor"], [home_first["pagination"].get("next_offset"), new[0]["request"].get("offset") if new else None])
            checks.check("home_stale_cursor_restarts_once_cursorless", len(new) == 2 and new[0]["error_code"] == "stale_ranking_cursor" and new[1]["request"]["offset"] == 0
                         and "cursor" not in new[1]["request"] and new[1]["request"]["request_id"] != new[0]["request"]["request_id"] and new[1]["request"]["session_id"] == new[0]["request"]["session_id"],
                         [(d["request"].get("offset"), d["error_code"]) for d in new])
            checks.check("home_keeps_card_identity_and_dedupes", page.evaluate("document.querySelector('[data-ai-home-key]') === window.__keptCard")
                         and len(page.query_selector_all(f"[data-ai-home-key='{kept}']")) == 1 and len(page.query_selector_all("[data-ai-home-key]")) >= count_before
                         and page.get_attribute("[data-ai-load-more]", "aria-disabled") == "false")
            # 6b. A restart that adds nothing earns no second restart; manual Retry re-arms exactly one.
            #     The restart response is intercepted to carry no new cards, the deterministic "added nothing" case.
            page.route("**/api/feed", empty_restart)
            cards_before = len(page.query_selector_all("[data-ai-home-key]"))
            control_call(f"{control}/stale", "POST")
            sent = len(control_call(f"{control}/deliveries"))
            page.click("[data-ai-load-more]")
            wait_for(lambda: len(control_call(f"{control}/deliveries")) >= sent + 2, timeout=10)
            page.wait_for_function("document.querySelector('[data-ai-home-status][data-ai-state=\"loading\"]') === null", timeout=10000)
            first_round = control_call(f"{control}/deliveries")[sent:]
            page.unroute("**/api/feed")
            control_call(f"{control}/stale", "POST")
            sent = len(control_call(f"{control}/deliveries"))
            page.click("[data-ai-load-more]")
            page.wait_for_selector("[data-ai-home-status][data-ai-state='error'] [data-ai-retry]", timeout=10000)
            refused = control_call(f"{control}/deliveries")[sent:]
            checks.check("home_second_stale_refusal_does_not_rearm", len(first_round) == 2 and first_round[0]["error_code"] == "stale_ranking_cursor" and first_round[1]["request"]["offset"] == 0
                         and len(page.query_selector_all("[data-ai-home-key]")) == cards_before and len(refused) == 1 and refused[0]["error_code"] == "stale_ranking_cursor",
                         [[(d["request"].get("offset"), d["error_code"]) for d in first_round], [(d["request"].get("offset"), d["error_code"]) for d in refused]])
            sent = len(control_call(f"{control}/deliveries"))
            page.click("[data-ai-home-status] [data-ai-retry]")
            page.wait_for_function("document.querySelector('[data-ai-home-status][data-ai-state=\"loading\"]') === null", timeout=10000)
            retried = control_call(f"{control}/deliveries")[sent:]
            control_call(f"{control}/stale", "POST")
            sent = len(control_call(f"{control}/deliveries"))
            page.click("[data-ai-load-more]")
            rearmed = wait_for(lambda: (lambda d: d if len(d) >= 2 else None)(control_call(f"{control}/deliveries")[sent:]), timeout=10) or control_call(f"{control}/deliveries")[sent:]
            page.wait_for_function("document.querySelector('[data-ai-home-status][data-ai-state=\"loading\"]') === null", timeout=10000)
            checks.check("home_retry_rearms_exactly_one_restart", len(retried) == 1 and retried[0]["request"]["offset"] == 0 and "cursor" not in retried[0]["request"] and retried[0]["status"] == "ok"
                         and len(rearmed) == 2 and rearmed[0]["error_code"] == "stale_ranking_cursor" and rearmed[1]["request"]["offset"] == 0 and "cursor" not in rearmed[1]["request"],
                         [[(d["request"].get("offset"), d["error_code"]) for d in retried], [(d["request"].get("offset"), d["error_code"]) for d in rearmed]])
            # 6c. Skip-aware forwarding with a genuine gap: the served page reports next_offset beyond offset + items
            def with_gap(route):
                response = route.fetch()
                payload = response.json()
                if payload.get("pagination") and payload["pagination"].get("has_more"):
                    payload["pagination"]["next_offset"] = payload["pagination"]["next_offset"] + 3
                    payload["pagination"]["next_cursor"] = dict(payload["pagination"]["next_cursor"], offset=payload["pagination"]["next_offset"])
                route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))
            page.route("**/api/feed", with_gap)
            open_view("home")
            page.wait_for_selector("[data-ai-home-key]", timeout=15000)
            page.wait_for_function("document.querySelector('[data-ai-home-status][data-ai-state=\"loading\"]') === null", timeout=10000)
            page.unroute("**/api/feed")
            gap_first = control_call(f"{control}/deliveries")[-1]
            expected_offset = gap_first["pagination"]["next_offset"] + 3
            sent = len(control_call(f"{control}/deliveries"))
            page.click("[data-ai-load-more]")
            wait_for(lambda: len(control_call(f"{control}/deliveries")) > sent, timeout=10)
            page.wait_for_function("document.querySelector('[data-ai-home-status][data-ai-state=\"loading\"]') === null", timeout=10000)
            forwarded = control_call(f"{control}/deliveries")[sent]
            checks.check("home_forwards_skip_gap_offset", gap_first["pagination"]["has_more"] is True and forwarded["request"]["offset"] == expected_offset
                         and forwarded["request"]["cursor"]["offset"] == expected_offset, [gap_first["pagination"].get("next_offset"), forwarded["request"].get("offset")])
            shot(page, "home")
            # 7. Search and Similar over the labelled text space
            page.evaluate("location.hash = '#/search'")
            page.fill("#ai-search-query", "amber")
            page.click("form.search-form button[type='submit']")
            page.wait_for_selector("[data-ai-search-grid] .card", timeout=15000)
            checks.check("search_results_with_fixture_segment_positions", len(page.query_selector_all("[data-ai-search-grid] .card")) >= 1
                         and "Matching moment" in page.inner_text("[data-ai-search-grid]") and "position not measured" not in page.inner_text("[data-ai-search-grid]"))
            # 7b. Sound and Both dispatch once in the chosen mode; the mode lives in the route state
            searches_before = len(control_call(f"{control}/counters")["search"])
            page.select_option("#ai-search-mode", "sound")
            page.fill("#ai-search-query", "amber")
            page.click("form.search-form button[type='submit']")
            page.wait_for_selector("[data-ai-search-status][data-ai-state]", timeout=10000)
            page.wait_for_function("document.querySelector('[data-ai-search-status]').getAttribute('data-ai-state') !== 'loading'", timeout=10000)
            page.wait_for_timeout(300)
            sound_requests = control_call(f"{control}/counters")["search"][searches_before:]
            checks.check("search_mode_kept_in_route_state", sound_requests == [{"q": "amber", "mode": "sound"}] and "mode=sound" in page.evaluate("location.hash")
                         and page.evaluate("document.querySelector('#ai-search-mode').value") == "sound", [sound_requests, page.evaluate("location.hash")])
            page.evaluate("location.hash = '#/similar?id=1'")
            page.wait_for_selector("[data-ai-similar-grid] .card", timeout=15000)
            checks.check("similar_results", len(page.query_selector_all("[data-ai-similar-grid] .card")) >= 1)
            # 8. Engine: honest capture status, a labelled synthetic promotion, reversible through the real control
            checks.check("synthetic_promotion_applied", control_call(f"{control}/promote", "POST")["ok"] is True)
            page.evaluate("location.hash = '#/engine'")
            page.wait_for_selector("[data-ai-ledger]", timeout=15000)
            capture = page.get_attribute("[data-ai-capture-status]", "data-ai-capture-status")
            checks.check("engine_capture_status_honest", capture == "unavailable" and "not proof" in page.inner_text("[data-ai-capture-status]"), capture)
            checks.check("engine_promotion_visible_and_labelled", page.get_attribute("[data-ai-promotion]", "data-ai-promotion") == "automatic"
                         and page.query_selector("[data-ai-ledger-row][data-ai-ledger-status='applied']") is not None and "demonstration evidence" in page.inner_text("[data-ai-ledger]"))
            applied_id = page.get_attribute("[data-ai-ledger-row][data-ai-ledger-status='applied']", "data-ai-ledger-row")
            page.click(f".aid-revert[data-id='{applied_id}']")
            page.wait_for_selector(f"[data-ai-ledger-row='{applied_id}'][data-ai-ledger-status='reverted']", timeout=10000)
            ledger = control_call(f"{control}/ledger")
            checks.check("engine_revert_confirmed_by_server", any(str(row["id"]) == applied_id and row["status"] == "reverted" for row in ledger) and page.query_selector("[data-ai-tuner-error]") is None)
            shot(page, "engine")
            # 9. Denied delivery is a configuration-required state, not a dead end
            page.evaluate("sessionStorage.setItem('feedloop-key', 'wrong')")
            page.evaluate("location.hash = '#/feed'")
            page.wait_for_selector("[data-ai-feed] [data-ai-state='configuration-required']", timeout=15000)
            checks.check("denied_delivery_is_configuration_required", "Configuration required" in page.text_content("[data-ai-feed] [data-ai-state-message]")
                         and page.query_selector("[data-ai-feed] [data-ai-retry]") is not None and page.query_selector("[data-idx]") is None)
            views = ["feed", "home", "search", "similar", "engine"]
            checks.check("five_views_reachable", all(page.query_selector(f"a[data-view='{v}']") for v in views))
            errors = [c for c in console if c["type"] in ("error", "pageerror") and "Failed to load resource" not in c["text"]]
            checks.check("no_console_errors", not errors, errors[:5])
            browser.close()
    except Exception:
        checks.failures.append("exception")
        report["exception"] = traceback.format_exc().splitlines()[-4:]
    finally:
        try:
            process.stdin.close()
            process.wait(timeout=15)
        except Exception:
            process.kill()
    report["status"] = "PASS" if not checks.failures else "FAIL"
    (output / "client-report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"status": report["status"], "failures": report["failures"], "checks": len(report["checks"]), "report": str(output / "client-report.json")}))
    return 0 if report["status"] == "PASS" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=None, help="project interpreter with feedloop installed; runs the harness")
    parser.add_argument("--output", required=True, help="directory for the report, screenshots and harness state")
    parser.add_argument("--harness", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.harness:
        return harness(Path(args.output))
    if not args.python:
        parser.error("--python is required to run the browser contract")
    return runner(args.python, Path(args.output))


if __name__ == "__main__":
    sys.exit(main())

"""Fixtures for the FL-3 tests: a synthetic media folder with neutral names and labelled
synthetic sidecars, a fake clock, and an HTTP client over the transport-free app."""
from __future__ import annotations

import json
import os
from pathlib import Path
import types

from feedloop.server.app import DEMO_ATTRIBUTION, FeedloopApp, build_engine, origins_for

WORDS = ["amber", "cobalt", "granite", "linen", "moss", "slate", "umber", "willow", "cedar", "pewter"]


def make_folder(root: Path, *, videos=6, images=3, sidecars=True, segments=False, seed=0) -> Path:
    media = root / "media"
    media.mkdir(parents=True, exist_ok=True)
    for i in range(videos):
        (media / f"sample-{i:02d}.mp4").write_bytes(bytes([(i * 7 + j) % 251 for j in range(2048 + i)]))
        if sidecars:
            tags = [WORDS[(i + k * 3) % len(WORDS)] for k in range(3)]
            payload = {"title": f"Sample {i}", "tags": tags, "contributors": [f"maker-{i % 2}"], "duration_s": 600.0}
            if segments:
                payload["segments"] = [{"start_s": 30.0 * k, "tags": [tag]} for k, tag in enumerate(tags)]
            (media / f"sample-{i:02d}.mp4.json").write_text(json.dumps(payload))
    for i in range(images):
        (media / f"still-{i:02d}.jpg").write_bytes(bytes([(i * 11 + j) % 241 for j in range(1024 + i)]))
        if sidecars:
            (media / f"still-{i:02d}.jpg.json").write_text(json.dumps({"tags": [WORDS[i], WORDS[i + 1]]}))
    return media


class Clock:
    def __init__(self, start=1_700_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


def fake_time(clock):
    import time as real
    return types.SimpleNamespace(time=clock, monotonic=real.monotonic, perf_counter=real.perf_counter, sleep=real.sleep)


def make_app(root: Path, monkeypatch, *, clock=None, api_key="test-key", port=8765, attribution=DEMO_ATTRIBUTION, config=None, **kwargs):
    from feedloop import engine as engine_module, ledger, serving, tuning
    clock = clock or Clock()
    for module in (ledger, tuning, engine_module, serving):
        monkeypatch.setattr(module, "time", fake_time(clock))
    media = make_folder(root, **kwargs)
    source, engine = build_engine(media, root / "state", initialize=True, attribution=attribution, clock=clock,
                                  config=config or {"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0})
    app = FeedloopApp(engine, api_key=api_key, allowed_origins=origins_for("127.0.0.1", port), source=source, clock=clock)
    return types.SimpleNamespace(app=app, engine=engine, source=source, clock=clock, media=media, key=api_key, origin=f"http://127.0.0.1:{port}")


def request(ctx, method, target, body=None, *, auth=True, origin=None, headers=None):
    hdrs = dict(headers or {})
    if auth:
        hdrs.update({"x-ai-api-key": ctx.key, "origin": origin or ctx.origin})
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    status, response_headers, payload = ctx.app.handle(method, target, hdrs, b"" if data is None else data)
    try:
        decoded = json.loads(payload) if payload else None
    except ValueError:
        decoded = payload
    return status, decoded


def feed_request(**overrides):
    base = {"limit": 8, "images": True, "surface": "feed", "session_id": "session-1", "request_id": "req-1", "client_request_id": "client-1"}
    base.update(overrides)
    return base

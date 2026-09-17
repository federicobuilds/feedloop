"""``feedloop demo ./folder``, ``feedloop serve``, ``feedloop extract``.

``demo`` explicitly creates new local stores under the state directory, builds the
labelled sidecar-text space, and serves the folder on loopback with the five second
demonstration attribution window. ``serve`` reuses existing stores with the
production policy and never creates them unless ``--init`` is passed. ``extract``
runs an optional learned extractor and needs the ``[extract]`` extra.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time

from feedloop.server.app import DEMO_ATTRIBUTION, FeedloopApp, build_engine, new_api_key, origins_for, run_server


FIXTURE_TAGS = ["amber", "cobalt", "granite", "linen", "moss", "slate", "umber", "willow"]
FIXTURE_NOTE = "synthetic demonstration fixture; not derived from or describing the media"
MEDIA_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov", ".m4v", ".jpg", ".jpeg", ".png", ".gif", ".webp")


def write_fixture_sidecars(folder) -> int:
    """Labelled synthetic sidecars for a demonstration folder: three tags per file and, for
    videos, three timed segments (starts 0, 30, 60 s) each carrying one tag. Every value is
    a fixture and says so in the sidecar's ``note``; nothing is read from the media.
    Existing sidecars are never touched, and a media name the filesystem refuses to extend
    with ``.json`` is skipped (the source records that item's reason). Returns the count
    written."""
    import json
    import random
    from feedloop.sources.filesystem import sidecar_path_reason
    rng = random.Random(7)
    written = 0
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in MEDIA_SUFFIXES or path.name.startswith("."):
            continue
        sidecar = Path(str(path) + ".json")
        if sidecar_path_reason(sidecar) or sidecar.exists():
            continue
        tags = sorted(rng.sample(FIXTURE_TAGS, 3))
        payload = {"title": path.stem, "tags": tags, "note": FIXTURE_NOTE}
        if path.suffix.lower() in MEDIA_SUFFIXES[:5]:
            payload["segments"] = [{"start_s": 30.0 * n, "tags": [tag], "note": FIXTURE_NOTE} for n, tag in enumerate(tags)]
        sidecar.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        written += 1
    return written


def _common(parser: argparse.ArgumentParser, *, state_default):
    parser.add_argument("folder", help="media folder (videos and images, optional <file>.json sidecars)")
    parser.add_argument("--state", default=state_default, help="state directory for the source, ledger and tuner stores"
                        + (" (default: <folder>/.feedloop)" if state_default is None else ""))


def _serve_args(parser):
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="port, 0 picks a free one (default 8765)")
    parser.add_argument("--api-key", default=None, help="shared key for mutations; default: FEEDLOOP_API_KEY or a fresh random key")
    parser.add_argument("--tick-interval", type=float, default=None, help="seconds between attribution/tuner ticks (0 disables)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="feedloop", description="A learning recommendation feed over a media folder.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="initialize new local stores for a folder and serve the client with the demo attribution window")
    _common(demo, state_default=None)
    _serve_args(demo)
    demo.add_argument("--fixture-sidecars", action="store_true", help="write labelled synthetic sidecars (tags and timed segments) for files that have none")
    demo.set_defaults(run=run_demo)
    serve = commands.add_parser("serve", help="serve an existing state directory with the production attribution policy")
    _common(serve, state_default=None)
    _serve_args(serve)
    serve.add_argument("--init", action="store_true", help="explicitly create the stores when they do not exist yet")
    serve.set_defaults(run=run_serve)
    extract = commands.add_parser("extract", help="compute learned features with an optional extractor (needs pip install 'feedloop[extract]')")
    _common(extract, state_default=None)
    extract.add_argument("--kind", choices=("visual", "audio"), required=True, help="visual (CLIP/SigLIP) or audio (CLAP)")
    extract.add_argument("--model", required=True, help="model identifier passed to the extractor family")
    extract.add_argument("--space", default=None, help="feature space name to write (default: visual or audioembed)")
    extract.add_argument("--vocabulary", default=None, help="visual only: text file with one zero-shot tag per line")
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--overwrite", action="store_true", help="replace an existing space and the generated fields of existing <file>.generated.json files; user fields are always kept")
    extract.set_defaults(run=run_extract)
    return parser


def _state_dir(args) -> Path:
    return Path(args.state) if args.state else Path(args.folder) / ".feedloop"


def _serve(args, *, initialize: bool, attribution):
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"folder not found: {folder}", file=sys.stderr)
        return 2
    state = _state_dir(args)
    if not initialize and not (state / "ledger.sqlite").exists():
        print("no stores under the state directory; run `feedloop demo` or `feedloop serve --init`", file=sys.stderr)
        return 2
    api_key = args.api_key or os.environ.get("FEEDLOOP_API_KEY") or new_api_key()
    source, engine = build_engine(folder, state, initialize=initialize, attribution=attribution)
    app = FeedloopApp(engine, api_key=api_key, allowed_origins=origins_for(args.host, args.port or 0), source=source)
    server = run_server(app, args.host, args.port)
    port = server.server_address[1]
    app.allowed_origins = origins_for(args.host, port)
    interval = args.tick_interval if args.tick_interval is not None else (15.0 if initialize else 300.0)
    stop = threading.Event()

    def ticker():
        while not stop.wait(interval):
            try:
                engine.tick()
            except Exception:
                pass
    if interval > 0:
        threading.Thread(target=ticker, name="feedloop-tick", daemon=True).start()
    print(f"feedloop serving {folder} on http://{args.host}:{port}/")
    print(f"open http://{args.host}:{port}/#key={api_key}")
    print(f"attribution window {attribution['window_s'] if attribution else 'production'} s; tick every {interval} s; state in {state}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
    return 0


def run_demo(args):
    if getattr(args, "fixture_sidecars", False) and Path(args.folder).is_dir():
        print(f"fixture sidecars written: {write_fixture_sidecars(args.folder)}")
    return _serve(args, initialize=True, attribution=DEMO_ATTRIBUTION)


def run_serve(args):
    return _serve(args, initialize=args.init, attribution=None)


def run_extract(args):
    from feedloop.extractors import MissingExtra
    from feedloop.sources.filesystem import FilesystemSource
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"folder not found: {folder}", file=sys.stderr)
        return 2
    source = FilesystemSource(folder, _state_dir(args))
    try:
        if args.kind == "visual":
            from feedloop.extractors.visual import VisualExtractor
            vocabulary = Path(args.vocabulary).read_text(encoding="utf-8").splitlines() if args.vocabulary else None
            extractor = VisualExtractor(args.model, device=args.device, vocabulary=vocabulary)
            report = extractor.extract_folder(source, space=args.space or "visual", overwrite=args.overwrite)
        else:
            from feedloop.extractors.audio import AudioExtractor
            extractor = AudioExtractor(args.model, device=args.device)
            report = extractor.extract_folder(source, space=args.space or "audioembed", overwrite=args.overwrite)
    except MissingExtra as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(report)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())

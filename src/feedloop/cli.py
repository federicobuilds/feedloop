"""``feedloop demo ./folder``, ``feedloop serve``, ``feedloop extract``.

``demo`` explicitly creates new local stores under the state directory, builds the
labelled sidecar-text space, and serves the folder on loopback with the five second
demonstration attribution window. ``serve`` reuses existing stores with the
production policy and never creates them unless ``--init`` is passed. ``extract``
runs an optional learned extractor and needs the ``[extract]`` extra.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import sqlite3
import threading
import time

from feedloop import ledger
from feedloop.server.app import DEMO_ATTRIBUTION, WILDCARD_HOSTS, FeedloopApp, build_engine, lan_addresses, new_api_key, origins_for, run_server


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


def _port(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port: {text!r}") from None
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 0..65535, got {value}")
    return value


def _interval(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid interval: {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"interval must be a finite number >= 0, got {text}")
    return value


def _serve_args(parser):
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; 0.0.0.0 also allows this machine's LAN addresses)")
    parser.add_argument("--port", type=_port, default=8765, help="port, 0 picks a free one (default 8765)")
    parser.add_argument("--api-key", default=None, help="shared key for mutations; default: FEEDLOOP_API_KEY or a fresh random key")
    parser.add_argument("--tick-interval", type=_interval, default=None, help="seconds between attribution/tuner ticks (0 disables)")


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
    serve.add_argument("--space-role", action="append", default=[], metavar="ROLE=SPACE", help="select a feature space for a recommendation role")
    serve.add_argument("--init", action="store_true", help="explicitly create the stores when they do not exist yet")
    serve.set_defaults(run=run_serve)
    extract = commands.add_parser("extract", help="compute learned features with an optional extractor (needs pip install 'feedloop[extract]')")
    _common(extract, state_default=None)
    extract.add_argument("--kind", choices=("visual", "audio"), required=True, help="visual (CLIP/SigLIP) or audio (CLAP)")
    extract.add_argument("--model", required=True, help="model identifier passed to the extractor family")
    extract.add_argument("--space", default=None, help="feature space name to write (default: visual or audioembed)")
    extract.add_argument("--pretrained", default=None, help="visual only: open_clip weights tag (default laion2b_s34b_b79k for ViT-B-32, else open_clip's first listed tag)")
    extract.add_argument("--vocabulary", default=None, help="visual only: text file with one zero-shot tag per line")
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--overwrite", action="store_true", help="replace an existing space and the generated fields of existing <file>.generated.json files; user fields are always kept")
    extract.set_defaults(run=run_extract)
    ingest = commands.add_parser("ingest", help="incrementally publish features from a folder to the feed")
    _common(ingest, state_default=None)
    ingest.add_argument("--kind", choices=("visual", "audio"), default="visual")
    ingest.add_argument("--model", required=True)
    ingest.add_argument("--pretrained", default=None, help="visual only: OpenCLIP weights tag")
    ingest.add_argument("--model-revision", default=None, help="weights revision for a custom or updated checkpoint")
    ingest.add_argument("--space", default=None, help="space to write (default: visual or audioembed)")
    ingest.add_argument("--device", default="cpu")
    ingest.add_argument("--overwrite", action="store_true", help="rebuild the space using the chosen model")
    ingest.set_defaults(run=run_ingest)
    pipeline = commands.add_parser("pipeline", help="run an incremental multi-stage model pipeline")
    _common(pipeline, state_default=None)
    pipeline.add_argument("--stages", type=Path, required=True, help="JSON stage manifest")
    pipeline.add_argument("--provider", choices=("local", "runpod"), default="local")
    pipeline.add_argument("--allow-remote", action="store_true", help="authorize sending selected media to your Pod")
    pipeline.add_argument("--image", help="worker image with FeedLoop and the manifest's adapters installed")
    pipeline.add_argument("--gpu", help="Runpod GPU catalog ID")
    pipeline.add_argument("--ssh-key", type=Path, help="operator's Ed25519 SSH private key; .pub must also exist")
    pipeline.add_argument("--registry", help="optional Runpod private registry credential ID")
    pipeline.add_argument("--max-hourly-rate", type=float, help="maximum GPU compute rate in USD per hour")
    pipeline.add_argument("--min-cuda", default="12.8")
    pipeline.add_argument("--disk", type=int, default=30, help="ephemeral Pod disk size in GB")
    pipeline.set_defaults(run=run_pipeline_command)
    cleanup = commands.add_parser("pipeline-cleanup", help="reconcile a tracked Runpod allocation from an interrupted pipeline")
    cleanup.add_argument("--state", type=Path, required=True, help="state directory passed to the pipeline")
    cleanup.set_defaults(run=run_pipeline_cleanup)
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
    try:
        from feedloop.slots import DEFAULT_SPACE_ROLES

        roles = {}
        for mapping in getattr(args, "space_role", []):
            role, _, space = mapping.partition("=")
            if role not in DEFAULT_SPACE_ROLES or not space or role in roles:
                raise ValueError(f"invalid or duplicate space role: {mapping}")
            roles[role] = space
        source, engine = build_engine(folder, state, initialize=initialize, attribution=attribution,
                                      space_roles=roles, demo_space=attribution is not None)
        app = FeedloopApp(engine, api_key=api_key, allowed_origins=origins_for(args.host, args.port or 0), source=source)
        server = run_server(app, args.host, args.port)
    except (OSError, ValueError, sqlite3.Error, ledger.ContractError) as exc:
        print(f"feedloop: error: {exc}", file=sys.stderr)
        return 2
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
    shown = "127.0.0.1" if args.host in WILDCARD_HOSTS else args.host
    print(f"feedloop serving {folder} on http://{shown}:{port}/")
    print(f"open http://{shown}:{port}/#key={api_key}")
    if args.host in WILDCARD_HOSTS:
        for address in lan_addresses():
            print(f"on your network: http://{address}:{port}/#key={api_key}")
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
    from feedloop.extractors import MissingExtra, MissingWeights
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
            extractor = VisualExtractor(args.model, device=args.device, pretrained=args.pretrained, vocabulary=vocabulary)
            report = extractor.extract_folder(source, space=args.space or "visual", overwrite=args.overwrite)
        else:
            from feedloop.extractors.audio import AudioExtractor
            extractor = AudioExtractor(args.model, device=args.device)
            report = extractor.extract_folder(source, space=args.space or "audioembed", overwrite=args.overwrite)
    except (MissingExtra, MissingWeights) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(report)
    return 0


def run_ingest(args):
    import json
    from feedloop.extractors import MissingExtra, MissingWeights
    from feedloop.ingest import ingest_space
    from feedloop.sources.filesystem import FilesystemSource

    try:
        source = FilesystemSource(args.folder, _state_dir(args))
        recipe = {"kind": args.kind, "model": args.model, "model_revision": args.model_revision,
                  "pretrained": args.pretrained, "version": 1}
        if args.kind == "visual":
            from feedloop.extractors.visual import VisualExtractor

            extractor = VisualExtractor(args.model, device=args.device, pretrained=args.pretrained)
            kinds, space = source.kinds, args.space or "visual"
        else:
            from feedloop.extractors.audio import AudioExtractor

            if args.pretrained is not None:
                raise ValueError("--pretrained applies only to visual models")
            extractor = AudioExtractor(args.model, device=args.device)
            kinds, space = source.kinds[:1], args.space or "audioembed"
        report = ingest_space(source, extractor, space=space, recipe=recipe, kinds=kinds, replace=args.overwrite)
    except (OSError, ValueError, sqlite3.Error, MissingExtra, MissingWeights) as exc:
        print(f"feedloop: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 1 if report["skipped"] else 0



def run_pipeline_command(args):
    import json
    import signal

    from feedloop.folder_pipeline import load_stages, run_folder
    from feedloop.remote_pipeline import runpod_compute
    from feedloop.runpod import hourly_rate
    from feedloop.sources.filesystem import FilesystemSource

    compute = None
    try:
        if args.provider == "runpod":
            if not args.allow_remote or not args.image or not args.gpu or not args.ssh_key or args.max_hourly_rate is None:
                raise ValueError("Runpod needs --allow-remote, --image, --gpu, --ssh-key, and --max-hourly-rate")
            hourly_rate(args.max_hourly_rate)
            if args.disk <= 0 or not args.ssh_key.is_file() or not args.ssh_key.with_name(args.ssh_key.name + ".pub").is_file():
                raise ValueError("Runpod needs a positive disk size and an existing SSH key pair")
            compute = runpod_compute(_state_dir(args), image=args.image, gpu=args.gpu, ssh_key=args.ssh_key,
                                     max_hourly_rate=args.max_hourly_rate, disk=args.disk, min_cuda=args.min_cuda,
                                     registry=args.registry)
        stages = load_stages(args.stages)
        source = FilesystemSource(args.folder, _state_dir(args))
        def interrupt(_number, _frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, interrupt) if threading.current_thread() is threading.main_thread() else None
        try:
            report = run_folder(source, stages, _state_dir(args), compute=compute)
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
    except KeyboardInterrupt:
        print("feedloop: pipeline stopped", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        print(f"feedloop: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 1 if report["failed"] or any(space["skipped"] for space in report["publication"].values()) else 0


def run_pipeline_cleanup(args):
    from feedloop.remote_pipeline import cleanup_runpod

    try:
        cleanup_runpod(args.state)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"feedloop: error: {exc}", file=sys.stderr)
        return 2
    print("FeedLoop Runpod cleanup confirmed")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())

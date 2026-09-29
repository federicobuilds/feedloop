"""CLI: help, demo initializes explicit stores and serves, serve refuses missing stores, extract reports the missing extra."""
import json
import threading
import urllib.request

import pytest

from feedloop import cli
from feedloop.server import app as app_module
from fl3_helpers import make_folder


def test_help_lists_the_three_commands(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert all(word in out for word in ("demo", "serve", "extract"))


def test_serve_refuses_a_missing_state_and_missing_folder(tmp_path, capsys):
    media = make_folder(tmp_path)
    assert cli.main(["serve", str(media), "--state", str(tmp_path / "state")]) == 2
    assert "feedloop demo" in capsys.readouterr().err
    assert cli.main(["demo", str(tmp_path / "missing")]) == 2


def test_extract_reports_the_missing_extra(tmp_path, capsys, monkeypatch):
    media = make_folder(tmp_path)
    import feedloop.extractors as extractors
    real_import = extractors.importlib.import_module
    monkeypatch.setattr(extractors.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name)) if name in ("torch", "open_clip", "transformers") else real_import(name))
    assert cli.main(["extract", str(media), "--kind", "visual", "--model", "ViT-B-32", "--state", str(tmp_path / "state")]) == 2
    err = capsys.readouterr().err
    assert "feedloop[extract]" in err and "torch" in err


def test_demo_serves_and_prints_the_keyed_address(tmp_path, monkeypatch, capsys):
    media = make_folder(tmp_path)
    started = threading.Event()
    captured = {}
    real_run_server = app_module.run_server

    def run_server(app, host, port):
        server = real_run_server(app, host, 0)
        captured["server"], captured["app"] = server, app
        original = server.serve_forever

        def serve_forever():
            started.set()
            original()
        server.serve_forever = serve_forever
        return server
    monkeypatch.setattr(cli, "run_server", run_server)
    thread = threading.Thread(target=lambda: captured.setdefault("exit", cli.main(["demo", str(media), "--state", str(tmp_path / "state"), "--port", "0", "--api-key", "abc", "--tick-interval", "0"])), daemon=True)
    thread.start()
    assert started.wait(10)
    server = captured["server"]
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/config", timeout=5) as response:
            config = json.loads(response.read())
        assert config["attribution"]["policy_revision"].startswith("demo") and "sidecar_text" in config["spaces"]
        assert (tmp_path / "state" / "ledger.sqlite").exists() and (tmp_path / "state" / "tuner.sqlite").exists()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/api/tick", method="POST", data=b"", headers={"x-ai-api-key": "abc", "Origin": f"http://127.0.0.1:{port}"})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert "attributed" in json.loads(response.read())
    finally:
        server.shutdown()
    thread.join(10)
    out = capsys.readouterr().out
    assert f"http://127.0.0.1:{port}/#key=abc" in out and "attribution window 5.0 s" in out
    # serve reuses the stores without re-initializing and refuses production stores it cannot find
    assert cli.main(["serve", str(media), "--state", str(tmp_path / "other")]) == 2


def test_port_and_interval_are_validated(capsys):
    for argv in (["--port", "70000"], ["--port", "-1"], ["--port", "http"], ["--tick-interval", "nan"], ["--tick-interval", "-1"], ["--tick-interval", "inf"]):
        with pytest.raises(SystemExit) as exit_info:
            cli.build_parser().parse_args(["demo", "folder", *argv])
        assert exit_info.value.code == 2, argv
    assert cli.build_parser().parse_args(["demo", "folder", "--port", "0", "--tick-interval", "0"]).port == 0


def test_busy_port_is_one_error_line(tmp_path, capsys):
    import socket
    media = make_folder(tmp_path)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert cli.main(["demo", str(media), "--state", str(tmp_path / "state"), "--port", str(port), "--tick-interval", "0"]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("feedloop: error: ")


def test_wildcard_host_prints_loopback_and_allows_lan_origins(monkeypatch):
    monkeypatch.setattr(app_module, "lan_addresses", lambda: ["192.168.1.20"])
    assert app_module.origins_for("0.0.0.0", 80) == ("http://127.0.0.1:80", "http://192.168.1.20:80", "http://localhost:80")
    assert app_module.origins_for("127.0.0.1", 80) == ("http://127.0.0.1:80", "http://localhost:80")

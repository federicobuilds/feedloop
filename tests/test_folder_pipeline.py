import json
from pathlib import Path
import subprocess
import sys

from feedloop.folder_pipeline import run_folder
from feedloop.multistage import Stage
from feedloop.remote_pipeline import runpod_compute
from feedloop.sources.filesystem import FilesystemSource


def fixture_model(path, recipe):
    return {"embedding": [float(path.stat().st_size), float(recipe["value"])],
            "labels": ["fixture"], "detections": []}


def pipeline(tmp_path):
    folder = tmp_path / "media"
    folder.mkdir()
    (folder / "first.png").write_bytes(b"fixture")
    state = tmp_path / "state"
    source = FilesystemSource(folder, state)
    adapter = __name__ + ":fixture_model"
    stages = [Stage("visual", adapter, {"value": 1, "kind": "visual"}, space="visual"),
              Stage("labels", adapter, {"value": 2})]
    return source, stages, state


def test_folder_pipeline_publishes_and_reuses(tmp_path):
    source, stages, state = pipeline(tmp_path)
    first = run_folder(source, stages, state)
    assert first["processed"] == 2 and not first["failed"]
    revision = source.revision("visual")
    assert revision and first["publication"]["visual"]["revision"] == revision
    second = run_folder(source, stages, state)
    assert second["processed"] == 0 and second["reused"] == 2
    assert source.revision("visual") == revision
    nested = source.folder / "nested"
    nested.mkdir()
    (nested / "second.png").write_bytes(b"new fixture")
    third = run_folder(source, stages, state)
    assert third["reused"] == third["processed"] == 2
    assert len(source.matrix("visual")[0]) == 2


def test_remote_pipeline_streams_work_and_confirms_cleanup(tmp_path, monkeypatch):
    source, stages, state = pipeline(tmp_path)
    monkeypatch.setenv("PYTHONPATH", f"{Path(__file__).parent}:{Path(__file__).parents[1] / 'src'}")
    from feedloop import remote_pipeline

    real_popen = subprocess.Popen
    launches = []

    def local_worker(argv, **kwargs):
        launches.append(argv)
        assert "-oStrictHostKeyChecking=yes" in argv
        return real_popen([sys.executable, "-u", "-m", "feedloop.remote_worker"], **kwargs)

    monkeypatch.setattr(remote_pipeline.subprocess, "Popen", local_worker)
    monkeypatch.setattr(remote_pipeline, "pin_host_key", lambda *a, **k: "[192.0.2.1]:2222 ssh-ed25519 fixture\n")
    key = tmp_path / "key"
    key.with_suffix(".pub").write_text("ssh-ed25519 fixture")
    calls = []
    pod = {}

    def api(method, path, body=None, *, deadline):
        calls.append((method, path))
        if path.startswith("/catalog/gpus?"):
            return 200, {"gpus": [{"id": "GPU", "availability": "HIGH", "price": {"secure": 0.4}}]}
        if (method, path) == ("POST", "/pods"):
            pod.update(id="pod-1", name=body["name"], image=body["image"], status="RUNNING", cost=0.4,
                       cudaVersion="12.8", ssh={"direct": {"host": "192.0.2.1", "port": 2222, "username": "root"}})
            return 201, {"id": "pod-1"}
        assert path == "/pods/pod-1"
        if method == "DELETE":
            pod.clear()
            return 204, {}
        return (200, pod) if pod else (404, {})

    def compute():
        return runpod_compute(state, image="test/worker:fixture", gpu="GPU", ssh_key=key, api=api,
                              max_hourly_rate=0.5, operation_timeout=20)

    first = run_folder(source, stages, state, compute=compute())
    assert first["processed"] == 2 and not first["failed"]
    assert first["publication"]["visual"]["items"] == 1
    assert [call for call in calls if call[0] in {"POST", "DELETE"}] == [("POST", "/pods"), ("DELETE", "/pods/pod-1")]
    assert calls[-1] == ("GET", "/pods/pod-1")
    assert len(launches) == 1
    assert json.loads((state / "runpod.json").read_text())["status"] == "gone"
    count = len(calls)
    assert run_folder(source, stages, state, compute=compute())["reused"] == 2
    assert len(calls) == count

import hashlib
import json

import pytest

from feedloop.multistage import Item, Job, ResultStore, Stage, StageFailure, execute_job, run_pipeline


def item_for(root, name, *, key="media:1", kind="image"):
    return Item(key, name, hashlib.sha256((root / name).read_bytes()).hexdigest(), kind)


def fixture_adapter(path, recipe):
    return {"labels": [path.read_text()], "detections": [{"label": recipe["label"], "score": 0.9}],
            "embedding": [1.0, 2.0]}


def test_durable_retry_and_independent_stage_revisions(tmp_path):
    media = tmp_path / "fixture.txt"
    media.write_text("sample")
    item = item_for(tmp_path, media.name)
    adapter = f"{__name__}:fixture_adapter"
    stages = [Stage("labels", adapter, {"label": "sample"}),
              Stage("vector", adapter, {"label": "feature"}, kinds=("image",), space="visual")]
    ledger = tmp_path / "state" / "pipeline.sqlite"
    failed_job = Job(item, stages[1])

    def fail_one(folder, job):
        if job == failed_job:
            raise StageFailure("fixture_failure")
        return execute_job(folder, job)

    store = ResultStore(ledger)
    first = run_pipeline(tmp_path, [item], stages, store, dispatch=fail_one)
    assert first["processed"] == 1
    assert first["failed"] == {failed_job.id: "fixture_failure"}
    assert store.progress(failed_job) == {"output": None, "error": "fixture_failure", "attempts": 1}

    store = ResultStore(ledger)
    second = run_pipeline(tmp_path, [item], stages, store)
    assert second == {"processed": 1, "reused": 1, "failed": {}, "stopped": False}
    assert store.progress(failed_job)["attempts"] == 2
    assert store.get(failed_job)["embedding"] == [1.0, 2.0]
    assert run_pipeline(tmp_path, [item], stages, store)["reused"] == 2

    stages[1] = Stage("vector", adapter, {"label": "feature", "checkpoint": "v2"}, space="visual")
    third = run_pipeline(tmp_path, [item], stages, store)
    assert (third["processed"], third["reused"]) == (1, 1)
    media.write_text("updated")
    changed = run_pipeline(tmp_path, [item_for(tmp_path, media.name)], stages, store)
    assert (changed["processed"], changed["reused"]) == (2, 0)


def test_wire_execution_verifies_media_and_rejects_external_path(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    path = root / "fixture.txt"
    path.write_text("sample")
    stage = Stage("label", f"{__name__}:fixture_adapter", {"label": "sample"})
    original = Job(item_for(root, path.name), stage)
    wire = Job.from_json(json.loads(json.dumps(original.to_json())))
    assert wire.id == original.id
    assert execute_job(root, wire)["labels"] == ["sample"]
    path.write_text("different bytes")
    with pytest.raises(StageFailure, match="input_changed"):
        execute_job(root, wire)
    outside = tmp_path / "external.txt"
    outside.write_text("sample")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(StageFailure, match="input_outside_folder"):
        execute_job(root, wire)
    with pytest.raises(ValueError, match="relative"):
        Item("1", "../external.txt", wire.item.fingerprint, "image")


def test_transport_failure_aborts_publication_without_discarding_output(tmp_path):
    path = tmp_path / "fixture.txt"
    path.write_text("sample")
    item = item_for(tmp_path, path.name)
    stages = [Stage(name, f"{__name__}:fixture_adapter", {"label": "sample"})
              for name in ("first", "second", "third")]
    store = ResultStore(tmp_path / "pipeline.sqlite")
    dispatched = []

    def interrupted(folder, job):
        dispatched.append(job.stage.name)
        if job.stage.name == "second":
            raise OSError("transport interrupted")
        return execute_job(folder, job)

    def publish(*_args):
        pytest.fail("interrupted dispatch must not publish")

    with pytest.raises(OSError, match="transport interrupted"):
        run_pipeline(tmp_path, [item], stages, store, dispatch=interrupted, publish=publish)
    assert dispatched == ["first", "second"]
    assert store.get(Job(item, stages[0]))["labels"] == ["sample"]
    assert store.progress(Job(item, stages[1]))["error"] == "OSError"
    assert store.progress(Job(item, stages[2])) is None

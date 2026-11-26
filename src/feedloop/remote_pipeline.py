"""Credentialed, disposable GPU execution for the folder pipeline."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryFile
import time
import uuid

from .multistage import StageFailure, checked_input
from .remote_worker import encode, size
from .runpod import (
    RunpodAPI, RunpodError, catalog_quote, checked_hourly_rate, compatible_cuda_version,
    create_pod, direct_endpoint, pin_host_key, reconcile_allocation,
)


@contextmanager
def _journal_lock(directory):
    try:
        import fcntl
    except ImportError:
        raise RunpodError("runpod_controller_requires_posix") from None
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runpod.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunpodError("runpod_pipeline_already_running") from None
        yield directory / "runpod.json"


def _record(path, name, status, pod_id, http_status):
    data = {"name": name, "status": status, "pod_id": pod_id, "http_status": http_status}
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            json.dump(data, out)
            out.flush()
            os.fsync(out.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _reconcile(api, journal):
    if not journal.is_file():
        return
    row = json.loads(journal.read_text(encoding="utf-8"))
    name = row["name"]
    if not re.fullmatch(r"feedloop-[0-9a-f]{32}", name):
        raise RunpodError("pod_allocation_name_invalid")
    if row["status"] == "gone":
        return
    outcome = reconcile_allocation(api, name=name, pod_id=row["pod_id"], deadline=time.time() + 120,
                                   record=lambda *args: _record(journal, name, *args))
    if not outcome["confirmed"] or outcome.get("journaled") is False:
        raise RunpodError("pod_cleanup_unconfirmed")


def cleanup_runpod(state):
    with _journal_lock(state) as journal:
        _reconcile(RunpodAPI(), journal)


def _endpoint(api, pod_id, name, image, rate, minimum, deadline):
    while time.time() < deadline:
        code, info = api("GET", f"/pods/{pod_id}", deadline=deadline)
        if code != 200 or not isinstance(info, dict):
            raise RunpodError("pod_status_unknown")
        if info.get("id") != pod_id or info.get("name") != name or info.get("image") != image:
            raise RunpodError("pod_runtime_identity_mismatch")
        if info.get("status") in ("ERROR", "EXITED", "TERMINATED"):
            raise RunpodError("pod_not_running")
        if info.get("status") == "RUNNING" and (endpoint := direct_endpoint(info)):
            checked_hourly_rate(info.get("cost"), rate)
            if not compatible_cuda_version(info.get("cudaVersion"), minimum):
                raise RunpodError("pod_cuda_incompatible")
            return endpoint
        time.sleep(min(5, max(0, deadline - time.time())))
    raise RunpodError("pod_no_endpoint")


def _wait_fd(fd, *, writing, deadline):
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise RunpodError("pipeline_worker_operation_timeout")
    ready = select.select([] if writing else [fd], [fd] if writing else [], [], timeout)
    if not (ready[0] or ready[1]):
        raise RunpodError("pipeline_worker_operation_timeout")


def _write(fd, data, deadline):
    view = memoryview(data)
    while view:
        _wait_fd(fd, writing=True, deadline=deadline)
        try:
            view = view[os.write(fd, view[:4096]):]
        except BlockingIOError:
            continue
        except BrokenPipeError:
            raise RunpodError("pipeline_worker_disconnected") from None


def _read(fd, count, deadline):
    data = bytearray()
    while len(data) < count:
        _wait_fd(fd, writing=False, deadline=deadline)
        block = os.read(fd, min(64 * 1024, count - len(data)))
        if not block:
            raise RunpodError("pipeline_worker_disconnected")
        data.extend(block)
    return data


def _dispatch(worker, folder, job, timeout):
    # Verify a stable local copy before transferring any bytes. The worker
    # independently checks the same fingerprint before invoking the adapter.
    source = checked_input(folder, job)
    with TemporaryFile() as snapshot, source.open("rb") as media:
        shutil.copyfileobj(media, snapshot)
        length = snapshot.tell()
        snapshot.seek(0)
        if hashlib.file_digest(snapshot, "sha256").hexdigest() != job.item.fingerprint:
            raise StageFailure("input_changed")
        snapshot.seek(0)
        deadline = time.monotonic() + timeout
        writer, reader = worker.stdin.fileno(), worker.stdout.fileno()
        _write(writer, encode({"job": job.to_json(), "bytes": length}), deadline)
        for block in iter(lambda: snapshot.read(64 * 1024), b""):
            _write(writer, block, deadline)
        try:
            response = json.loads(_read(reader, size(_read(reader, 4, deadline)), deadline))
            if not isinstance(response, dict) or response.get("job_id") != job.id:
                raise ValueError("wrong response")
            if set(response) == {"job_id", "error"} and isinstance(response["error"], str):
                raise StageFailure(response["error"])
            if set(response) != {"job_id", "output"} or not isinstance(response["output"], dict):
                raise ValueError("invalid output")
        except (ValueError, TypeError):
            raise RunpodError("pipeline_worker_response_invalid") from None
        return response["output"]


@contextmanager
def runpod_compute(state, *, image, gpu, ssh_key, max_hourly_rate, disk=30, cloud="SECURE",
                   min_cuda="12.8", registry=None, operation_timeout=3600, api=None):
    """Yield a dispatch function and allocate only if the pipeline needs it."""
    with _journal_lock(state) as journal:
        worker = None
        api = api or RunpodAPI()
        _reconcile(api, journal)
        try:
            def dispatch(folder, job):
                nonlocal worker
                if worker is None:
                    quote = catalog_quote(api, gpu=gpu, cloud=cloud, min_cuda=min_cuda, deadline=time.time() + 30)
                    if quote["available"] is not True:
                        raise RunpodError("configured_nvidia_gpu_has_no_capacity")
                    checked_hourly_rate(quote["hourly_rate"], max_hourly_rate)
                    # The key is used only by the local SSH client. Runpod receives the public half.
                    public = ssh_key.with_name(ssh_key.name + ".pub").read_text(encoding="utf-8").strip()
                    if not public.startswith("ssh-ed25519 ") or any(c in public for c in "\r\n"):
                        raise RunpodError("runpod_ed25519_key_required")
                    allocation = "feedloop-" + uuid.uuid4().hex
                    body = {"name": allocation, "image": image, "gpu": {"id": gpu, "count": 1, "minCudaVersion": min_cuda},
                            "cloud": cloud, "disk": disk, "ports": ["22/tcp"], "startSsh": True,
                            "env": {"PUBLIC_KEY": public}}
                    if registry:
                        body["registry"] = registry
                    deadline = time.time() + 600
                    pod = create_pod(api, body, deadline=deadline,
                                     record=lambda *args: _record(journal, allocation, *args))
                    host, port = _endpoint(api, pod, allocation, image, max_hourly_rate, min_cuda, deadline)
                    with NamedTemporaryFile(mode="w", dir=state, prefix="runpod-host-", delete=True) as trust:
                        trust.write(pin_host_key(api, pod, host=host, port=port, deadline=deadline))
                        trust.flush()
                        # Authenticate using the temporary pin before unlinking it.
                        ssh = ["ssh", "-F", "/dev/null", "-T", "-oBatchMode=yes", "-oIdentitiesOnly=yes",
                               "-oStrictHostKeyChecking=yes", "-oGlobalKnownHostsFile=/dev/null",
                               "-oConnectTimeout=15", "-oServerAliveInterval=15", "-oServerAliveCountMax=3",
                               "-oUserKnownHostsFile=" + trust.name, "-i", str(ssh_key), "-p", str(port),
                               "root@" + host]
                        # Use the same open pinned transport throughout the pipeline.
                        worker = subprocess.Popen(ssh + ["python -u -m feedloop.remote_worker"],
                                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                  stderr=subprocess.DEVNULL)
                        os.set_blocking(worker.stdin.fileno(), False)
                        # The pin must remain present when OpenSSH opens the file.
                        return _dispatch(worker, folder, job, operation_timeout)
                return _dispatch(worker, folder, job, operation_timeout)
            yield dispatch
        finally:
            try:
                if worker is not None:
                    try:
                        worker.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.communicate()
            finally:
                _reconcile(api, journal)

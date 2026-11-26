"""Shared ingest run sequencing and compute supervision."""
from __future__ import annotations

from contextlib import contextmanager
import time


class WorkflowError(RuntimeError):
    """A run could not advance safely."""


def execute_workflow(discover, process, *, publish, finalize=None, after_publish=None):
    """Run one durable attempt and gate host effects on verified publication.

    Discovery can resume from durable state. Processing may commit incrementally
    and must close compute before returning. Publication verifies those commits.
    Finalization returns a completion verdict and any application-specific detail.
    """
    work = discover()
    processed = process(work)
    publish(work)
    if finalize is None:
        return processed
    complete, report = finalize(work, processed)
    if complete and after_publish is not None:
        after_publish(work)
    return complete, report


@contextmanager
def compute_session(acquire, release):
    """Keep cleanup paired with a successful acquisition, even on cancellation."""
    worker = acquire()
    try:
        yield worker
    except BaseException as exc:
        try:
            confirmed = release(worker)
        except Exception:
            confirmed = False
        if not confirmed:
            exc.add_note("compute_cleanup_unconfirmed")
        raise
    else:
        if not release(worker):
            raise WorkflowError("compute_cleanup_unconfirmed")


def watch_compute(observe, *, restart, cancelled, check_owner, report, deadline,
                  stall_seconds, interval=5, clock=time.time, sleep=time.sleep):
    """Follow an externally managed worker until the lane exits successfully."""
    last_change, last_counts = clock(), None
    unreachable, unready = 0, 0
    while True:
        check_owner()
        if clock() >= deadline:
            raise WorkflowError("run_deadline")
        if cancelled():
            raise WorkflowError("stopped_by_user")
        sleep(min(interval, max(0, deadline - clock())))
        health, counts, status, exit_code = observe()
        report(health, counts, status)
        if counts != last_counts:
            last_change, last_counts = clock(), counts
        unreachable = 0 if health.get("reachable") else unreachable + 1
        unready = 0 if health.get("server_ok") else unready + 1
        if unreachable >= 3:
            raise WorkflowError("compute_unreachable")
        if unready >= 3:
            raise WorkflowError("compute_unready")
        if health.get("workers") == "0" and health.get("reachable"):
            restart()
        if status == "exited":
            if exit_code == 0:
                return
            raise WorkflowError("lane_failed")
        if clock() - last_change > stall_seconds():
            raise WorkflowError("compute_stalled")

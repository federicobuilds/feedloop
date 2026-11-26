"""Private SSH stream protocol for one disposable pipeline worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

from .multistage import Job, StageFailure, execute_job


MAX_MESSAGE = 8 * 1024 * 1024


def encode(message):
    data = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_MESSAGE:
        raise ValueError("pipeline worker message too large")
    return struct.pack("!I", len(data)) + data


def size(header):
    length, = struct.unpack("!I", header)
    if not 0 < length <= MAX_MESSAGE:
        raise ValueError("invalid pipeline worker message size")
    return length


def serve(source, sink):
    while header := source.read(4):
        request = json.loads(source.read(size(header)))
        job = Job.from_json(request["job"])
        remaining = request["bytes"]
        if type(remaining) is not int or remaining < 0:
            raise ValueError("invalid pipeline input size")
        with TemporaryDirectory(prefix="feedloop-worker-") as directory:
            root = Path(directory)
            path = root / job.item.path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                while remaining:
                    block = source.read(min(remaining, 64 * 1024))
                    if not block:
                        raise EOFError("truncated pipeline input")
                    output.write(block)
                    remaining -= len(block)
            try:
                response = {"output": execute_job(root, job)}
            except StageFailure as exc:
                response = {"error": str(exc)}
        sink.write(encode({"job_id": job.id, **response}))
        sink.flush()


def main():
    # Reserve a separate descriptor for protocol output. Model libraries may
    # write to stdout through native code as well as Python.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        serve(sys.stdin.buffer, protocol)


if __name__ == "__main__":
    main()

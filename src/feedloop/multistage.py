"""Resumable, model-independent extraction over local or remote workers."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3

from .ingest import processing_revision


_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}")
_ADAPTER = re.compile(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _relative(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or relative.as_posix() != path or any(c in path for c in "\\\x00\r\n")
            or any(re.match(r"[a-zA-Z]:", part) for part in relative.parts)):
        raise ValueError("media path must be relative to the selected folder")
    return relative


@dataclass(frozen=True)
class Item:
    key: str
    path: str
    fingerprint: str
    kind: str

    def __post_init__(self):
        _relative(self.path)
        if not self.key or not _SHA256.fullmatch(self.fingerprint) or not _NAME.fullmatch(self.kind):
            raise ValueError("invalid pipeline item or SHA-256 fingerprint")


@dataclass(frozen=True)
class Stage:
    name: str
    adapter: str
    recipe: dict
    kinds: tuple[str, ...] = ()
    space: str | None = None

    def __post_init__(self):
        if (not _NAME.fullmatch(self.name) or not _ADAPTER.fullmatch(self.adapter)
                or not isinstance(self.recipe, dict) or isinstance(self.kinds, str)
                or any(not _NAME.fullmatch(value) for value in self.kinds)
                or (self.space is not None and not _NAME.fullmatch(self.space))):
            raise ValueError("invalid pipeline stage")
        object.__setattr__(self, "kinds", tuple(self.kinds))
        object.__setattr__(self, "recipe", json.loads(_json(self.recipe)))

    @property
    def revision(self) -> str:
        """The recipe must identify all weights and settings that affect output."""
        return processing_revision({"adapter": self.adapter, "recipe": self.recipe})


@dataclass(frozen=True)
class Job:
    item: Item
    stage: Stage

    @property
    def id(self) -> str:
        return processing_revision({"item": asdict(self.item), "stage": self.stage.name,
                                    "revision": self.stage.revision})

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping) -> Job:
        return cls(item=Item(**data["item"]), stage=Stage(**data["stage"]))


class StageFailure(RuntimeError):
    """The adapter could not produce a result for one item."""


def _output(result: Mapping) -> str:
    if not isinstance(result, dict) or any(not isinstance(key, str) for key in result):
        raise ValueError("a stage must return a JSON object")
    if "embedding" in result:
        vector = result["embedding"]
        if not isinstance(vector, list) or not vector or any(type(v) not in (int, float) for v in vector):
            raise ValueError("embedding output must be a nonempty numeric list")
    return _json(result)


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _verified_input(folder: Path, job: Job) -> tuple[Path, tuple]:
    root = Path(folder).resolve()
    path = root.joinpath(*_relative(job.item.path).parts).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise StageFailure("input_outside_folder")
    before = _signature(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != job.item.fingerprint or before != _signature(path):
        raise StageFailure("input_changed")
    return path, before


def checked_input(folder: Path, job: Job) -> Path:
    """Return the verified absolute path for a dispatch implementation."""
    try:
        return _verified_input(folder, job)[0]
    except OSError as exc:
        raise StageFailure(type(exc).__name__) from None


def execute_job(folder: Path, job: Job) -> dict:
    """Execute an installed adapter against a verified copy of the input file."""
    try:
        path, before = _verified_input(folder, job)
    except OSError as exc:
        raise StageFailure(type(exc).__name__) from None
    try:
        module, name = job.stage.adapter.split(":", 1)
        adapter = getattr(import_module(module), name)
        result = _output(adapter(path, json.loads(_json(job.stage.recipe))))
        after = _signature(path)
    except Exception as exc:
        raise StageFailure(type(exc).__name__) from None
    if before != after:
        raise StageFailure("input_changed")
    return json.loads(result)


class ResultStore:
    """Persist successful results and retryable failures in a local SQLite file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS pipeline_results (
                job_id TEXT PRIMARY KEY, output TEXT, error TEXT, attempts INTEGER NOT NULL
            )""")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(self, job: Job) -> dict | None:
        row = self.progress(job)
        return row["output"] if row else None

    def progress(self, job: Job) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT output,error,attempts FROM pipeline_results WHERE job_id=?",
                               (job.id,)).fetchone()
        if row is None:
            return None
        output, error, attempts = row
        return {"output": json.loads(output) if output is not None else None,
                "error": error, "attempts": attempts}

    def record(self, job: Job, output: Mapping) -> None:
        with self._connect() as conn:
            conn.execute("""INSERT INTO pipeline_results VALUES (?, ?, NULL, 1)
                ON CONFLICT(job_id) DO UPDATE SET output=excluded.output, error=NULL,
                attempts=pipeline_results.attempts+1 WHERE pipeline_results.output IS NULL""",
                         (job.id, _output(output)))

    def fail(self, job: Job, code: str) -> None:
        with self._connect() as conn:
            conn.execute("""INSERT INTO pipeline_results VALUES (?, NULL, ?, 1)
                ON CONFLICT(job_id) DO UPDATE SET error=excluded.error,
                attempts=pipeline_results.attempts+1 WHERE pipeline_results.output IS NULL""",
                         (job.id, code))


def plan_jobs(items, stages):
    """Yield applicable item-stage pairs in the supplied order."""
    items, stages = tuple(items), tuple(stages)
    if len({item.key for item in items}) != len(items):
        raise ValueError("pipeline item keys must be unique")
    spaces = [stage.space for stage in stages if stage.space is not None]
    if not stages or len({stage.name for stage in stages}) != len(stages) or len(set(spaces)) != len(spaces):
        raise ValueError("provide distinct pipeline stage names and feature spaces")
    for item in items:
        for stage in stages:
            if not stage.kinds or item.kind in stage.kinds:
                yield Job(item, stage)


def run_pipeline(folder, items, stages, store, *, dispatch=execute_job, publish=None, should_stop=None) -> dict:
    """Run changed and failed work, preserving each success before advancing."""
    folder, items, stages = Path(folder), tuple(items), tuple(stages)
    report = {"processed": 0, "reused": 0, "failed": {}, "stopped": False}
    for job in plan_jobs(items, stages):
        if should_stop is not None and should_stop():
            report["stopped"] = True
            return report
        if store.get(job) is not None:
            report["reused"] += 1
            continue
        try:
            output = dispatch(folder, job)
            _output(output)
        except StageFailure as exc:
            code = str(exc) if _NAME.fullmatch(str(exc)) else "stage_failed"
            store.fail(job, code)
            report["failed"][job.id] = code
        except Exception as exc:
            store.fail(job, type(exc).__name__)
            raise
        else:
            store.record(job, output)
            report["processed"] += 1
    if should_stop is not None and should_stop():
        report["stopped"] = True
    elif publish is not None:
        report["publication"] = publish(items, stages, store)
    return report

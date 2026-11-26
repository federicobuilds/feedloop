"""Incremental feature extraction and revision publication."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol
import uuid

import numpy as np


ItemKey = tuple[str, int]
Rows = Sequence[tuple[ItemKey, Mapping]]


class FeatureExtractor(Protocol):
    def extract_rows(self, folder: Path, rows: Rows) -> tuple[Sequence[ItemKey], Sequence[np.ndarray], Mapping[str, str]]:
        """Return one vector per successful key, plus reasons for skipped inputs."""
        ...


def publish_revisions(committed: Mapping, visible: Callable[[], Mapping | None], *, refresh: Callable[[], object] | None = None) -> None:
    """Refresh the serving consumer and confirm it sees the committed revisions."""
    if refresh is not None and not refresh():
        raise PublicationError("refresh_unconfirmed")
    if visible() != committed:
        raise PublicationError("feature_revisions_not_visible")


class PublicationError(RuntimeError):
    """The serving consumer did not acknowledge the committed features."""


def processing_revision(recipe: Mapping) -> str:
    """Identify the weights and settings that determine a model's output."""
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()


def needs_processing(previous: tuple | None, incoming: tuple, *, invalidated=False) -> bool:
    """Version one work unit by both its input fingerprint and processing revision."""
    return invalidated or previous != incoming


def result_key(attempt_id: str, stage: str) -> str:
    """Give each stage result a stable ID across retries of a commit."""
    return str(uuid.uuid5(uuid.UUID(attempt_id), stage))


def ingest_space(source, extractor: FeatureExtractor, *, space: str, recipe: Mapping,
                 kinds: Sequence[str], replace=False) -> dict:
    """Process changed inputs and publish a homogeneous vector space atomically.

    The recipe identifies the model weights and settings that affect the output.
    Failed items are eligible for a retry. A changed item that fails extraction
    has no current vector. An exception before publication leaves the old space.
    """
    recipe_id = processing_revision(recipe)
    old_meta = source.space_meta(space)
    old_ingest = (old_meta or {}).get("ingest", {})
    matches = old_ingest.get("recipe") == recipe_id
    if space in source.spaces() and not matches and not replace:
        raise ValueError(f"space {space!r} has a different recipe; choose another space or pass --overwrite")
    rows = {key: entry for key, entry in sorted(source.refresh()["items"].items()) if key[0] in kinds}
    fingerprints = {f"{key[0]}:{key[1]}": entry["sha256"] for key, entry in rows.items()}
    previous = source.matrix(space) if matches and not replace else None
    old_fingerprints = old_ingest.get("files", {})
    retained = {}
    if previous is not None:
        for key, vector in zip(*previous):
            token = f"{key[0]}:{key[1]}"
            if token in fingerprints and not needs_processing((old_fingerprints.get(token), old_ingest["recipe"]),
                                                                (fingerprints[token], recipe_id)):
                retained[key] = vector
    pending = [(key, entry) for key, entry in rows.items() if key not in retained]
    updated_keys, updated_vectors, skipped = extractor.extract_rows(source.folder, pending) if pending else ([], [], {})
    if len(updated_keys) != len(updated_vectors) or len(set(updated_keys)) != len(updated_keys):
        raise ValueError("extractor returned invalid vector keys")
    if not set(updated_keys) <= {key for key, _ in pending}:
        raise ValueError("extractor returned an unexpected item")
    combined = {**retained, **dict(zip(updated_keys, updated_vectors))}
    keys = sorted(combined)
    successful = {f"{kind}:{item_id}": fingerprints[f"{kind}:{item_id}"] for kind, item_id in keys}
    manifest = {"recipe": recipe_id, "files": successful}
    revision = source.revision(space)
    if updated_keys or (matches and manifest != old_ingest) or (replace and old_meta is not None):
        dim = old_meta["dim"] if matches else 1
        matrix = np.stack([combined[key] for key in keys]) if keys else np.zeros((0, dim), dtype=np.float32)
        meta = {"provenance": str(recipe["model"]), "window_scope": "none", "kind": str(recipe["kind"]),
                "built_at": source.clock(), "ingest": manifest}
        revision = source.write_space(space, keys, matrix, meta=meta, expected_meta=(old_meta,))
        publish_revisions({space: revision}, lambda: {space: source.revision(space)})
    return {"space": space, "items": len(keys), "processed": len(updated_keys), "reused": len(retained),
            "skipped": dict(skipped), "revision": revision}

"""Filesystem adapters for the portable pipeline."""
from __future__ import annotations

from functools import lru_cache
import json

from .ingest import ingest_space, processing_revision
from .multistage import Item, Job, ResultStore, Stage, StageFailure, execute_job, run_pipeline
from .workflow import execute_workflow


@lru_cache(maxsize=4)
def _extractor(settings):
    recipe = json.loads(settings)
    if recipe["kind"] == "visual":
        from .extractors.visual import VisualExtractor

        return VisualExtractor(recipe["model"], device=recipe.get("device", "cpu"),
                               pretrained=recipe.get("pretrained"), vocabulary=recipe.get("vocabulary"))
    if recipe["kind"] == "audio":
        from .extractors.audio import AudioExtractor

        return AudioExtractor(recipe["model"], device=recipe.get("device", "cpu"))
    raise ValueError("embedding kind must be visual or audio")


def embed(path, recipe):
    """Built-in adapter for the same visual and audio families as feedloop ingest."""
    extractor = _extractor(json.dumps(recipe, sort_keys=True))
    from .sources.filesystem import _kind_of

    key = (_kind_of(path), 1)
    _keys, vectors, skipped = extractor.extract_rows(path.parent, [(key, {"relpath": path.name})])
    if skipped or not vectors:
        raise StageFailure("embedding_unavailable")
    output = {"embedding": vectors[0].tolist()}
    if recipe.get("vocabulary") and recipe["kind"] == "visual":
        output["labels"] = extractor.tags_for(vectors)[0]
    return output


class _RecordedEmbeddings:
    def __init__(self, stage, store):
        self.stage, self.store = stage, store

    def extract_rows(self, _folder, rows):
        keys, vectors, skipped = [], [], {}
        for key, entry in rows:
            kind, number = key
            token = f"{kind}:{number}"
            item = Item(token, entry["relpath"], entry["sha256"], kind)
            result = self.store.get(Job(item, self.stage))
            if result and "embedding" in result:
                keys.append(key)
                vectors.append(result["embedding"])
            else:
                skipped[token] = "embedding_unavailable"
        return keys, vectors, skipped


def load_stages(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("stage manifest must be a nonempty array")
    return tuple(Stage(**entry) for entry in value)


def run_folder(source, stages, state_dir, *, compute=None):
    """Reuse durable stage results and publish each configured embedding space."""
    stages = tuple(stages)
    store = ResultStore(state_dir / "pipeline.sqlite")
    report = {}

    def recipe(stage):
        return {"kind": stage.recipe.get("kind", "custom"),
                "model": stage.recipe.get("model", stage.adapter), "stage_revision": stage.revision}

    def discover():
        for stage in stages:
            if stage.space is None:
                continue
            existing = source.space_meta(stage.space)
            if existing is not None and existing.get("ingest", {}).get("recipe") != processing_revision(recipe(stage)):
                raise ValueError(f"space {stage.space!r} has a different recipe; choose another space")
        rows = source.refresh()["items"]
        return tuple(Item(f"{kind}:{key}", row["relpath"], row["sha256"], kind)
                     for (kind, key), row in sorted(rows.items()))

    def process(items):
        if compute is None:
            result = run_pipeline(source.folder, items, stages, store, dispatch=execute_job)
        else:
            with compute as dispatch:
                result = run_pipeline(source.folder, items, stages, store, dispatch=dispatch)
        report.update(result)

    def publish(_items):
        spaces = {}
        for stage in stages:
            if stage.space is None:
                continue
            spaces[stage.space] = ingest_space(source, _RecordedEmbeddings(stage, store), space=stage.space,
                                               recipe=recipe(stage), kinds=stage.kinds or source.kinds)
        report["publication"] = spaces

    execute_workflow(discover, process, publish=publish)
    return report

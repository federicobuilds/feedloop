"""Open CLIP / SigLIP image vectors and zero-shot tags from an editable vocabulary.

The extractor is explicit: construction stores names only; ``load()`` imports torch
and open_clip and downloads or opens the weights; ``extract_folder`` embeds the
folder's image files and writes one means-only feature space labelled with the
model name. A bare open_clip architecture name gets pretrained weights (see
``resolve_pretrained``) instead of random ones. Videos are embedded as the mean of
frames sampled by ffmpeg (from the ``imageio-ffmpeg`` wheel in the extra) unless the
caller supplies its own frame sampler. Zero-shot tags are the
softmax over cosine similarities to the vocabulary's prompts and are weaker than
a trained tagger; they are written to a separate generated file that never
touches the user's own sidecar.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from feedloop.extractors import MissingExtra, MissingWeights, require
from feedloop.sources.filesystem import GENERATED_SUFFIX, FilesystemSource, sidecar_path_reason

DEFAULT_PROMPT = "a photo of {}"
DEFAULT_PRETRAINED = {"ViT-B-32": "laion2b_s34b_b79k"}
FRAMES_PER_VIDEO = 8


def resolve_pretrained(model: str, pretrained: str | None, listed_tags: Sequence[str]) -> str | None:
    """The weights tag for ``model``: the explicit tag; none for a prefixed name such as
    ``hf-hub:org/repo``, which carries its own weights; ``laion2b_s34b_b79k`` for bare
    ``ViT-B-32``; otherwise the first tag open_clip lists for the architecture."""
    if pretrained or ":" in model:
        return pretrained
    if model in DEFAULT_PRETRAINED:
        return DEFAULT_PRETRAINED[model]
    if listed_tags:
        return listed_tags[0]
    raise MissingWeights(f"open_clip lists no pretrained weights for model '{model}'; pass --pretrained TAG or use an hf-hub: model name")


def ffmpeg_frame_sampler(path: Path, *, frames: int = FRAMES_PER_VIDEO) -> list:
    """``frames`` RGB images at evenly spaced timestamps; an unreadable file raises OSError."""
    ffmpeg = require("imageio_ffmpeg", purpose="the video frame sampler").get_ffmpeg_exe()
    image_module = require("PIL.Image", purpose="the video frame sampler")
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace")
    found = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
    if not found:
        raise OSError(f"ffmpeg found no duration in {path.name}")
    hours, minutes, seconds = found.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    images = []
    for index in range(frames):
        at = duration * (index + 0.5) / frames
        grab = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{at:.3f}", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                              capture_output=True)
        if grab.returncode == 0 and grab.stdout:
            with image_module.open(io.BytesIO(grab.stdout)) as image:
                images.append(image.convert("RGB"))
    return images


class OpenClipBackend:
    """The default runtime: open_clip model + tokenizer + preprocessing on one device."""

    def __init__(self, model: str, *, pretrained: str | None = None, device: str = "cpu"):
        self.model_name, self.pretrained, self.device = model, pretrained, device
        self.model = self.preprocess = self.tokenizer = None
        self.torch = None

    def load(self):
        self.torch = require("torch", purpose="the visual extractor")
        open_clip = require("open_clip", purpose="the visual extractor")
        require("PIL", purpose="the visual extractor")
        self.pretrained = resolve_pretrained(self.model_name, self.pretrained, open_clip.list_pretrained_tags_by_model(self.model_name))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        return self

    def embed_images(self, images) -> np.ndarray:
        with self.torch.no_grad():
            batch = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
            return self.model.encode_image(batch).float().cpu().numpy()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        with self.torch.no_grad():
            tokens = self.tokenizer(list(texts)).to(self.device)
            return self.model.encode_text(tokens).float().cpu().numpy()

    def open_image(self, path: Path):
        image_module = require("PIL.Image", purpose="the visual extractor")
        with image_module.open(path) as image:
            return image.convert("RGB")


def unit_rows(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def zero_shot_tags(image_vectors, vocabulary_vectors, vocabulary: Sequence[str], *, top_k=5, min_probability=0.05, scale=100.0):
    """Per image: the vocabulary entries whose softmax probability clears the floor."""
    images, words = unit_rows(image_vectors), unit_rows(vocabulary_vectors)
    logits = scale * images @ words.T
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    out = []
    for row in probabilities:
        order = np.argsort(-row)[:top_k]
        out.append([(vocabulary[i], float(row[i])) for i in order if row[i] >= min_probability])
    return out


class VisualExtractor:
    def __init__(self, model: str, *, device="cpu", pretrained=None, vocabulary: Sequence[str] | None = None, prompt=DEFAULT_PROMPT,
                 backend=None, frame_sampler: Callable[[Path], Sequence[Any]] | None = None):
        self.model = model
        self.backend = backend if backend is not None else OpenClipBackend(model, pretrained=pretrained, device=device)
        self.vocabulary = [w.strip() for w in (vocabulary or []) if w and w.strip()]
        self.prompt = prompt
        self.frame_sampler = frame_sampler if frame_sampler is not None else ffmpeg_frame_sampler
        self.loaded = False

    def load(self):
        if not self.loaded:
            self.backend.load()
            self.loaded = True
        return self

    def embed_paths(self, paths: Sequence[Path]) -> np.ndarray:
        self.load()
        return unit_rows(self.backend.embed_images([self.backend.open_image(Path(p)) for p in paths]))

    def tags_for(self, vectors) -> list[list[tuple[str, float]]]:
        if not self.vocabulary:
            return [[] for _ in range(len(vectors))]
        self.load()
        words = self.backend.embed_texts([self.prompt.format(word) for word in self.vocabulary])
        return zero_shot_tags(vectors, words, self.vocabulary)

    def extract_folder(self, source: FilesystemSource, *, space="visual", write_tags=True, overwrite=False, batch_size=16) -> dict:
        """Embed every image and every video the frame sampler can open, and write the space."""
        self.load()
        if space in source.spaces() and not overwrite:
            return {"space": space, "model": self.model, "items": 0, "skipped": {}, "revision": source.revision(space), "zero_shot_tags": {},
                    "generated_files_kept": [], "kept_existing_space": True}
        scan = source.refresh()
        keys, vectors, skipped, tagged = [], [], {}, {}
        rows = sorted(scan["items"].items())
        for start in range(0, len(rows), batch_size):
            batch, images = [], []
            for key, entry in rows[start:start + batch_size]:
                path = source.folder / entry["relpath"]
                if key[0] == "image":
                    images.append(self.backend.open_image(path))
                    batch.append(key)
                else:
                    try:
                        frames = list(self.frame_sampler(path))
                    except MissingExtra:
                        raise
                    except Exception as exc:
                        skipped[f"{key[0]}:{key[1]}"] = "undecodable:" + type(exc).__name__
                        continue
                    if not frames:
                        skipped[f"{key[0]}:{key[1]}"] = "no_frames"
                        continue
                    frame_vectors = unit_rows(self.backend.embed_images(frames))
                    keys.append(key)
                    vectors.append(unit_rows(frame_vectors.mean(axis=0, keepdims=True))[0])
            if images:
                embedded = unit_rows(self.backend.embed_images(images))
                keys.extend(batch)
                vectors.extend(embedded)
        matrix = np.stack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
        kept_existing, unwritable = [], {}
        if write_tags and self.vocabulary and keys:
            reasons = source.sidecar_reasons(keys)
            for key, tags in zip(keys, self.tags_for(matrix)):
                tagged[f"{key[0]}:{key[1]}"] = tags
                if reasons.get(key) in ("sidecar_path_too_long", "sidecar_unreadable"):
                    unwritable[f"{key[0]}:{key[1]}"] = reasons[key]
                elif write_generated_tags(source, key, tags, model=self.model, overwrite=overwrite) is None:
                    kept_existing.append(f"{key[0]}:{key[1]}")
        revision = None
        if keys:
            meta = {"provenance": f"open_clip:{self.model}", "window_scope": "none", "kind": "visual", "built_at": source.clock()}
            revision = source.write_space(space, keys, matrix, meta=meta)
        return {"space": space, "model": self.model, "items": len(keys), "skipped": skipped, "revision": revision, "zero_shot_tags": tagged,
                "generated_files_kept": kept_existing, "generated_files_unwritable": unwritable}


GENERATED_FIELDS = ("model", "method", "tags")


def write_generated_tags(source: FilesystemSource, key, tags: Sequence[tuple[str, float]], *, model: str, overwrite: bool = False):
    """Generated tags live in ``<file>.generated.json`` beside the media, separate from the
    user's sidecar. An existing file is left untouched unless ``overwrite`` is set, and even
    then only the generated fields change: every other key and every tag entry that is not
    ``zero_shot`` (a user's correction inside that file) is preserved. Returns the path, or
    None when an existing file was kept or the filesystem refuses the generated path (the
    source records that item's ``sidecar_reason``)."""
    relpath = source.refresh()["items"][key]["relpath"]
    path = Path(str(source.folder / relpath) + GENERATED_SUFFIX)
    if sidecar_path_reason(path):
        return None
    existing = {}
    if path.exists():
        if not overwrite:
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, dict) else {"_previous": loaded}
        except ValueError:
            existing = {"_previous": path.read_text(encoding="utf-8")}
    kept = [t for t in (existing.get("tags") or []) if not (isinstance(t, dict) and t.get("category") == "zero_shot")]
    generated = [{"name": name, "category": "zero_shot", "probability": round(p, 4)} for name, p in tags]
    payload = {k: v for k, v in existing.items() if k not in GENERATED_FIELDS}
    payload.update({"model": model, "method": "zero_shot", "tags": kept + generated})
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temporary, path)
    return path


__all__ = ["VisualExtractor", "OpenClipBackend", "resolve_pretrained", "ffmpeg_frame_sampler", "DEFAULT_PRETRAINED", "zero_shot_tags", "unit_rows", "write_generated_tags", "DEFAULT_PROMPT", "GENERATED_FIELDS"]

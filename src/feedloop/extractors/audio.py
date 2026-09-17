"""CLAP audio vectors that double as the text encoder for sound search.

``load()`` imports torch and transformers and opens the CLAP checkpoint. Audio is
decoded by a caller-supplied decoder (``path -> (waveform, sample_rate)``) or, by
default, ``soundfile``, which reads audio files but not the audio track of a video
container; a file the decoder cannot open is skipped and named in the report. The
written space holds one mean vector per item (no windows) labelled with the model name; ``encode`` returns
the matching text vector for exactly that space and ``None`` for any other.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from feedloop.extractors import require
from feedloop.sources.filesystem import FilesystemSource


class ClapBackend:
    def __init__(self, model: str, *, device="cpu"):
        self.model_name, self.device = model, device
        self.model = self.processor = self.torch = None

    def load(self):
        self.torch = require("torch", purpose="the audio extractor")
        transformers = require("transformers", purpose="the audio extractor")
        self.model = transformers.ClapModel.from_pretrained(self.model_name).to(self.device).eval()
        self.processor = transformers.ClapProcessor.from_pretrained(self.model_name)
        return self

    @property
    def sample_rate(self) -> int:
        return int(self.processor.feature_extractor.sampling_rate)

    def embed_audio(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.processor(audios=[np.asarray(w, dtype=np.float32) for w in waveforms], sampling_rate=self.sample_rate, return_tensors="pt")
            return self.model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()}).float().cpu().numpy()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
            return self.model.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()}).float().cpu().numpy()


def soundfile_decoder(path: Path):
    soundfile = require("soundfile", purpose="the default audio decoder")
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(rate)


def resample(waveform: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return waveform
    positions = np.linspace(0, len(waveform) - 1, max(1, int(round(len(waveform) * target / rate))))
    return np.interp(positions, np.arange(len(waveform)), waveform).astype(np.float32)


def unit_rows(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)


class AudioExtractor:
    def __init__(self, model: str, *, device="cpu", backend=None, decoder: Callable[[Path], tuple[np.ndarray, int]] | None = None, space="audioembed"):
        self.model = model
        self.backend = backend if backend is not None else ClapBackend(model, device=device)
        self.decoder = decoder if decoder is not None else soundfile_decoder
        self.space = space
        self.loaded = False

    def load(self):
        if not self.loaded:
            self.backend.load()
            self.loaded = True
        return self

    def encode(self, space: str, text: str):
        """TextEncoder slot: a CLAP text vector for this extractor's space only."""
        if space != self.space or not str(text).strip():
            return None
        self.load()
        return unit_rows(self.backend.embed_texts([str(text)]))[0]

    def extract_folder(self, source: FilesystemSource, *, space=None, overwrite=False, batch_size=8) -> dict:
        """Writes only the feature space file; no sidecar or generated tag file is touched. An
        existing space is replaced only with ``overwrite``."""
        self.load()
        space = space or self.space
        self.space = space
        if space in source.spaces() and not overwrite:
            return {"space": space, "model": self.model, "items": 0, "skipped": {}, "revision": source.revision(space), "kept_existing_space": True}
        scan = source.refresh()
        keys, vectors, skipped = [], [], {}
        rows = [(key, entry) for key, entry in sorted(scan["items"].items()) if key[0] == source.kinds[0]]
        for start in range(0, len(rows), batch_size):
            batch, waves = [], []
            for key, entry in rows[start:start + batch_size]:
                try:
                    waveform, rate = self.decoder(source.folder / entry["relpath"])
                except Exception as exc:
                    skipped[f"{key[0]}:{key[1]}"] = "undecodable:" + type(exc).__name__
                    continue
                waves.append(resample(np.asarray(waveform, dtype=np.float32), int(rate), self.backend.sample_rate))
                batch.append(key)
            if waves:
                keys.extend(batch)
                vectors.extend(unit_rows(self.backend.embed_audio(waves)))
        revision = None
        if keys:
            matrix = np.stack(vectors)
            meta = {"provenance": f"clap:{self.model}", "window_scope": "none", "kind": "audio", "built_at": source.clock()}
            revision = source.write_space(space, keys, matrix, meta=meta)
        return {"space": space, "model": self.model, "items": len(keys), "skipped": skipped, "revision": revision}


__all__ = ["AudioExtractor", "ClapBackend", "soundfile_decoder", "resample", "unit_rows"]

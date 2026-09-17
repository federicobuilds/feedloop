"""Extractors with controlled backends: no learned runtime is loaded in tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from feedloop.extractors import MissingExtra, require
from feedloop.extractors.audio import AudioExtractor, resample
from feedloop.extractors.visual import VisualExtractor, zero_shot_tags
from feedloop.sources.filesystem import FilesystemSource
from fl3_helpers import Clock, make_folder


class FakeVisualBackend:
    def __init__(self):
        self.loaded = 0

    def load(self):
        self.loaded += 1

    def open_image(self, path):
        return Path(path).name

    def embed_images(self, images):
        return np.stack([np.array([len(name) % 3 == 0, len(name) % 3 == 1, len(name) % 3 == 2, 1.0], dtype=np.float32) for name in images])

    def embed_texts(self, texts):
        return np.eye(4, dtype=np.float32)[: len(texts)]


class FakeAudioBackend:
    sample_rate = 8

    def load(self):
        pass

    def embed_audio(self, waveforms):
        return np.stack([np.array([w.mean(), w.std(), 1.0], dtype=np.float32) for w in waveforms])

    def embed_texts(self, texts):
        return np.array([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32)


def test_missing_extra_message_is_actionable():
    with pytest.raises(MissingExtra) as failure:
        require("feedloop_missing_runtime_module", purpose="the test")
    assert "pip install 'feedloop[extract]'" in str(failure.value)
    with pytest.raises(MissingExtra):
        VisualExtractor("ViT-B-32").load()


def test_visual_extractor_writes_labelled_space_and_generated_tags(tmp_path):
    media = make_folder(tmp_path, videos=2, images=3)
    source = FilesystemSource(media, tmp_path / "state", clock=Clock())
    extractor = VisualExtractor("fake-model", backend=FakeVisualBackend(), vocabulary=["alpha", "beta", "gamma", "delta"])
    report = extractor.extract_folder(source, space="visual")
    assert report["items"] == 3 and set(report["skipped"].values()) == {"no_frame_sampler"} and report["revision"]
    assert source.space_meta("visual")["provenance"] == "open_clip:fake-model" and source.space_meta("visual")["window_scope"] == "none"
    assert source.windows("visual") is None, "means only: no invented window timestamps"
    keys, matrix = source.matrix("visual")
    assert all(key[0] == "image" for key in keys) and matrix.shape == (3, 4)
    assert (media / "still-00.jpg.generated.json").exists() and not (media / "sample-00.mp4.generated.json").exists()
    assert (media / "still-00.jpg.json").read_text().startswith("{\"tags\""), "the user sidecar is untouched"
    source.refresh()
    assert any(tag in ("alpha", "beta", "gamma", "delta") for tag in source.fetch([("image", 1)])["items"][0]["tags"])
    # user edits inside the generated file survive: without --overwrite nothing is rewritten; with it only generated fields change
    generated = media / "still-00.jpg.generated.json"
    edited = json.loads(generated.read_text())
    edited["note"] = "user note"
    edited["tags"] = [t for t in edited["tags"] if t["name"] != edited["tags"][0]["name"]] + [{"name": "corrected", "category": "manual"}]
    generated.write_text(json.dumps(edited))
    before = generated.read_text()
    again = VisualExtractor("fake-model-2", backend=FakeVisualBackend(), vocabulary=["alpha", "beta", "gamma", "delta"])
    kept = again.extract_folder(source, space="visual")
    assert kept.get("kept_existing_space") is True and generated.read_text() == before, "existing output is refused by default"
    kept = again.extract_folder(source, space="visual2")
    assert "image:1" in kept["generated_files_kept"] and generated.read_text() == before
    rewritten = again.extract_folder(source, space="visual", overwrite=True)
    after = json.loads(generated.read_text())
    assert rewritten["generated_files_kept"] == [] and after["note"] == "user note" and after["model"] == "fake-model-2"
    assert {"name": "corrected", "category": "manual"} in after["tags"] and all(t["category"] in ("manual", "zero_shot") for t in after["tags"])
    assert sum(t["category"] == "zero_shot" for t in after["tags"]) >= 1 and source.space_meta("visual")["provenance"] == "open_clip:fake-model-2"
    with_frames = VisualExtractor("fake-model", backend=FakeVisualBackend(), frame_sampler=lambda path: ["frame-a", "frame-bb"])
    assert with_frames.extract_folder(source, space="visual", write_tags=False, overwrite=True)["items"] == 5


def test_zero_shot_tags_threshold():
    images = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    words = np.eye(3)
    tags = zero_shot_tags(images, words, ["a", "b", "c"], top_k=2, min_probability=0.3)
    assert [t[0][0] for t in tags] == ["a", "b"] and all(len(t) == 1 for t in tags)


def test_audio_extractor_encodes_text_for_its_space_only(tmp_path):
    media = make_folder(tmp_path, videos=3, images=1)
    source = FilesystemSource(media, tmp_path / "state", clock=Clock())
    calls = []

    def decoder(path):
        calls.append(path.name)
        if path.name.endswith("01.mp4"):
            raise OSError("no audio track")
        return np.linspace(-1, 1, 16, dtype=np.float32), 16
    extractor = AudioExtractor("fake-clap", backend=FakeAudioBackend(), decoder=decoder)
    report = extractor.extract_folder(source)
    assert report["items"] == 2 and report["skipped"] == {"video:2": "undecodable:OSError"} and calls == ["sample-00.mp4", "sample-01.mp4", "sample-02.mp4"]
    assert source.space_meta("audioembed")["provenance"] == "clap:fake-clap" and source.windows("audioembed") is None
    revision = source.revision("audioembed")
    assert AudioExtractor("other-clap", backend=FakeAudioBackend(), decoder=decoder).extract_folder(source).get("kept_existing_space") is True
    assert source.revision("audioembed") == revision and source.space_meta("audioembed")["provenance"] == "clap:fake-clap", "an existing space is kept without overwrite"
    assert AudioExtractor("other-clap", backend=FakeAudioBackend(), decoder=decoder).extract_folder(source, overwrite=True)["items"] == 2
    assert source.space_meta("audioembed")["provenance"] == "clap:other-clap"
    assert extractor.encode("visual", "rain") is None and extractor.encode("audioembed", " ") is None
    assert np.allclose(extractor.encode("audioembed", "rain"), [1.0, 0.0, 0.0])
    assert resample(np.arange(8, dtype=np.float32), 8, 4).shape == (4,) and resample(np.arange(8, dtype=np.float32), 8, 8).shape == (8,)

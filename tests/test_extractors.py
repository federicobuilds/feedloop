"""Extractors with controlled backends: no learned runtime is loaded in tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from feedloop import cli
from feedloop.extractors import MissingExtra, MissingWeights, require
from feedloop.extractors import audio, visual
from feedloop.extractors.audio import AudioExtractor, ffmpeg_decoder, resample
from feedloop.extractors.visual import VisualExtractor, ffmpeg_frame_sampler, resolve_pretrained, zero_shot_tags
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


def test_missing_extra_message_is_actionable(monkeypatch):
    with pytest.raises(MissingExtra) as failure:
        require("feedloop_missing_runtime_module", purpose="the test")
    assert "pip install 'feedloop[extract]'" in str(failure.value)
    import feedloop.extractors as extractors
    real_import = extractors.importlib.import_module
    monkeypatch.setattr(extractors.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name)) if name in ("torch", "imageio_ffmpeg") else real_import(name))
    with pytest.raises(MissingExtra):
        VisualExtractor("ViT-B-32").load()
    with pytest.raises(MissingExtra):
        ffmpeg_decoder(Path("clip.mp4"))


def test_visual_extractor_writes_labelled_space_and_generated_tags(tmp_path):
    media = make_folder(tmp_path, videos=2, images=3)
    source = FilesystemSource(media, tmp_path / "state", clock=Clock())
    extractor = VisualExtractor("fake-model", backend=FakeVisualBackend(), vocabulary=["alpha", "beta", "gamma", "delta"], frame_sampler=lambda path: [])
    report = extractor.extract_folder(source, space="visual")
    assert report["items"] == 3 and set(report["skipped"].values()) == {"no_frames"} and report["revision"]
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


def test_bare_open_clip_names_resolve_to_pretrained_weights():
    assert resolve_pretrained("ViT-B-32", None, ["openai", "laion2b_s34b_b79k"]) == "laion2b_s34b_b79k"
    assert resolve_pretrained("ViT-B-32", "openai", []) == "openai", "an explicit tag wins"
    assert resolve_pretrained("RN50", None, ["openai", "yfcc15m"]) == "openai", "open_clip's first listed tag"
    assert resolve_pretrained("hf-hub:org/repo", None, []) is None, "a prefixed name carries its own weights"
    with pytest.raises(MissingWeights) as failure:
        resolve_pretrained("custom-arch", None, [])
    assert "--pretrained" in str(failure.value)


def test_cli_threads_pretrained_and_reports_missing_weights(tmp_path, capsys, monkeypatch):
    media = make_folder(tmp_path)
    seen = {}

    class Recorder:
        def __init__(self, model, **kwargs):
            seen.update(kwargs, model=model)

        def extract_folder(self, source, **kwargs):
            raise MissingWeights("pass --pretrained TAG")
    monkeypatch.setattr(visual, "VisualExtractor", Recorder)
    args = ["extract", str(media), "--kind", "visual", "--model", "RN50", "--pretrained", "yfcc15m", "--state", str(tmp_path / "state")]
    assert cli.main(args) == 2 and seen["pretrained"] == "yfcc15m" and seen["model"] == "RN50"
    assert "--pretrained" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["extract", "--help"])
    assert "--pretrained" in capsys.readouterr().out


class FakeRun:
    def __init__(self, duration="00:00:04.00", frame=b"png", audio=b""):
        self.calls, self.duration, self.frame, self.audio = [], duration, frame, audio

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        result = type("Completed", (), {"returncode": 0, "stdout": b"", "stderr": ""})()
        if "-frames:v" in command:
            result.stdout = self.frame
        elif "f32le" in command:
            result.stdout, result.stderr = self.audio, b"" if self.audio else b"Output file does not contain any stream"
            result.returncode = 0 if self.audio else 1
        else:
            result.stderr = f"  Duration: {self.duration}, start: 0.000000" if self.duration else "Invalid data found"
        return result


class FakeImage:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def convert(self, mode):
        return (mode, self.data.getvalue())


def fake_require(name, *, purpose):
    if name == "imageio_ffmpeg":
        return type("Ffmpeg", (), {"get_ffmpeg_exe": staticmethod(lambda: "/bin/ffmpeg")})
    return type("Image", (), {"open": staticmethod(FakeImage)})


def test_ffmpeg_frame_sampler_grabs_evenly_spaced_frames(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(visual.subprocess, "run", run)
    monkeypatch.setattr(visual, "require", fake_require)
    frames = ffmpeg_frame_sampler(Path("clip.mov"), frames=4)
    assert frames == [("RGB", b"png")] * 4
    assert [c[c.index("-ss") + 1] for c in run.calls[1:]] == ["0.500", "1.500", "2.500", "3.500"]
    assert all(c[0] == "/bin/ffmpeg" for c in run.calls)
    monkeypatch.setattr(visual.subprocess, "run", FakeRun(duration=None))
    with pytest.raises(OSError):
        ffmpeg_frame_sampler(Path("empty.mov"))
    monkeypatch.setattr(visual.subprocess, "run", FakeRun(frame=b""))
    assert ffmpeg_frame_sampler(Path("audio-only.mp4")) == []


def test_ffmpeg_decoder_reads_the_container_audio_track(monkeypatch):
    samples = np.array([0.25, -0.5, 1.0], dtype="<f4")
    run = FakeRun(audio=samples.tobytes())
    monkeypatch.setattr(audio.subprocess, "run", run)
    monkeypatch.setattr(audio, "require", fake_require)
    waveform, rate = ffmpeg_decoder(Path("clip.mp4"))
    assert rate == 48000 and np.allclose(waveform, samples) and waveform.flags.writeable
    assert run.calls[0][-8:] == ["-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "-"]
    monkeypatch.setattr(audio.subprocess, "run", FakeRun())
    with pytest.raises(OSError):
        ffmpeg_decoder(Path("silent.mp4"))


def test_ffmpeg_is_the_default_and_undecodable_videos_are_skipped(tmp_path):
    assert VisualExtractor("fake", backend=FakeVisualBackend()).frame_sampler is ffmpeg_frame_sampler
    assert AudioExtractor("fake", backend=FakeAudioBackend()).decoder is ffmpeg_decoder
    media = make_folder(tmp_path, videos=2, images=1)
    source = FilesystemSource(media, tmp_path / "state", clock=Clock())

    def sampler(path):
        if path.name.endswith("00.mp4"):
            raise OSError("zero-byte file")
        return ["frame-a"]
    report = VisualExtractor("fake", backend=FakeVisualBackend(), frame_sampler=sampler).extract_folder(source)
    assert report["items"] == 2 and list(report["skipped"].values()) == ["undecodable:OSError"]

    def missing(path):
        raise MissingExtra("needs imageio_ffmpeg")
    with pytest.raises(MissingExtra):
        VisualExtractor("fake", backend=FakeVisualBackend(), frame_sampler=missing).extract_folder(source, overwrite=True)
    with pytest.raises(MissingExtra):
        AudioExtractor("fake", backend=FakeAudioBackend(), decoder=missing).extract_folder(source)

import numpy as np
import pytest

from feedloop.ingest import ingest_space
from feedloop.server.app import build_engine


class VectorExtractor:
    def __init__(self, dim):
        self.dim = dim
        self.seen = []

    def extract_rows(self, folder, rows):
        keys, vectors = [], []
        for key, entry in rows:
            self.seen.append(entry["relpath"])
            keys.append(key)
            vectors.append(np.full(self.dim, folder.joinpath(entry["relpath"]).read_bytes()[0], dtype=np.float32))
        return keys, vectors, {}


def test_incremental_ingest_publishes_learned_spaces_to_similar(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "seed.mp4").write_bytes(b"a")
    (media / "match.mp4").write_bytes(b"b")
    state = tmp_path / "state"
    source, engine = build_engine(media, state, initialize=True, demo_space=False)
    model = VectorExtractor(2)
    recipe = {"model": "one", "kind": "visual"}
    first = ingest_space(source, model, space="visual", recipe=recipe, kinds=("video",))
    assert first["processed"] == 2
    unchanged = ingest_space(source, model, space="visual", recipe=recipe, kinds=("video",))
    assert unchanged["processed"] == 0 and unchanged["revision"] == first["revision"]

    nested = media / "nested"
    nested.mkdir()
    (nested / "new.mp4").write_bytes(b"c")
    updated = ingest_space(source, model, space="visual", recipe=recipe, kinds=("video",))
    assert model.seen == ["match.mp4", "seed.mp4", "nested/new.mp4"]
    assert updated["reused"] == 2 and updated["processed"] == 1 and updated["revision"] != first["revision"]
    rows = engine.similar(("video", 2))["items"]
    assert len(rows) == 2 and all(item["similar"]["sources"] == ["visual"] for item in rows)

    revised = {"model": "two", "kind": "visual"}
    other = VectorExtractor(3)
    with pytest.raises(ValueError, match="different recipe"):
        ingest_space(source, other, space="visual", recipe=revised, kinds=("video",))
    ingest_space(source, other, space="visual-two", recipe=revised, kinds=("video",))
    assert source.matrix("visual")[1].shape == (3, 2)
    assert source.matrix("visual-two")[1].shape == (3, 3)
    _, switched = build_engine(media, state, space_roles={"visual": "visual-two"})
    assert switched.roles["visual"] == "visual-two"

    (media / "seed.mp4").write_bytes(b"changed")
    (media / "match.mp4").unlink()
    rerun = ingest_space(source, model, space="visual", recipe=recipe, kinds=("video",))
    assert model.seen[-1] == "seed.mp4"
    assert rerun["items"] == 2 and rerun["reused"] == 1

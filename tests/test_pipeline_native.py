"""WP2a: the native pipeline reads spaces through the host's roles, over unit rows, fenced by
revision, and a cold start with images requested serves images."""
import random

import numpy as np

from feedloop.engine import Engine, initialize_stores
from feedloop.pipeline import KindMatrices
from fl2_helpers import MemoryCatalog, MemorySignals, MemorySpaces, catalog_row, unit_rows
from test_engine import REQUEST, build_catalog, clock  # noqa: F401  (clock is a fixture)

ROLES = {"visual": "visual", "semantic": "semvisual", "voice": "audioembed", "sound": "audiomix"}


def native_engine(path, fake, catalog, spaces, *, signals=None, roles=ROLES):
    path.mkdir()
    signals = signals or MemorySignals({("video", i): {"rating": None, "engagement_count": 0,
                                                       "watch": {"watched_s": 500.0, "last_at": 40.0 - i, "visit_days": [0], "intervals": [(0, 500)]}}
                                        for i in (1, 11, 21, 31)}, observed_at=50.0)
    ledger_path, tuner_path = str(path / "events.sqlite"), str(path / "tuner.sqlite")
    initialize_stores(ledger_path=ledger_path, tuner_path=tuner_path, cutover_ts=80.0, clock=fake)
    return Engine(catalog=catalog, signals=signals, spaces=spaces, ledger_path=ledger_path, tuner_path=tuner_path,
                  clock=fake, automatic_tuning=False, space_roles=roles,
                  config={"explore_slots": 0, "control_rate": 0.0, "cooldown_days": 0.0})


def served(page):
    assert page["status"] == "ok", page
    return [(i["kind"], i["id"], i["score"], tuple(i["explanation"]["sources"])) for i in page["items"]]


def feed_over(path, now, fake, matrices, roles=ROLES):
    now[0] = 90.0
    catalog = build_catalog()
    eng = native_engine(path, fake, catalog, MemorySpaces(matrices), roles=roles)
    now[0] = 100.0
    # the engine draws the generation seed from the global random with no request override, so
    # equal seeds make runs comparable; the state is restored so later test files are unaffected
    state = random.getstate()
    try:
        random.seed(7)
        return served(eng.feed(REQUEST, record_delivery=False))
    finally:
        random.setstate(state)


def test_custom_space_name_ranks_like_the_builtin_name(tmp_path, clock):
    now, fake = clock
    keys = sorted(build_catalog().rows)
    m = np.abs(unit_rows(len(keys), 16, 1))
    builtin = feed_over(tmp_path / "a", now, fake, {"visual": (keys, m)})
    custom = feed_over(tmp_path / "b", now, fake, {"custom_visual": (keys, m)}, roles={**ROLES, "visual": "custom_visual"})
    assert custom == builtin
    assert any("visual" in sources for *_rest, sources in custom) and any(score > 0 for _k, _i, score, _s in custom)


def test_row_scale_does_not_change_the_order(tmp_path, clock):
    now, fake = clock
    keys = sorted(build_catalog().rows)
    m = np.abs(unit_rows(len(keys), 16, 1))
    scaled = m.copy()
    scaled[keys.index(("video", 5))] *= 100.0
    assert feed_over(tmp_path / "b", now, fake, {"visual": (keys, scaled)}) == feed_over(tmp_path / "a", now, fake, {"visual": (keys, m)})
    zero = m.copy()
    zero[0] = 0.0
    ids, _m, index = KindMatrices(MemorySpaces({"visual": (keys, zero)})).get("visual", keys[0][0])
    assert keys[0][1] not in index and len(ids) == sum(1 for k in keys if k[0] == keys[0][0]) - 1


def test_revision_bump_with_the_same_array_rebuilds_the_view():
    keys = [("video", i) for i in range(1, 5)]
    m = unit_rows(4, 8, 1)
    spaces = MemorySpaces({"visual": (keys, m)})
    matrices = KindMatrices(spaces)
    first = matrices.get("visual", "video")
    assert matrices.get("visual", "video") is first
    m[:] = unit_rows(4, 8, 2)
    assert matrices.get("visual", "video") is first, "same array, same revision: the view is kept"
    spaces.revisions["visual"] = 2
    fresh = matrices.get("visual", "video")
    assert fresh is not first and np.allclose(fresh[1], m)
    spaces.revisions["visual"] = None
    assert matrices.get("visual", "video") is not matrices.get("visual", "video"), "no revision: nothing is cached"


def test_cold_start_with_images_requested_serves_images(tmp_path, clock):
    now, fake = clock
    now[0] = 90.0
    rows = [catalog_row("video", i) for i in (1, 2, 3)] + [catalog_row("image", i, duration=0.0) for i in range(1, 9)]
    eng = native_engine(tmp_path / "a", fake, MemoryCatalog(rows), MemorySpaces({}), signals=MemorySignals({}, observed_at=50.0))
    now[0] = 100.0
    items = served(eng.feed(REQUEST, record_delivery=False))
    kinds = [kind for kind, *_rest in items]
    assert all(sources == ("fallback",) for *_rest, sources in items) and "image" in kinds and "video" in kinds, items


def test_unit_view_survives_huge_and_tiny_finite_rows():
    keys = [("video", 1), ("video", 2)]
    m = np.array([[3.0, 4.0], [3.0, 4.0]], dtype=np.float32) * np.array([[1e20], [1e-30]], dtype=np.float32)
    ids, unit, _index = KindMatrices(MemorySpaces({"visual": (keys, m)})).get("visual", "video")
    assert ids == [1, 2] and unit.dtype == np.float32
    assert np.allclose(unit, [[0.6, 0.8], [0.6, 0.8]])


def test_a_space_named_means_is_the_legacy_look_view(tmp_path, clock):
    now, fake = clock
    keys = sorted(build_catalog().rows)
    m = np.abs(unit_rows(len(keys), 16, 1))
    ids, view, _index = KindMatrices(MemorySpaces({"means": (keys, m)})).paired("visual", "semvisual", "video")
    assert ids == [i for kind, i in keys if kind == "video"] and np.allclose(view, m[[keys.index(("video", i)) for i in ids]])
    items = feed_over(tmp_path / "a", now, fake, {"means": (keys, m)})
    assert any("visual" in sources for *_rest, sources in items), items
    both = {"visual": (keys, m), "semvisual": (keys, np.abs(unit_rows(len(keys), 16, 2)))}
    assert KindMatrices(MemorySpaces({**both, "means": (keys, m)})).paired("visual", "semvisual", "video")[1].shape[1] == 32, \
        "the paired roles win over the legacy space"

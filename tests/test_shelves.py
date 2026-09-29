"""Home insight shelves: one sentence per item from its own explanation (2026-09-29)."""
import random

import numpy as np

from feedloop import pipeline, serving
from fl2_helpers import MemorySpaces, unit_rows
from test_engine import REQUEST, build_catalog, clock  # noqa: F401  (clock is a fixture)
from test_pipeline_native import native_engine

NAMES = {1: "Outdoor", 2: "Beach", 3: "Night", 4: "42"}


def pick(i, explanation, category="scene"):
    return {"kind": "video", "id": i, "category": category, "explanation": explanation}


def liked(tag, share=1.0):
    return {"sources": ["tags"], "tag_contributions": [{"tag_id": tag, "contribution": share}, {"tag_id": 3, "contribution": -5.0}]}


EXPLORE = {"sources": ["explore"], "explore": True}
FALLBACK = {"sources": ["fallback"], "fallback": "no_positive_feature_match"}


def shelves(items):
    return [item["shelf"] for item in serving.assign_shelves(items, NAMES)]


def test_top_positive_named_tag_gives_because_you_like():
    assert serving.shelf_label(liked(1), NAMES) == "Because you like Outdoor"
    numeric_first = {"tag_contributions": [{"tag_id": 4, "contribution": 9.0}, {"tag_id": 99, "contribution": 8.0},
                                           {"tag_id": 2, "contribution": 1.0}]}
    assert serving.shelf_label(numeric_first, NAMES) == "Because you like Beach"
    assert serving.shelf_label({"tag_contributions": [{"tag_id": 3, "contribution": -2.0}]}, NAMES) is None


def test_explore_gives_something_new():
    assert serving.shelf_label(EXPLORE, NAMES) == "Something new"


def test_fallback_without_evidence_gives_new_to_you():
    assert serving.shelf_label(FALLBACK, NAMES) == "New to you"


def test_no_insight_falls_to_category():
    assert shelves([pick(i, {"sources": ["visual"]}, "action") for i in range(3)]) == ["action"] * 3


def test_small_shelf_falls_back_to_category():
    items = [pick(1, liked(1)), pick(2, liked(1)), pick(3, liked(2), "beach-cat")] + [pick(i, EXPLORE) for i in range(4, 7)]
    assert shelves(items) == ["scene", "scene", "beach-cat"] + ["Something new"] * 3


def test_at_most_eight_insight_shelves_first_appearance_wins():
    names = {t: f"Tag{t}" for t in range(10)}
    items = [pick(t * 3 + k, liked(t)) for t in range(10) for k in range(3)]
    result = [item["shelf"] for item in serving.assign_shelves(items, names)]
    insight = list(dict.fromkeys(s for s in result if s.startswith("Because")))
    assert insight == [f"Because you like Tag{t}" for t in range(8)]
    assert result[-6:] == ["scene"] * 6


def test_every_item_on_one_named_shelf_and_names_unique_in_order():
    items = [pick(i, liked(1)) for i in range(3)] + [pick(i, EXPLORE) for i in range(3, 6)] + [pick(i, liked(2)) for i in range(6, 9)]
    result = serving.assign_shelves(items, NAMES)
    assert all(isinstance(item["shelf"], str) for item in result) and len(result) == 9
    assert list(dict.fromkeys(item["shelf"] for item in result)) == ["Because you like Outdoor", "Something new", "Because you like Beach"]


def test_cold_start_is_one_new_to_you_shelf():
    ranked = {"items": [{"kind": "video", "id": i, "category": "other", "explanation": dict(FALLBACK)} for i in range(1, 7)], "has_more": False}
    result = serving.build_feed(ranked, names=NAMES, offset=0)
    assert {item["shelf"] for item in result["items"]} == {"New to you"}


def test_partial_watch_gives_continue_watching():
    assert serving.shelf_label({**liked(1), "watch_fraction": 0.4}, NAMES) == "Continue watching"
    assert serving.shelf_label({**liked(1), "watch_fraction": 0.95}, NAMES) == "Because you like Outdoor"


def test_named_nearest_like_on_a_look_pick_gives_because_you_watched():
    look = {**liked(1), "sources": ["tags", "visual"], "nearest_like": {"kind": "video", "id": 7, "cosine": 0.9, "title": "Harbor"}}
    assert serving.shelf_label(look, NAMES) == "Because you watched Harbor"
    untitled = {**look, "nearest_like": {"kind": "video", "id": 7, "cosine": 0.9}}
    assert serving.shelf_label(untitled, NAMES) == "Because you like Outdoor"


def test_without_watch_or_seed_the_previous_rules_hold():
    assert [serving.shelf_label(e, NAMES) for e in (liked(2), EXPLORE, FALLBACK, {"sources": ["visual"]})] == \
        ["Because you like Beach", "Something new", "New to you", None]


def test_explanation_additions_leave_order_and_scores_unchanged(tmp_path, clock, monkeypatch):
    now, fake = clock
    keys = sorted(build_catalog().rows)
    m = np.abs(unit_rows(len(keys), 16, 1))

    def page(path, minimum):
        monkeypatch.setattr(pipeline, "NEAREST_LIKE_MIN", minimum)
        now[0] = 90.0
        eng = native_engine(path, fake, build_catalog(), MemorySpaces({"visual": (keys, m)}))
        now[0] = 100.0
        state = random.getstate()
        try:
            random.seed(7)
            return eng.feed(REQUEST, record_delivery=False)["items"]
        finally:
            random.setstate(state)

    off, on = page(tmp_path / "off", np.inf), page(tmp_path / "on", -1.0)
    assert [(i["kind"], i["id"], i["score"]) for i in on] == [(i["kind"], i["id"], i["score"]) for i in off]
    assert not any("nearest_like" in i["explanation"] for i in off)
    named = [(i["id"], i["explanation"]["nearest_like"]) for i in on if "nearest_like" in i["explanation"]]
    assert named and all(seed["title"] and seed["id"] != item for item, seed in named)

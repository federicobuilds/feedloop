"""The v0.4.0 prepared-ranking boundary: host components reach rank_page unchanged, fences still apply."""
from unittest.mock import Mock

import numpy as np

from feedloop import engine as engine_module, ranking
from feedloop.engine import PREPARED_CONTRACT, PREPARED_RANKING_REVISION, RANKING_REVISION
from test_engine import REQUEST, clock, make_engine  # noqa: F401  (clock is a fixture)
from fl2_helpers import unit_rows

PAGE = {**REQUEST, "images": False}


def preparation(**overrides):
    comps = [(("video", i), {1: 300.0}, "acts", 1.0 / i, 0.5, None, None, 1.0, None) for i in range(2, 9)]
    prep = {"comps": comps, "image_comps": [], "target_shares": {"acts": 1.0},
            "explanations": {row[0]: {"sources": ["tags"], "host": row[0][1]} for row in comps},
            "admitted": None, "excluded": [("video", 1)], "seeds": [], "explore": [("video", 40)], "control": [("video", 41)],
            "fallback": lambda: [("video", 50)], "fallback_reasons": ["no_positive_feature_match"],
            "profile": {"weights": {1: 1.0}}, "source_counts": {"tags": len(comps)}}
    prep.update(overrides)
    return prep


def prepared_engine(tmp_path, fake, prep, config=None):
    eng, signals, spaces = make_engine(tmp_path, fake, config=config)
    calls = []
    def prepare_feed(**kwargs):
        calls.append(kwargs)
        return prep
    eng.prepare_feed = prepare_feed
    return eng, calls, spaces


def test_prepared_components_reach_rank_page_unchanged(tmp_path, clock, monkeypatch):
    now, fake = clock
    now[0] = 90.0
    prep = preparation()
    eng, calls, _spaces = prepared_engine(tmp_path, fake, prep, config={"explore_slots": 2, "control_rate": 1.0})
    now[0] = 100.0
    selector, rebuild = Mock(wraps=ranking.rank_page), Mock(wraps=ranking.rank)
    monkeypatch.setattr(engine_module.ranking, "rank_page", selector)
    monkeypatch.setattr(engine_module.ranking, "rank", rebuild)
    page = eng.feed(PAGE)
    assert page["status"] == "ok", page
    rebuild.assert_not_called()
    selector.assert_called_once()
    args, kwargs = selector.call_args
    assert args == (prep["comps"], prep["image_comps"])
    for name in ("target_shares", "explanations", "admitted", "explore", "control", "fallback_reasons"):
        assert kwargs[name] == prep[name], name
    assert kwargs["fallback"] == [] and ("video", 1) in kwargs["excluded"] and kwargs["page_size"] == 24
    provenance = page["items"][0]["provenance"]
    (call,) = calls
    assert call["seed"] == provenance["seed"] == kwargs["seed"]
    assert call["kinds"] == ("video",) and call["catalog"]["present"] == eng_present(eng)
    assert call["config"]["contributor_affinity_weight"] == kwargs["config"]["contributor_affinity_weight"]
    assert provenance["ranking_revision"] == PREPARED_RANKING_REVISION != RANKING_REVISION
    served = [(i["kind"], i["id"]) for i in page["items"]]
    assert {("video", 40), ("video", 41)} <= set(served)
    assert all(key in {row[0] for row in prep["comps"]} | {("video", 40), ("video", 41)} for key in served)
    assert PREPARED_CONTRACT["historical_ranker_reproduction"] is False
    assert all(v["historical_ranker_reproduction"] is False for v in ranking.VARIANT_CONTRACT.values())


def eng_present(eng):
    return {(kind, i) for kind in ("video", "image") for i in range(1, 101)} & set(eng.catalog.rows)


def test_fallback_is_prepared_only_when_the_selection_is_empty(tmp_path, clock):
    now, fake = clock
    now[0] = 90.0
    asked = []
    prep = preparation(comps=[], explanations={}, explore=[], control=[], fallback=lambda: asked.append(1) or [("video", 50)])
    eng, _calls, _spaces = prepared_engine(tmp_path, fake, prep)
    now[0] = 100.0
    page = eng.feed(PAGE)
    assert asked == [1] and [(i["kind"], i["id"]) for i in page["items"]] == [("video", 50)]
    assert page["items"][0]["explanation"]["sources"] == ["fallback"]


def test_unknown_or_ineligible_prepared_keys_are_rejected(tmp_path, clock):
    now, fake = clock
    now[0] = 90.0
    image_row = (("image", 3), 0.9, 1.0, None)
    for n, prep in enumerate((preparation(explore=[("video", 999)]),  # not in the pinned catalog
                 preparation(image_comps=[image_row]),  # images were not requested
                 preparation(control=[("video", 7)]),  # excluded by the request intent
                 preparation(seeds=[("video", 5)]))):  # not the request's seeds
        (tmp_path / str(n)).mkdir()
        eng, _calls, _spaces = prepared_engine(tmp_path / str(n), fake, prep)
        now[0] = 100.0
        page = eng.feed({**PAGE, "intent": {"excluded_items": [{"kind": "video", "id": 7}]}})
        assert (page["status"], page["error_code"]) == ("error", "ranking_contract_unavailable"), page
        now[0] = 90.0


def test_every_space_the_preparer_reads_is_fenced(tmp_path, clock):
    now, fake = clock
    now[0] = 90.0
    eng, calls, spaces = prepared_engine(tmp_path, fake, None)
    keys = sorted(eng.catalog.rows)
    spaces.matrices["means"] = (keys, np.abs(unit_rows(len(keys), 32, 3)))
    spaces.revisions["means"] = 1
    prep = preparation()
    def moving(**kwargs):
        spaces.revisions["means"] = 2  # the paired means space is no role, yet the preparer used it
        return prep
    eng.prepare_feed = moving
    now[0] = 100.0
    page = eng.feed(PAGE)
    assert page["status"] == "partial" and page["error_code"] == "ranking_provenance_unavailable", page
    assert eng._cursors == {}
    eng.prepare_feed = lambda **kwargs: prep
    stable = eng.feed({**PAGE, "request_id": "req-2", "client_request_id": "client-2"})
    assert stable["status"] == "ok" and stable["items"][0]["provenance"]["revision_status"] == "stable"

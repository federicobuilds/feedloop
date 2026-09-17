"""More-like-this and text search over in-memory slot facts: recruitment, cosine
over complete features, per-space budgets, eligibility, duplicate seeds, one
duration per item, fallback, provenance, and the window/mean/fusion/taste-band
search math with a fake encoder."""
import ast
import copy
import inspect
import json
from unittest.mock import Mock

import numpy as np
import pytest

from feedloop import discovery
from feedloop.slots import DEFAULT_SPACE_ROLES
from fl2_helpers import MemoryCatalog, MemorySignals, MemorySpaces, catalog_row, md5_file


def build_sources(complete, *, omitted=(), fail_hydration=False, durations=None, files=None, spaces=None,
                  links=None, signals=None, encoder=None, extra_rows=(), clock=lambda: 2000000.0):
    """complete: {item_id: {tag: seconds}} for every primary item in the corpus."""
    rows = [catalog_row("video", i, duration=(durations or {}).get(i, 1000.0),
                        files=(files or {}).get(i, [{"fingerprints": [{"type": "md5", "value": f"{i:032x}"}]}]))
            for i in complete if i not in omitted] + list(extra_rows)
    features = {("video", i): {"tag_seconds": dict(tags), "watched_tag_seconds": None,
                               "tag_categories": {t: "acts" for t in tags}} for i, tags in complete.items()}
    catalog = MemoryCatalog(rows, features, {10: "Tag ten", 20: "Tag twenty"})
    if fail_hydration:
        catalog.fetch = Mock(side_effect=RuntimeError("synthetic hydration failure"))
    return discovery.Sources(catalog=catalog, spaces=spaces or MemorySpaces({}), signals=signals or MemorySignals(),
                             links=links, encoder=encoder, clock=clock)


def run_similar(complete, *, seed_ids=(1,), config=None, context=None, limit=20, offset=0, resolver=None, **fixture):
    sources = build_sources(complete, **fixture)
    cfg = {"cooldown_days": 0, "contributor_weight": 0, "diversity": 0, **(config or {})}
    return discovery.similar(sources, seed_ids=list(seed_ids), config=cfg, context=context or {}, limit=limit,
                             offset=offset, resolve_eligibility=resolver), sources


def ids(result):
    return [row["id"] for row in result["items"]]


class Links:
    def __init__(self, mapping):
        self.mapping = mapping

    def links(self, keys=None):
        return {k: v for k, v in self.mapping.items() if keys is None or k in set(keys)}


# ------------------------------------------------------------------- similar

def test_review_b_duplicate_seed_cannot_starve_remaining_similars():
    complete = {i: {10: 100.0} for i in range(1, 7)}
    same = [md5_file("d")]
    result, _ = run_similar(complete, files={1: same, 2: same, 3: same}, limit=2)
    assert ids(result) == [4, 5]
    assert result["total"] == 3
    assert result["has_more"]


def test_each_embedding_source_recruits_without_seed_or_candidate_tags():
    for role, config in [("visual", {"semantic_share": 0, "audio_weight": 0, "mix_weight": 0}),
                         ("semantic", {"semantic_share": 1, "audio_weight": 0, "mix_weight": 0}),
                         ("voice", {"audio_weight": 1, "mix_weight": 0}),
                         ("sound", {"audio_weight": 0, "mix_weight": 1})]:
        space = DEFAULT_SPACE_ROLES[role]
        keys = [("video", 1), ("video", 2)]
        spaces = MemorySpaces({space: (keys, np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))})
        result, _ = run_similar({1: {}, 2: {}}, config={"embed_weight": 1, **config}, spaces=spaces)
        assert ids(result) == [2], role
        detail = result["items"][0]["similar"]
        assert detail["sources"] == [space]
        assert detail["tag_term"] == 0
        assert detail["embedding_term"] == 1


def test_identity_only_and_featureless_fallback_respect_eligibility():
    links = Links({("video", 1): {"actor"}, ("video", 2): {"actor"}})
    result, _ = run_similar({1: {}, 2: {}}, config={"embed_weight": 0, "contributor_weight": 1}, links=links)
    assert ids(result) == [2]
    assert result["items"][0]["similar"]["sources"] == ["identity"]
    result, _ = run_similar({1: {}}, extra_rows=[catalog_row("video", 2), catalog_row("video", 3)],
                            context={"eligibility": '{"video_ids":[1,2]}'})
    assert ids(result) == [2]
    assert result["items"][0]["similar"]["fallback"] == "no_positive_feature_match"


def test_streamed_embedding_retrieval_uses_bounded_conversion_and_stable_cuts(monkeypatch):
    # 2,049 candidate vectors after the seed: the second batch holds item 2147 and item 3
    keys = [("video", i) for i in range(100, 2148)] + [("video", 3)]
    matrix = np.tile(np.array([1.0, 0.0], dtype=np.float16), (len(keys), 1))
    seed_keys = [("video", 1)]
    spaces = MemorySpaces({"visual": (seed_keys + keys, np.vstack([np.array([[1.0, 0.0]], dtype=np.float16), matrix]))})
    dot = Mock(side_effect=discovery.chunked_dot)
    monkeypatch.setattr(discovery, "chunked_dot", dot)
    eligible = [3] + list(range(100, 200))      # 101 eligible, equal scores, over the 40-vector budget
    complete = {i: {} for i in [1] + eligible}
    result, _ = run_similar(complete, config={"embed_weight": 1, "semantic_share": 0, "audio_weight": 0, "mix_weight": 0,
                                              "candidate_pool": 8}, spaces=spaces, limit=200,
                            context={"eligibility": json.dumps({"video_ids": eligible})})
    # bounded conversion: rows are scored per 2,048-row batch after the allowlist filter, never as one matrix
    assert [call.args[0].shape for call in dot.call_args_list] == [(100, 2), (1, 2)]
    assert all(call.args[0].dtype == np.float16 for call in dot.call_args_list)
    # stable cut: exactly max(40, pool // 4) = 40 vectors survive, equal scores break by id across batches
    assert ids(result) == [3] + list(range(100, 139))
    assert result["total"] == 40
    assert all(row["similar"]["sources"] == ["visual"] for row in result["items"])


def test_similar_content_keeps_provenance_and_only_caches_stable_features():
    keys = [("video", 1), ("video", 2)]
    unit = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    for revision, expected in [({"visual": 1}, "stable"), ({"visual": None}, "unavailable_or_changed")]:
        spaces = MemorySpaces({"visual": (keys, unit)}, revisions=revision)
        result, sources = run_similar({1: {10: 100.0}, 2: {10: 100.0}}, spaces=spaces)
        provenance = result["items"][0]["provenance"]
        assert provenance["revision_status"] == expected
        assert provenance["config"]["cooldown_days"] == 0
        assert provenance["seed"] == 0
        assert provenance["experiment"] is None
        assert result["provenance"] == provenance
        if expected == "stable":
            # nothing is cached by discovery, so the publication contract is copy semantics: the
            # published snapshot and the returned features are copies, never aliases of each other
            # or of the slot data
            original = copy.deepcopy(provenance)
            provenance["config"]["diversity"] = 0.9
            assert result["provenance"] == original and result["provenance"] is not provenance
            assert result["items"][0]["provenance"] is not result["provenance"]
            slot_features = copy.deepcopy(sources.catalog.feature_rows)
            result["items"][0]["similar"]["tag_contributions"][0]["seconds"] = -1.0
            result["items"][0]["similar"]["tag_contributions"].clear()
            assert sources.catalog.feature_rows == slot_features
            assert sources.catalog.feature_rows[("video", 2)]["tag_seconds"] == {10: 100.0}
    # a revision that moves during the build cannot publish a stable generation
    spaces = MemorySpaces({"visual": (keys, unit)})
    spaces.revision = Mock(side_effect=[1, 2, 2, 2])
    result, _ = run_similar({1: {10: 100.0}, 2: {10: 100.0}}, spaces=spaces)
    assert result["provenance"]["revision_status"] == "unavailable_or_changed"


def test_complete_features_recruiting_query_does_not_define_cosine(monkeypatch):
    # tag 10 and tag 20 each cover three corpus items, so their idf is equal (the source stub's
    # semantics) and the cosine of {10, 20} against the seed's {10} is 1/sqrt(2); items 4 and 5
    # carry only tag 20, which recruits nothing, and only define the document frequency
    complete = {1: {10: 100.0}, 2: {10: 100.0, 20: 100.0}, 3: {10: 100.0}, 4: {20: 100.0}, 5: {20: 100.0}}
    result, sources = run_similar(complete)
    assert ids(result) == [3, 2]
    assert result["items"][1]["similar"]["tag_similarity"] == pytest.approx(0.7071, abs=5e-5)
    for row in result["items"]:
        detail = row["similar"]
        assert detail["score"] == pytest.approx(((detail["tag_term"] + detail["embedding_term"])
                                                 * detail["face_blend_multiplier"] + detail["face_term"]) * detail["history_multiplier"])
        assert sum(c["contribution"] for c in detail["tag_contributions"]) == pytest.approx(detail["tag_similarity"], abs=1e-4)
        assert "selection" in detail
    details = lambda r: [(row["id"], row["similar"]["tag_similarity"], row["similar"]["score"], row["similar"]["tag_contributions"])
                         for row in r["items"]]
    # a recruiting perturbation: item 3 recruits with twice the seconds, so the rough recruiting
    # order flips from [2, 3] to [3, 2]; its complete single-tag vector still scores a cosine of 1.0
    recruited = []
    real_admit = discovery.admit_sources
    def spy(source_orders, *args, **kwargs):
        recruited.append([key[1] for key in source_orders["tags"]])
        return real_admit(source_orders, *args, **kwargs)
    monkeypatch.setattr(discovery, "admit_sources", spy)
    run_similar(complete)
    other, _ = run_similar({**complete, 3: {10: 200.0}})
    assert recruited == [[2, 3], [3, 2]], "the recruiting order changed"
    assert ids(other) == ids(result) and details(other) == details(result)
    monkeypatch.setattr(discovery, "admit_sources", real_admit)
    # featureless corpus items never enter the document-frequency denominator
    padded, _ = run_similar({**complete, 6: {}, 7: {}, 8: {}})
    assert ids(padded) == ids(result) and details(padded) == details(result)
    assert padded["items"][1]["similar"]["tag_similarity"] == pytest.approx(0.7071, abs=5e-5)
    # ... also when the frequencies are unequal, where the denominator would show
    skewed = {1: {10: 100.0}, 2: {10: 100.0, 20: 100.0}, 3: {10: 100.0}}
    idf = {10: np.log(1.0 + 3 / 3), 20: np.log(1.0 + 3 / 1)}
    expected = idf[10] / np.sqrt(idf[10] ** 2 + idf[20] ** 2)
    for corpus in (skewed, {**skewed, 6: {}, 7: {}}):
        value, _ = run_similar(corpus)
        assert value["items"][1]["similar"]["tag_similarity"] == pytest.approx(round(float(expected), 4), abs=5e-5)


def test_transient_intent_tags_and_seeds_reach_real_handler():
    complete = {2: {10: 100.0}, 3: {20: 100.0}}
    result, _ = run_similar(complete, seed_ids=[], context={"intent": '{"tag_ids":[10]}'})
    assert ids(result) == [2]
    result, _ = run_similar({1: {10: 100.0}, 2: {10: 100.0}}, seed_ids=[], context={"intent": '{"seed_video_ids":[1]}'})
    assert result["items"][0]["similar"]["seed_ids"] == [1]


def test_similar_transient_exclusions_and_empty_allowlist():
    for context in [{"intent": '{"exclude_video_ids":[2]}'}, {"eligibility": '{"video_ids":[]}'}]:
        result, _ = run_similar({1: {10: 100.0}, 2: {10: 100.0}}, context=context)
        assert result["items"] == []


def test_similar_resolves_saved_filter_snapshot_references():
    snapshot_id = "elig:v1:" + "a" * 64
    context = {"eligibility": json.dumps({"snapshot_id": snapshot_id})}
    resolver = Mock(return_value={"video": [3], "image": []})
    complete = {1: {10: 100.0}, 2: {10: 100.0}, 3: {10: 100.0}}
    result, _ = run_similar(complete, context=context, resolver=resolver)
    assert ids(result) == [3]
    resolver.assert_called_once_with(snapshot_id)
    # without a resolver, references stay rejected
    with pytest.raises(ValueError, match="eligibility snapshot"):
        run_similar(complete, context=context)


def test_reads_do_not_schedule_destructive_cleanup():
    tree = ast.parse(inspect.getsource(discovery))
    assert not any(isinstance(n, ast.Attribute) and n.attr == "delete" for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Constant) and isinstance(n.value, str) and "DELETE FROM" in n.value for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("write_tags", "write_links") for n in ast.walk(tree))
    complete = {1: {10: 100.0}, 2: {10: 100.0}}
    result, sources = run_similar(complete)
    assert ids(result) == [2]
    assert all(call[0] in ("fetch", "enumerate") for call in sources.catalog.calls)


def test_partial_empty_and_failed_hydration_do_not_delete():
    complete = {1: {10: 100.0}, 2: {10: 100.0, 20: 100.0}, 3: {10: 100.0}}
    for omitted in [(2,), (2, 3)]:
        result, sources = run_similar(complete, omitted=omitted)
        assert ids(result) == [i for i in [3, 2] if i not in omitted]
        assert all(call[0] in ("fetch", "enumerate") for call in sources.catalog.calls)
    with pytest.raises(RuntimeError, match="synthetic hydration failure"):
        run_similar(complete, fail_hydration=True)


def test_multifile_seed_and_candidate_use_same_duration():
    complete = {1: {10: 100.0, 20: 40.0}, 2: {10: 100.0, 20: 40.0}, 3: {10: 100.0, 20: 20.0}}
    two_files = {i: [md5_file(chr(96 + 2 * i)), md5_file(chr(97 + 2 * i))] for i in complete}
    result, _ = run_similar(complete, durations={i: 100.0 for i in complete}, files=two_files)
    by_id = {row["id"]: row["similar"] for row in result["items"]}
    assert by_id[2]["tag_similarity"] == 1.0
    assert by_id[2]["duration_s"] == 100.0
    assert by_id[3]["tag_similarity"] < by_id[2]["tag_similarity"]
    assert ids(result) == [2, 3]


def test_multiseed_merges_one_duration_per_item():
    complete = {1: {10: 60.0, 20: 5.0}, 4: {10: 40.0, 20: 15.0}, 2: {10: 100.0, 20: 20.0}}
    result, _ = run_similar(complete, seed_ids=[1, 4], durations={1: 100.0, 4: 100.0, 2: 200.0},
                            files={1: [md5_file("a"), md5_file("b")], 4: [md5_file("c"), md5_file("d")],
                                   2: [md5_file("e"), md5_file("f")]})
    assert result["items"][0]["similar"]["tag_similarity"] == 1.0


def test_similar_pool_cut_is_independent_of_recruiting_order():
    # tag 10 recruits its 400 strongest carriers (2..401), tag 20 recruits 402..502: 501 candidates,
    # cut to the 500 strongest recruiting vectors by (score, id)
    complete = {i: {10: 100.0, **({20: 1.0} if i >= 402 or i == 1 else {})} for i in range(1, 503)}
    pages = []
    for order in [range(2, 503), reversed(range(2, 503))]:
        ordered = {i: complete[i] for i in [1] + list(order)}
        result, _ = run_similar(ordered, limit=600)
        pages.append(ids(result))
    assert pages[0] == pages[1]
    assert sorted(pages[0]) == list(range(2, 502)), "the pool cut is by rough score then id, never by recruiting order"


# -------------------------------------------------------------------- search

class Encoder:
    def __init__(self, vectors):
        self.vectors, self.calls = vectors, []

    def encode(self, space, text):
        self.calls.append((space, text))
        value = self.vectors.get(space)
        if isinstance(value, Exception):
            raise value
        return value


def search_spaces(*, look=True, sound=True):
    """Windows: item 1 has two windows (best .9 at t=30, mean .7); item 2 one window .8 at t=5."""
    look_rows = ([("video", 1), ("video", 1), ("video", 2), ("video", 3)],
                 np.array([30.0, 60.0, 5.0, 0.0], dtype=np.float32),
                 np.array([[0.9, np.sqrt(1 - 0.81)], [0.5, np.sqrt(0.75)], [0.8, 0.6], [0.05, 0.99875]], dtype=np.float32))
    sound_rows = ([("video", 2), ("video", 3)], np.array([12.0, 3.0], dtype=np.float32),
                  np.array([[0.7, np.sqrt(1 - 0.49)], [0.6, 0.8]], dtype=np.float32))
    windows = {}
    if look:
        windows[DEFAULT_SPACE_ROLES["semantic"]] = look_rows
    if sound:
        windows[DEFAULT_SPACE_ROLES["sound"]] = sound_rows
    return MemorySpaces({}, revisions={}, windows=windows)


def search_sources(spaces, encoder, **kwargs):
    complete = {1: {10: 100.0}, 2: {10: 100.0}, 3: {20: 100.0}}
    return build_sources(complete, spaces=spaces, encoder=encoder, **kwargs)


def test_text_blend_preserves_window_mean_fusion_and_taste_bands():
    query = np.array([1.0, 0.0], dtype=np.float32)
    encoder = Encoder({DEFAULT_SPACE_ROLES["semantic"]: query, DEFAULT_SPACE_ROLES["sound"]: query})
    sources = search_sources(search_spaces(), encoder)
    # look: best window .75 weight plus .25 of the window mean; the best timestamp is the best window's
    look = discovery.search(sources, "sunset", "look", config={"personalization": 0})
    assert look["status"] == "ok" and [i["id"] for i in look["items"]] == [1, 2]
    top = look["items"][0]["search"]
    assert top["query_score"] == pytest.approx(0.75 * 0.9 + 0.25 * 0.7, abs=1e-4)
    assert top["best_window_similarity"] == pytest.approx(0.9, abs=1e-4) and top["item_mean_similarity"] == pytest.approx(0.7, abs=1e-4)
    assert top["best_t"] == 30.0
    assert all(i["id"] != 3 for i in look["items"]), "below the .12 semantic floor"
    # both: reciprocal-rank fusion over ranks, 1/(60+rank), not a cosine average
    both = discovery.search(sources, "both: sunset", "look", config={"personalization": 0})
    assert both["status"] == "ok" and both["components"] == {"look": {"status": "ok"}, "sound": {"status": "ok"}}
    scores = {i["id"]: i["search"]["query_score"] for i in both["items"]}
    assert scores[2] == pytest.approx(1 / 61 + 1 / 60, abs=1e-6)     # second in look, first in sound
    assert scores[1] == pytest.approx(1 / 60, abs=1e-6)              # look only
    assert scores[3] == pytest.approx(1 / 61, abs=1e-6)              # sound only (.6 >= .08)
    assert [i["id"] for i in both["items"]] == [2, 1, 3]
    assert both["items"][2]["search"]["best_t"] == 3.0, "best timestamp from the selected component"
    # taste reorders only anchored bands within 2% of the top query score
    rows = [(1.00, 1.0, 1.0, 0.0, ("video", 1)), (0.999, 0.999, 0.999, 0.0, ("video", 2)), (0.50, 0.5, 0.5, 0.0, ("video", 3))]
    taste = {("video", 1): -1.0, ("video", 2): 1.0, ("video", 3): 1.0}
    assert [row[4][1] for row in discovery.rerank_query_bands(rows, taste, 0.15)] == [2, 1, 3]
    assert [row[4][1] for row in discovery.rerank_query_bands(rows, taste, 0.0)] == [1, 2, 3]
    # exact fusion helper: ranks, not scores
    fused = discovery.fuse([(0.9, 0.9, 0.9, 0.0, "a"), (0.1, 0.1, 0.1, 0.0, "b")], [(0.2, 0.2, 0.2, 1.0, "b")], None)
    assert [(key, round(score, 6)) for score, _mx, _mn, _bt, key in fused] == [("b", round(1 / 61 + 1 / 60, 6)), ("a", round(1 / 60, 6))]
    assert fused[0][3] == 0.0, "the look component supplies the timestamp when both match"
    # constants carried from the source
    assert (discovery.D_MIN_SIMILARITY, discovery.D_SOUND_MIN_SIM, discovery.D_MEAN_WEIGHT, discovery.D_POOL,
            discovery.D_PERSONALIZATION, discovery.CHUNK) == (0.12, 0.08, 0.25, 400, 0.15, 4096)


def test_search_modes_and_missing_features():
    query = np.array([1.0, 0.0], dtype=np.float32)
    encoder = Encoder({DEFAULT_SPACE_ROLES["semantic"]: query, DEFAULT_SPACE_ROLES["sound"]: query})
    sources = search_sources(search_spaces(), encoder)
    assert discovery.search(sources, "x", "bogus")["error_code"] == "invalid_mode"
    assert discovery.search(sources, "   ", "look")["status"] == "empty"
    assert discovery.search(sources, "SOUND:   ", "look")["status"] == "empty"
    # prefixes override the selected mode, case-insensitively; visual is an alias of look
    assert discovery.search(sources, "SoUnD: rain", "look")["items"][0]["search"]["space"] == "sound"
    assert discovery.search(sources, "rain", "visual")["items"][0]["search"]["space"] == "look"
    assert discovery.search(sources, "rain", "sound")["items"][0]["id"] == 2, "sound uses the sound space, not voice"
    assert encoder.calls[-1][0] == DEFAULT_SPACE_ROLES["sound"]
    # no encoder: an explicit no-feature component and result, no fabricated matches
    result = discovery.search(search_sources(search_spaces(), None), "rain", "look")
    assert (result["status"], result["items"], result["components"]) == ("no-feature", [], {"look": {"status": "no-feature"}})
    assert "error_code" not in result
    # no window data for the space: no-feature too
    result = discovery.search(search_sources(search_spaces(sound=False), encoder), "rain", "sound")
    assert result["status"] == "no-feature" and result["components"] == {"sound": {"status": "no-feature"}}
    # an actual encoder failure is unavailable
    broken = Encoder({DEFAULT_SPACE_ROLES["semantic"]: RuntimeError("encoder down")})
    result = discovery.search(search_sources(search_spaces(), broken), "rain", "look")
    assert (result["status"], result["error_code"]) == ("unavailable", "encoder_or_features_unavailable")
    assert result["components"] == {"look": {"status": "unavailable"}}
    # a partially working both query keeps the working component and reports partial
    partial = discovery.search(search_sources(search_spaces(), Encoder({DEFAULT_SPACE_ROLES["semantic"]: query,
                                                                          DEFAULT_SPACE_ROLES["sound"]: RuntimeError("x")})),
                               "both: rain", "look", config={"personalization": 0})
    assert partial["status"] == "partial" and partial["error_code"] == "search_component_unavailable"
    assert partial["components"] == {"look": {"status": "ok"}, "sound": {"status": "unavailable"}}
    assert [i["id"] for i in partial["items"]] == [1, 2]
    # a malformed encoder vector is a component failure, never a match
    bad = Encoder({DEFAULT_SPACE_ROLES["semantic"]: np.array([np.nan, 0.0], dtype=np.float32)})
    assert discovery.search(search_sources(search_spaces(), bad), "rain", "look")["components"]["look"]["status"] == "unavailable"
    # eligibility and exclusions apply; search never writes
    sources = search_sources(search_spaces(), encoder)
    only_two = discovery.search(sources, "rain", "look", context={"eligibility": '{"video_ids":[2]}'}, config={"personalization": 0})
    assert [i["id"] for i in only_two["items"]] == [2]
    excluded = discovery.search(sources, "rain", "look", context={"intent": '{"exclude_video_ids":[1]}'}, config={"personalization": 0})
    assert [i["id"] for i in excluded["items"]] == [2]
    assert all(call[0] in ("fetch", "enumerate") for call in sources.catalog.calls)
    assert sources.signals.rows == {}

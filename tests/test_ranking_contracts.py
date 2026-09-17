"""Offline ranking contracts over the installed feedloop.ranking and feedloop.taste modules.

Run directly with Python, or collect with pytest. No service or database access.
"""
import ast
import builtins
from collections import defaultdict
import copy
import datetime
import __future__
import json
from functools import partial
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing

import feedloop.catalog
import feedloop.discovery
import feedloop.engine
import feedloop.ledger
import feedloop.profiles
import feedloop.ranking as RANKING_MODULE
import feedloop.serving
import feedloop.slots
import feedloop.taste as TASTE
import feedloop.tuning

RANKING_PATH = Path(RANKING_MODULE.__file__)
SHIPPED_MODULES = [Path(module.__file__) for module in (
    TASTE, RANKING_MODULE, feedloop.ledger, feedloop.slots, feedloop.profiles, feedloop.tuning,
    feedloop.serving, feedloop.discovery, feedloop.catalog, feedloop.engine)]
PURE_TOP_LEVEL = set(sys.stdlib_module_names) | {"numpy", "feedloop"}


def forbidden_io(*args, **kwargs):
    raise AssertionError("unexpected import or external I/O in ranking logic")


class _Ranking:
    """The real module, with the serialized-text boundary given its JSON decoder.

    Only parse_context needs JSON; rank itself has no decoder."""
    def __init__(self, module):
        self._module = module
        self.parse_context = partial(module.parse_context, decode=json.loads)

    def __getattr__(self, name):
        return getattr(self._module, name)


RANKING = _Ranking(RANKING_MODULE)


# Pinned verdict oracle, copied for the comparison test only (never production code):
# the like bar, rating override, engagement override and their constants.
ORACLE_LIKE_ABS_MIN_S, ORACLE_LIKE_ABS_MAX_S, ORACLE_LIKE_ABS_DURATION_SHARE = 120.0, 240.0, 0.05
ORACLE_D_DISLIKE_MIN_WATCH = 60.0


def oracle_like_bar(duration):
    if duration <= 0:
        return ORACLE_LIKE_ABS_MIN_S
    return min(max(ORACLE_LIKE_ABS_DURATION_SHARE * duration, ORACLE_LIKE_ABS_MIN_S), ORACLE_LIKE_ABS_MAX_S)


def oracle_rating_override(rating, is_like, is_dis, strength):
    if rating is None or strength <= 0:
        return is_like, is_dis, 1.0
    if rating >= 80:
        return True, False, 1.0 + strength
    if rating <= 40:
        return False, True, 1.0 + 0.5 * strength
    return False, False, 1.0


def oracle_engagement_override(engagement_count, is_like, is_dis, boost):
    if engagement_count and engagement_count > 0:
        return True, False, boost * (1.0 + 0.4 * min(engagement_count, 5))
    return is_like, is_dis, boost


def oracle_verdict(watched_s, duration, *, finished_ratio, abandon_ratio, dislike_min_watch=ORACLE_D_DISLIKE_MIN_WATCH,
                   rating=None, rating_strength=1.0, engagement_count=None):
    bar = oracle_like_bar(duration)
    if duration <= 0:
        is_like, is_dis = watched_s >= bar, False
    else:
        completion = watched_s / duration
        is_like = completion >= finished_ratio or watched_s >= bar
        is_dis = (not is_like and completion <= abandon_ratio and dislike_min_watch <= watched_s < bar)
    is_like, is_dis, boost = oracle_rating_override(rating, is_like, is_dis, rating_strength)
    return oracle_engagement_override(engagement_count, is_like, is_dis, boost)


def compile_definitions(nodes, ns, filename):
    nodes = copy.deepcopy(nodes)
    for node in nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise AssertionError("only function definitions may be executed")
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, filename, "exec", flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), ns)


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        # Fail even if ranking logic catches an unexpected I/O error.
        for target in ["threading.Thread.start", "socket.socket", "sqlite3.connect"]:
            guard = self.enterContext(patch(target, side_effect=forbidden_io))
            self.addCleanup(guard.assert_not_called)
        original_import = builtins.__import__

        def import_dependency(name, globals=None, locals=None, fromlist=(), level=0):
            if not level and name.split(".")[0] not in PURE_TOP_LEVEL:
                forbidden_io()
            return original_import(name, globals, locals, fromlist, level)

        self.enterContext(patch("builtins.__import__", side_effect=import_dependency))


class HarnessSafetyContracts(OfflineTestCase):
    def test_ranking_uses_same_math_objects_without_host_imports(self):
        self.assertIs(RANKING_MODULE.item_preferences, TASTE.item_preferences)
        self.assertIs(RANKING_MODULE.chunked_dot, TASTE.chunked_dot)
        self.assertIs(RANKING_MODULE.select.__globals__["cosine"], RANKING_MODULE.cosine)
        for module in (RANKING_MODULE, TASTE):
            imported = {value.__name__ for value in vars(module).values() if isinstance(value, type(sys))}
            self.assertTrue(all(name.split(".")[0] in PURE_TOP_LEVEL for name in imported), imported)

    def test_ranker_imports_only_canonical_shared_math_without_fallback(self):
        tree = ast.parse(RANKING_PATH.read_text())
        imports = [(n.module, n.level) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                   and (n.module or "").startswith("feedloop")]
        self.assertEqual(imports, [("feedloop.taste", 0)])


def old_select(scored, *, want, diversity, calibration, target_shares, cosine):
    chosen, chosen_vectors = [], []
    counts = defaultdict(int)
    pool = scored[:]
    scale = abs(scored[0][0] if scored else 1.0) or 1.0
    while pool and len(chosen) < want:
        best_idx, best_val = 0, -1e18
        for idx, (rel, sid, vec, cat) in enumerate(pool):
            sim = 0.0
            for prev in chosen_vectors:
                s = cosine(vec, prev)
                if s > sim:
                    sim = s
            deficit = max(0.0, target_shares.get(cat, 0.0) - counts[cat] / (len(chosen) or 1))
            val = rel / scale - diversity * sim + calibration * deficit
            if val > best_val:
                best_idx, best_val = idx, val
        rel, sid, vec, cat = pool.pop(best_idx)
        chosen.append(sid)
        chosen_vectors.append(dict(vec))
        counts[cat] += 1
    return chosen


def fixture(seed, size=40):
    rng = random.Random(seed)
    scored = [(rng.choice([0.1, 0.5, 1.0]), sid,
               {t: rng.random() for t in rng.sample(range(30), rng.randrange(12))},
               rng.choice(["acts", "bodyparts", "other"])) for sid in range(size)]
    return sorted(scored, key=lambda x: x[0], reverse=True)


def raw_config(**changes):
    config = dict(half_life_days=21.0, min_watch_seconds=20.0, finished_ratio=0.45,
                  abandon_ratio=0.15, dislike_min_watch_seconds=60.0, short_watch_ratio=0.5,
                  history_limit=600, rating_strength=1.0, dislike_strength=1.0, profile_tags=24,
                  candidate_pool=600, bodyparts_weight=0.3, max_tag_share=0.35, length_floor_seconds=120.0,
                  embedding_weight=0.35, taste_audio_weight=0.1, taste_mix_weight=0.1,
                  diversity=0.35, calibration=0.25, cooldown_days=0.0, recovery_days=0.0,
                  impression_discount=0.95, contributor_affinity_weight=0.5,
                  image_events_enabled=True, include_images=True, images_share=0.2,
                  explore_slots=0, control_rate=0.1, experiment=None,
                  vector_spaces={"visual": "visual-v1", "semantic": "semantic-v1", "voice": "voice-v1", "sound": "sound-v1"})
    config.update(changes)
    assert set(config) == set(RANKING.REQUIRED_CONFIG)
    return config


def raw_fixture(*, kind="video", count=4):
    catalog, features = [], []
    for id_ in range(1, count + 1):
        catalog.append(dict(kind=kind, id=id_, available_at="2026-01-01T00:00:00Z", known_at="2026-01-01T00:00:00Z",
                            removed_at=None, eligible=True, duration_s=600 if kind == "video" else 0,
                            tag_ids=[10], feature_presence=[], duplicate_verified=False, duplicate_group=None))
        features.append(dict(kind=kind, id=id_, revision="f1", model_revision="fixture-v1", complete=True,
                             effective_at="2026-01-01T00:00:00Z", known_at="2026-01-01T00:00:00Z",
                             tag_seconds={"10": 100}, tag_categories={"10": "acts"}, watched_tag_seconds={},
                             evidence_ids=[], identity_ids=[], vectors={}))
    inputs = {"catalog": catalog, "features": features, "evidence": []}
    context = dict(id="context-1", session_id="session-1", cutoff="2026-01-03T00:00:00Z", now="2026-01-03T00:00:00Z",
                   intent={"tag_ids": [], "seed": None}, kinds=[kind], limit=20, offset=0,
                   eligible_ids=[{"kind": kind, "id": id_} for id_ in range(1, count + 1)])
    return inputs, context


class RawRankingContracts(OfflineTestCase):
    def test_extracted_verdict_matches_core_across_thresholds_and_overrides(self):
        verdict = RANKING_MODULE.item_preferences.__globals__["verdict"]
        self.assertIs(verdict, feedloop.profiles.verdict)
        for watched in [0, 5, 20, 60, 119, 120, 240, 600]:
            for duration in [0, 20, 120, 600, 10000]:
                for rating in [None, 0, 20, 60, 80, 100]:
                    for engagement_count in [None, 0, 1, 6]:
                        for strength in [0, 1]:
                            kw = dict(finished_ratio=0.45, abandon_ratio=0.15, dislike_min_watch=60,
                                      rating=rating, rating_strength=strength, engagement_count=engagement_count)
                            self.assertEqual(verdict(watched, duration, **kw), oracle_verdict(watched, duration, **kw))

    def test_shared_hard_eligibility_matches_all_variants_without_duplicate_selection(self):
        inputs, context = raw_fixture(count=6)
        images, _ = raw_fixture(kind="image", count=2)
        inputs["catalog"] += images["catalog"]
        inputs["features"] += images["features"]
        inputs["catalog"][5]["eligible"] = False
        for row in inputs["catalog"][2:4]:
            row.update(duplicate_verified=True, duplicate_group="same")
        context.update(kinds=["video", "image"], eligible_ids={"video": None, "image": None})
        context["intent"].update(tag_ids=[10], exclude_video_ids=[5])
        at = "2025-12-24T00:00:00Z"
        day = int(RANKING._timestamp(at) // 86400)
        for sid, seconds in ((1, 600), (2, 60), (3, 5)):
            value = {"watch": {"watched_s": seconds, "last_at": at, "visit_days": [day]}}
            if sid == 2:
                value["rating"] = 20
            inputs["evidence"].append(dict(event_id=f"video-{sid}", kind="video", id=sid, type="initial_state",
                                           value=value, known_at="2026-01-02T00:00:00Z", occurred_at=None))
        for iid in (1, 2):
            inputs["evidence"].append(dict(event_id=f"image-{iid}", kind="image", id=iid, type="initial_state",
                                           value={"rating": 20, "engagement_count": iid - 1}, known_at=at, occurred_at=None))
        config = raw_config(cooldown_days=45, recovery_days=120)
        before = copy.deepcopy((inputs, context, config))
        verdict = RANKING.shared_hard_eligibility(inputs, context=context, config=config)
        self.assertEqual(verdict, ["image:2", "video:3", "video:4"])
        with self.assertRaises(TypeError):
            RANKING.shared_hard_eligibility(inputs, context, config)
        for variant in RANKING.SUPPORTED_VARIANTS:
            ranked = RANKING.rank(inputs, context=context, config=config, seed=42, variant=variant)
            excluded = {f"{r['kind']}:{r['id']}" for r in ranked["exclusions"] if r["reason"] != "not_selected"}
            self.assertEqual(set(verdict), {f"{r['kind']}:{r['id']}" for r in inputs["catalog"]} - excluded)
            self.assertTrue(all(f"{r['kind']}:{r['id']}" in verdict for r in ranked["items"]))
        no_images = RANKING.shared_hard_eligibility(inputs, context=context, config={**config, "include_images": False})
        self.assertFalse(any(key.startswith("image:") for key in no_images))
        no_image_events = RANKING.shared_hard_eligibility(inputs, context=context, config={**config, "image_events_enabled": False})
        self.assertIn("image:1", no_image_events)
        unresolved = {**context, "eligible_ids": {"snapshot_id": "elig:v1:" + "a" * 64}}
        with self.assertRaisesRegex(ValueError, "resolved membership"):
            RANKING.shared_hard_eligibility(inputs, context=unresolved, config=config)
        with self.assertRaisesRegex(ValueError, "resolved membership"):
            RANKING.rank(inputs, context=unresolved, config=config, seed=42)
        self.assertEqual((inputs, context, config), before)

    def test_initial_watch_snapshot_equals_real_prior_days_and_subsequent_delta(self):
        inputs, context = raw_fixture(count=3)
        days = ["2026-01-01T00:10:00Z", "2026-01-02T00:10:00Z"]
        epoch_days = [int(RANKING._timestamp(at) // 86400) for at in days]
        events = [dict(event_id=str(n), kind="video", id=1, type="watch", value=300,
                       occurred_at=at, known_at=at, start_at=at[:11] + "00:00:00Z", end_at=at)
                  for n, at in enumerate(days)]
        initial = dict(event_id="initial", kind="video", id=1, type="initial_state", occurred_at=None,
                       known_at="2026-01-02T01:00:00Z", value={"watch": {
                           "watched_s": 600, "last_at": days[-1], "visit_days": epoch_days}})
        config = raw_config(cooldown_days=0, recovery_days=0)
        full = RANKING.rank({**inputs, "evidence": events}, context=context, config=config, seed=42)
        captured = RANKING.rank({**inputs, "evidence": [initial]}, context=context, config=config, seed=42)
        self.assertEqual(full["items"], captured["items"])
        self.assertEqual(full["admission_trace"]["invariants"]["profile"], captured["admission_trace"]["invariants"]["profile"])
        self.assertGreater(captured["items"][0]["explanation"]["profile"]["watch_evidence_s"], 0)
        delta = {**events[-1], "event_id": "later", "value": 60, "occurred_at": "2026-01-02T02:01:00Z",
                 "known_at": "2026-01-02T02:01:00Z", "start_at": "2026-01-02T02:00:00Z", "end_at": "2026-01-02T02:01:00Z"}
        a = RANKING.rank({**inputs, "evidence": events + [delta]}, context=context, config=config, seed=42)
        b = RANKING.rank({**inputs, "evidence": [initial, delta]}, context=context, config=config, seed=42)
        self.assertEqual(a["items"], b["items"])

    def test_initial_watch_invalid_metadata_rejected_even_outside_catalog(self):
        inputs, context = raw_fixture()
        valid = {"watched_s": 600, "last_at": "2026-01-01T00:10:00Z", "visit_days": [20454]}
        invalid = [{**valid, "watched_s": value} for value in (True, -1, float("nan"), float("inf"), 10**1000)]
        invalid += [{**valid, "last_at": value} for value in (None, 1, "2026-01-01T00:10:00", "2026-01-03T00:00:00Z")]
        invalid += [{**valid, "visit_days": value} for value in ([True], [1.0], [2, 1], [1, 1], [999999], None)]
        invalid += [{k: v for k, v in valid.items() if k != field} for field in valid]
        for kind, watch in [("video", value) for value in invalid] + [("image", valid)]:
            event = dict(event_id="snapshot", kind=kind, id=999, type="initial_state", occurred_at=None,
                         known_at="2026-01-02T00:00:00Z", value={"watch": watch})
            with self.subTest(kind=kind, watch=watch), self.assertRaises(ValueError):
                RANKING.rank({**inputs, "evidence": [event]}, context=context, config=raw_config(), seed=1)

    def test_audited_global_cut_and_corrected_union_change_only_admission(self):
        inputs, context = raw_fixture(count=70)
        context['intent']['seed_video_ids'] = [1]
        inputs['features'][-1]['tag_seconds'] = {}
        for row in (inputs['features'][0], inputs['features'][-1]):
            row['vectors'] = {'visual-v1':[1], 'semantic-v1':[1]}
        pair = RANKING.compare_admission(inputs, context=context, config=raw_config(candidate_pool=50), seed=42)
        old, new = pair['control'], pair['treatment']
        self.assertNotIn({'kind':'video','id':70}, old['admission_trace']['admitted_ids'])
        self.assertIn({'kind':'video','id':70}, new['admission_trace']['admitted_ids'])
        self.assertEqual(old['admission_trace']['invariants'], new['admission_trace']['invariants'])
        old_scores = {(row['kind'],row['id']):row['score'] for row in old['items']}
        for row in new['items']:
            key = (row['kind'],row['id'])
            if key in old_scores:
                self.assertEqual(row['score'], old_scores[key])
        self.assertEqual(pair['baseline']['revision'], '11b5266068fc46425eacc4662e4bd38152a55061')
        self.assertFalse(old['variant_contract']['historical_ranker_reproduction'])

    def test_temporal_spellings_order_by_instants_then_event_id(self):
        inputs, context = raw_fixture(count=2)
        inputs['features'][1]['tag_seconds'] = {'20': 100}
        inputs['evidence'] = [dict(event_id=id_, kind='video', id=1, type='rating',
            occurred_at=stamp, known_at=stamp, value=value) for id_, stamp, value in (
                ('a', '2026-01-01T01:00:00Z', 20), ('b', '2026-01-01T01:00:00.5Z', 100))]
        first = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        original = copy.deepcopy(inputs)
        for event in inputs['evidence']:
            for field in ('occurred_at', 'known_at'):
                event[field] = datetime.datetime.fromisoformat(event[field].replace('Z', '+00:00')).isoformat(timespec='microseconds').replace('+00:00', 'Z')
        second = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        self.assertEqual({k:v for k,v in first.items() if k != 'timings'}, {k:v for k,v in second.items() if k != 'timings'})
        self.assertGreater(first['items'][0]['score'], 0)
        inputs['evidence'][1].update(occurred_at='2026-01-01T01:00:00.000000Z', known_at='2026-01-01T01:00:00.000000Z')
        inputs['evidence'][0].update(occurred_at='2026-01-01T01:00:00Z', known_at='2026-01-01T01:00:00Z')
        forward = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        inputs['evidence'].reverse()
        backward = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        self.assertEqual({k:v for k,v in forward.items() if k != 'timings'}, {k:v for k,v in backward.items() if k != 'timings'})
        self.assertGreater(backward['items'][0]['score'], 0)
        self.assertEqual(original['evidence'][0]['occurred_at'], '2026-01-01T01:00:00Z')

    def test_evidence_timestamps_fail_closed_before_item_filtering(self):
        inputs, context = raw_fixture(count=2)
        event = dict(event_id='rating', kind='video', id=999, type='rating', value=100,
                     occurred_at='2026-01-01T00:00:00Z', known_at='2026-01-01T00:00:00Z')
        for field in ('occurred_at', 'known_at'):
            for value in (None, '', 'not-a-time', '2026-01-01Z', '2026-01-01T00:00:00', 0, True):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'evidence timestamp'):
                    RANKING.rank({**inputs, 'evidence': [{**event, field: value}]}, context=context, config=raw_config(), seed=1)
            missing = {k:v for k,v in event.items() if k != field}
            with self.assertRaisesRegex(ValueError, 'evidence timestamp'):
                RANKING.rank({**inputs, 'evidence': [missing]}, context=context, config=raw_config(), seed=1)
        with self.assertRaisesRegex(ValueError, 'evidence timestamp'):
            RANKING.rank({**inputs, 'evidence': [{**event, 'known_at':'2025-12-31T00:00:00Z'}]}, context=context, config=raw_config(), seed=1)

    def test_shared_page_scores_selects_and_paginates_one_kind_key_sequence(self):
        comps = [(('video', i), {}, 'acts', 1.0, None, None, None, 1.0, None) for i in range(1, 5)]
        images = [(('image', 1), 0.9, 1.0, -0.3), (('image', 2), 0.85, 1.0, 0.3)]
        kwargs = dict(config=raw_config(images_share=0.5, contributor_affinity_weight=0.5),
                      target_shares={}, seed=42, allowed={'video': None, 'image': None},
                      excluded=set(), duplicate_groups={}, seeds=(), page_size=2)
        pages = [RANKING.rank_page(comps, images, offset=o, limit=2, **kwargs) for o in (0, 2, 4)]
        keys = [item['key'] for page in pages for item in page['items']]
        self.assertEqual(keys, [('video', 1), ('image', 2), ('video', 2), ('image', 1), ('video', 3), ('video', 4)])
        self.assertEqual([p['total'] for p in pages], [6, 6, 6])
        self.assertEqual([p['next_offset'] for p in pages], [2, 4, None])
        self.assertEqual([i['position'] for p in pages for i in p['items']], list(range(6)))
        self.assertTrue(all(i['score'] == i['explanation']['score'] for p in pages for i in p['items']))

    def test_fingerprints_require_every_file_exact_md5_and_kind_scope(self):
        file = lambda value, type_='md5': {'fingerprints': [{'type': type_, 'value': value}]}
        rows = [{'kind': 'video', 'id': 1, 'files': [file('a' * 32)]},
                {'kind': 'video', 'id': 2, 'files': [file('A' * 32)]},
                {'kind': 'image', 'id': 1, 'files': [file('a' * 32)]},
                {'kind': 'video', 'id': 3, 'files': [file('a' * 32), file('b' * 32)]},
                {'kind': 'video', 'id': 4, 'files': [file('a' * 32), file('b' * 32, 'oshash')]}]
        groups = RANKING.fingerprint_groups(rows)
        self.assertEqual(groups[('video', 1)], groups[('video', 2)])
        self.assertNotEqual(groups[('video', 1)], groups[('video', 3)])
        self.assertNotIn(('video', 4), groups)
        self.assertEqual(RANKING.eligible_ranked_items([('video', 2), ('image', 1)], {}, set(), groups,
                                                     already_selected=[('video', 1)]), [('image', 1)])

    def test_admission_replay_changes_only_the_gate_without_variant_aliases(self):
        self.assertEqual(RANKING.SUPPORTED_VARIANTS, ("current", "audit_old", "admission_only"))
        for changed in (False, True):
            inputs, context = raw_fixture(count=3)
            context["intent"]["seed"] = {"kind": "video", "id": 1}
            for feature in inputs["features"]:
                feature["vectors"] = {"visual-v1": [1, 0], "semantic-v1": [1, 0], "voice-v1": [1, 0]}
            if changed:
                inputs["features"][2]["tag_seconds"] = {}
                del inputs["features"][2]["vectors"]["semantic-v1"]
            config = raw_config()
            before = copy.deepcopy((inputs, context, config))
            pair = RANKING.compare_admission(inputs, context=context, config=config, seed=42)
            control, union = pair["control"], pair["treatment"]
            self.assertFalse(pair["historical_ranker_reproduction"])
            self.assertEqual(control["admission_trace"]["policy"], "audited_tag_gated_look_then_cut")
            self.assertEqual(union["admission_trace"]["policy"], "union")
            self.assertEqual(control["admission_trace"]["invariants"], union["admission_trace"]["invariants"])
            self.assertEqual(control["admission_trace"]["admitted_ids"] != union["admission_trace"]["admitted_ids"], changed)
            common = {item["id"]: item for item in control["items"]}
            for item in union["items"]:
                if item["id"] in common:
                    self.assertEqual(item["score"], common[item["id"]]["score"])
            current = RANKING.rank(inputs, context=context, config=config, seed=42)
            self.assertEqual({k: v for k, v in current.items() if k not in ("timings", "variant_contract")},
                             {k: v for k, v in union.items() if k not in ("timings", "variant_contract")})
            union["admission_trace"]["invariants"]["configuration"]["vector_spaces"]["visual"] = "mutated-output"
            self.assertEqual((inputs, context, config), before)
        with self.assertRaisesRegex(ValueError, "overlap"):
            RANKING.compare_admission(inputs, context=context, config=raw_config(
                experiment={"knob": "embedding_weight", "candidate": 0.6}), seed=42)

    def test_route_inventory_is_before_source_budgets_not_sampled_outputs(self):
        inputs, context = raw_fixture(count=450)
        context["intent"]["tag_ids"] = [10]
        result = RANKING.rank(inputs, context=context, config=raw_config(candidate_pool=40), seed=42)
        self.assertLess(result["source_counts"]["tags"], 450)
        inventory = result["route_inventory"]
        self.assertTrue(inventory["complete"])
        self.assertEqual(inventory["routes"]["tags"]["stage"], "pre_budget")
        self.assertEqual(len(inventory["routes"]["tags"]["eligible_ids"]), 450)

    def test_weak_fallback_distinguishes_missing_history_tags_and_embeddings(self):
        inputs, context = raw_fixture(count=2)
        for feature in inputs["features"]:
            feature["tag_seconds"] = {}
        result = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        expected = ["no_preference_history", "no_positive_tag_profile", "no_embedding_profile"]
        self.assertEqual(result["fallback"], {"active": True, "reasons": expected})
        for item in result["items"]:
            self.assertEqual(item["explanation"]["strength"], "weak")
            self.assertEqual(item["explanation"]["fallback_reasons"], expected + ["no_positive_feature_match"])
            self.assertEqual(item["explanation"]["revisions"]["feature"], "f1")

    def test_interleaved_items_keep_their_nominating_arm_score(self):
        inputs, context = raw_fixture(count=2)
        inputs["features"][0]["vectors"] = {"visual-v1": [1, 0], "semantic-v1": [1, 0]}
        inputs["features"][1].update(tag_seconds={"20": 100}, vectors={"visual-v1": [0, 1], "semantic-v1": [0, 1]})
        image, _ = raw_fixture(kind="image", count=1)
        image["features"][0].update(tag_seconds={"20": 1}, vectors={"visual-v1": [1, 0], "semantic-v1": [1, 0]})
        inputs["catalog"] += image["catalog"]
        inputs["features"] += image["features"]
        inputs["evidence"] = [dict(event_id="initial-image", client_event_id=None, session_id=None,
                                   kind="image", id=1, type="initial_state", occurred_at=None,
                                   known_at="2026-01-01T00:00:00Z", value={"rating": 100, "engagement_count": 0, "play_count": 0})]
        result = RANKING.rank(inputs, context=context, seed=1, config=raw_config(
            experiment={"knob": "embedding_weight", "candidate": 0.9}))
        self.assertEqual({row["explanation"]["arm"] for row in result["items"]}, {"base", "cand"})
        for row in result["items"]:
            self.assertEqual(row["score"], row["explanation"]["score"])

    def test_explicit_image_tag_quantum_is_total_sixty_without_fake_watch(self):
        inputs, context = raw_fixture(count=3)
        for feature, tag in zip(inputs["features"], (10, 20, 30)):
            feature["tag_seconds"] = {str(tag): 100}
        images, _ = raw_fixture(kind="image", count=2)
        images["features"][0]["tag_seconds"] = {"10": 1, "20": 1}
        images["features"][1]["tag_seconds"] = {"30": 1}
        inputs["catalog"] += images["catalog"]
        inputs["features"] += images["features"]
        inputs["evidence"] = [dict(event_id=f"initial-{i}", client_event_id=None, session_id=None,
                                    kind="image", id=i, type="initial_state", occurred_at=None,
                                    known_at="2026-01-01T00:00:00Z",
                                    value={"rating": 100, "engagement_count": 0, "play_count": 0}) for i in (1, 2)]
        result = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        details = {item["id"]: item["explanation"] for item in result["items"]}
        self.assertAlmostEqual(details[3]["tag_score"], 2 * details[1]["tag_score"])
        self.assertEqual(details[3]["profile"]["watch_evidence_s"], 0)
        self.assertEqual(details[3]["profile"]["explicit_only_items"], 2)

    def test_mixed_kind_pagination_is_a_stable_disjoint_prefix(self):
        inputs, context = raw_fixture(count=50)
        images, _ = raw_fixture(kind="image", count=10)
        inputs["catalog"] += images["catalog"]
        inputs["features"] += images["features"]
        context.update(kinds=["video", "image"], limit=10,
                       eligible_ids=[{"kind": row["kind"], "id": row["id"]} for row in inputs["catalog"]])
        context["intent"]["tag_ids"] = [10]
        first = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)["items"]
        context["offset"] = 10
        second = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)["items"]
        first_keys = [(i["kind"], i["id"]) for i in first]
        second_keys = [(i["kind"], i["id"]) for i in second]
        self.assertFalse(set(first_keys) & set(second_keys))
        context.update(offset=0, limit=20)
        whole = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)["items"]
        self.assertEqual(first_keys + second_keys, [(i["kind"], i["id"]) for i in whole])

    def test_exploration_pages_do_not_repeat_outside_pool_items(self):
        inputs, context = raw_fixture(count=40)
        for feature in inputs["features"][20:]:
            feature["tag_seconds"] = {"20": 100}
        context.update(limit=5)
        context["intent"]["tag_ids"] = [10]
        config = raw_config(explore_slots=2, control_rate=1.0)
        first = RANKING.rank(inputs, context=context, config=config, seed=42)["items"]
        context["offset"] = 5
        second = RANKING.rank(inputs, context=context, config=config, seed=42)["items"]
        self.assertFalse({i["id"] for i in first} & {i["id"] for i in second})
        self.assertTrue(any(i["explanation"]["sources"] == ["control"] for i in first))

    def test_disabled_image_kind_cannot_exhaust_primary_fallback_budget(self):
        inputs, context = raw_fixture(count=1)
        images, _ = raw_fixture(kind="image", count=650)
        inputs["catalog"] += images["catalog"]
        inputs["features"] += images["features"]
        context.update(kinds=["video", "image"], eligible_ids=[{"kind": r["kind"], "id": r["id"]} for r in inputs["catalog"]])
        result = RANKING.rank(inputs, context=context, config=raw_config(include_images=False), seed=42)
        self.assertEqual([(i["kind"], i["id"]) for i in result["items"]], [("video", 1)])

    def test_raw_rank_is_repeatable_order_independent_and_does_not_mutate(self):
        inputs, context = raw_fixture()
        context["intent"]["tag_ids"] = [10]
        config = raw_config()
        original = copy.deepcopy((inputs, context, config))
        first = RANKING.rank(inputs, context=context, config=config, seed=42)
        self.assertEqual((inputs, context, config), original)
        inputs["catalog"].reverse()
        inputs["features"].reverse()
        second = RANKING.rank(inputs, context=context, config=config, seed=42)
        self.assertEqual({k: v for k, v in first.items() if k != "timings"},
                         {k: v for k, v in second.items() if k != "timings"})
        self.assertEqual(set(first["timings"]), {"profile", "candidates", "hydration", "scoring", "selection", "total"})
        self.assertTrue(all(v >= 0 for v in first["timings"].values()))
        self.assertTrue(all(type(v) is int and v >= 0 for v in first["source_counts"].values()))

    def test_each_vector_source_can_recruit_without_tags_or_visuals(self):
        for channel, source, kind in [("visual", "visual", "video"), ("voice", "voice", "video"),
                                      ("sound", "sound", "video"), ("visual", "images", "image")]:
            with self.subTest(channel=channel, kind=kind):
                inputs, context = raw_fixture(kind=kind, count=2)
                for feature in inputs["features"]:
                    feature["tag_seconds"] = {}
                    feature["vectors"] = {channel + "-v1": [1, 0]}
                    if channel == "visual":
                        feature["vectors"]["semantic-v1"] = [1, 0]
                context["intent"]["seed"] = {"kind": kind, "id": 1}
                if kind == "image":
                    seed, _ = raw_fixture(count=1)
                    seed["features"][0]["vectors"] = {"visual-v1": [1, 0], "semantic-v1": [1, 0]}
                    inputs["catalog"] += seed["catalog"]
                    inputs["features"] += seed["features"]
                    context["intent"]["seed"] = {"kind": "video", "id": 1}
                    context["eligible_ids"] = [{"kind": "image", "id": 2}]
                result = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
                self.assertEqual([(i["kind"], i["id"]) for i in result["items"]], [(kind, 2)])
                self.assertGreater(result["source_counts"][source], 0)

    def test_candidate_arm_can_enable_a_source_disabled_by_base(self):
        for knob, channel, source in [("embedding_weight", "visual", "visual"),
                                      ("taste_audio_weight", "voice", "voice"),
                                      ("taste_mix_weight", "sound", "sound")]:
            inputs, context = raw_fixture(count=2)
            for feature in inputs["features"]:
                feature["tag_seconds"] = {}
                feature["vectors"] = {channel + "-v1": [1, 0]}
                if channel == "visual":
                    feature["vectors"]["semantic-v1"] = [1, 0]
            context["intent"]["seed"] = {"kind": "video", "id": 1}
            result = RANKING.rank(inputs, context=context, config=raw_config(
                **{knob: 0}, experiment={"knob": knob, "candidate": 0.5}), seed=42)
            self.assertEqual([item["id"] for item in result["items"]], [2])
            self.assertGreater(result["source_counts"][source], 0)
            self.assertEqual(result["items"][0]["explanation"]["arm"], "cand")

    def test_no_history_and_negative_only_use_labelled_eligible_fallback(self):
        for negative in [False, True]:
            inputs, context = raw_fixture()
            context["eligible_ids"] = [{"kind": "video", "id": 2}]
            if negative:
                inputs["evidence"] = [dict(event_id="r1", client_event_id="r1", kind="video", id=1,
                                          session_id="earlier", occurred_at="2026-01-01T01:00:00Z",
                                          known_at="2026-01-01T01:00:00Z", type="rating", value=20)]
            result = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
            self.assertEqual([i["id"] for i in result["items"]], [2])
            self.assertEqual(result["items"][0]["explanation"]["fallback"], "no_positive_preference_evidence")

    def test_empty_allowlist_and_verified_duplicate_groups(self):
        inputs, context = raw_fixture()
        for row in inputs["catalog"][:2]:
            row.update(duplicate_verified=True, duplicate_group="same-md5")
        context["intent"]["tag_ids"] = [10]
        result = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        self.assertEqual(len({i["id"] for i in result["items"]} & {1, 2}), 1)
        context["eligible_ids"] = []
        self.assertEqual(RANKING.rank(inputs, context=context, config=raw_config(), seed=42)["items"], [])

    def test_explanations_are_values_used_by_score_and_selector(self):
        inputs, context = raw_fixture()
        context["intent"]["tag_ids"] = [10]
        result = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        for item in result["items"]:
            detail = item["explanation"]
            self.assertAlmostEqual(sum(t["contribution"] for t in detail["tag_contributions"]), detail["tag_score"])
            self.assertAlmostEqual(item["score"], (detail["tag_term"] + detail["embedding_term"])
                                   * detail["history_multiplier"] * detail["affinity_multiplier"])
            trace = detail["selection"]
            self.assertAlmostEqual(trace["value"], trace["relevance_normalized"] - trace["diversity_penalty"] + trace["calibration_bonus"])

    def test_current_empty_projection_does_not_reuse_catalog_tags(self):
        inputs, context = raw_fixture()
        context["intent"]["tag_ids"] = [10]
        inputs["features"][0]["tag_seconds"] = {}
        result = RANKING.rank(inputs, context=context, config=raw_config(), seed=42)
        self.assertNotIn(1, [item["id"] for item in result["items"]])

    def test_configured_secondary_kind_is_ranked_like_the_default_image_kind(self):
        for secondary in ("image", "photo"):
            with self.subTest(secondary=secondary):
                kinds = ("video", secondary)
                page = RANKING.rank_page([], [], config=raw_config(), target_shares={}, seed=1, allowed={secondary: None},
                                         excluded=set(), duplicate_groups={}, fallback=[(secondary, 1)], kinds=kinds)
                self.assertEqual([item["key"] for item in page["items"]], [(secondary, 1)])
                inputs, context = raw_fixture(count=2)
                images, _ = raw_fixture(kind=secondary, count=2)
                for feature in inputs["features"] + images["features"]:
                    feature["vectors"] = {"visual-v1": [1, 0], "semantic-v1": [1, 0]}
                inputs["catalog"] += images["catalog"]
                inputs["features"] += images["features"]
                context.update(kinds=list(kinds), eligible_ids=[{"kind": r["kind"], "id": r["id"]} for r in inputs["catalog"]])
                context["intent"]["seed"] = {"kind": "video", "id": 1}
                result = RANKING.rank(inputs, context=context, config=raw_config(images_share=0.5), seed=42, kinds=kinds)
                self.assertIn((secondary, 1), [(i["kind"], i["id"]) for i in result["items"]])
                self.assertGreater(result["source_counts"]["images"], 0)
                disabled = RANKING.rank(inputs, context=context, config=raw_config(include_images=False), seed=42, kinds=kinds)
                self.assertFalse(any(i["kind"] == secondary for i in disabled["items"]))
                self.assertIn({"kind": secondary, "id": 1, "reason": "images_disabled"}, disabled["exclusions"])

    def test_variants_are_not_aliases_and_config_must_be_resolved(self):
        inputs, context = raw_fixture()
        self.assertEqual(RANKING.SUPPORTED_VARIANTS, ("current", "audit_old", "admission_only"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            RANKING.rank(inputs, context=context, config=raw_config(), seed=1, variant="made_up")
        with self.assertRaisesRegex(ValueError, "unresolved"):
            RANKING.rank(inputs, context=context, config={}, seed=1)
        with self.assertRaisesRegex(ValueError, "raw catalog/features/evidence"):
            RANKING.rank({**inputs, "scores": []}, context=context, config=raw_config(), seed=1)
        with self.assertRaisesRegex(ValueError, "invalid ranking number"):
            RANKING.rank(inputs, context=context, config=raw_config(embedding_weight=float("nan")), seed=1)

class TransientBoundaryContracts(OfflineTestCase):
    def test_reference_parser_ast_requires_only_existing_boundary_helpers(self):
        tree = ast.parse(RANKING_PATH.read_text(encoding="utf-8-sig"))
        ns = {"DEFAULT_KINDS": TASTE.DEFAULT_KINDS}
        names = {"_ids", "_json_object", "parse_context", "context_key"}
        compile_definitions([node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
                            ns, "<isolated parser boundary>")
        sid = "elig:v1:" + "a" * 64
        parsed = ns["parse_context"]({"eligibility": json.dumps({"snapshot_id": sid})}, decode=json.loads,
                                     resolve_eligibility=lambda _: {"video": [1, 2], "image": []})
        self.assertEqual(parsed["eligible_ids"], {"video": (1, 2), "image": ()})
        self.assertEqual(parsed["eligibility_spec"], {"snapshot_id": sid})
        self.assertIn("snapshot_id", repr(ns["context_key"](parsed)))

    def test_reference_parser_keeps_compact_identity_and_full_detached_membership(self):
        sid = "elig:v1:" + "a" * 64
        ids = [1 + i * 100003 for i in range(100000)]
        resolve = Mock(return_value={"video": ids, "image": []})
        parsed = RANKING.parse_context({"eligibility": json.dumps({"snapshot_id": sid})}, resolve_eligibility=resolve)
        self.assertEqual(parsed["eligible_ids"], {"video": tuple(ids), "image": ()})
        self.assertEqual(parsed["eligibility_spec"], {"snapshot_id": sid})
        self.assertLess(len(repr(RANKING.context_key(parsed))), 500)
        resolve.assert_called_once_with(sid)
        ids.clear()
        self.assertEqual(len(parsed["eligible_ids"]["video"]), 100000)
        inline = RANKING.parse_context({"eligibility": '{"video_ids":[2,1,2]}'})
        self.assertEqual(inline["eligibility_spec"], {"video_ids": [1, 2], "image_ids": None})
        for config, resolver in [({"snapshot_id": sid}, None), ({"snapshot_id": sid, "video_ids": []}, resolve),
                                  ({"snapshot_id": "bad"}, resolve)]:
            with self.assertRaises(ValueError):
                RANKING.parse_context({"eligibility": json.dumps(config)}, resolve_eligibility=resolver)
        for member in ({"video": None, "image": []}, {"video": [True], "image": []},
                       {"video": [1, 1], "image": []}, {"video": [2**63], "image": []}, {"video": []}):
            with self.assertRaises(ValueError):
                RANKING.parse_context({"eligibility": json.dumps({"snapshot_id": sid})}, resolve_eligibility=lambda _: member)
        with self.assertRaises(ValueError):
            RANKING.parse_context({"eligibility": json.dumps({"video_ids": list(range(1, 2050))})})

    def test_seed_query_uses_available_features_without_mutation(self):
        matrix = np.array([[0.0, 1.0]], dtype=np.float16)
        query = np.array([1.0, 0.0], dtype=np.float32)
        combined = RANKING.seed_query(query, matrix, {5: 0}, (5, 6))
        np.testing.assert_allclose(combined, np.array([1, 1]) / np.sqrt(2), rtol=1e-6)
        np.testing.assert_array_equal(query, [1, 0])
        np.testing.assert_array_equal(matrix, [[0, 1]])
        np.testing.assert_array_equal(RANKING.seed_query(None, matrix, {5: 0}, (5,)), [0, 1])
        self.assertIsNone(RANKING.seed_query(None, None, {}, (5,)))

    def test_untrusted_serialization_is_strict_and_redacted(self):
        for cfg in [
            {"intent": {"tag_ids": [1]}}, {"intent": '[1]'},
            {"intent": '{"tag_ids":[true]}'}, {"intent": '{"tag_ids":[1.0]}'},
            {"intent": '{"tag_ids":[-1]}'}, {"intent": '{"tag_ids":[1],"tag_ids":[2]}'},
            {"intent": '{"query":"private-text"}'}, {"eligibility": '{"video_ids":"all"}'},
            {"session_id": "private-text\n"}, {"request_id": {"nested": "private-text"}},
        ]:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError) as error:
                RANKING.parse_context(cfg)
            self.assertNotIn("private-text", str(error.exception))

    def test_content_key_excludes_delivery_ids_but_preserves_context(self):
        first = RANKING.parse_context({"intent": '{"tag_ids":[2,1]}', "session_id": "s1"})
        same = RANKING.parse_context({"intent": '{"tag_ids":[1,2,1]}', "request_id": "r2"})
        self.assertEqual(RANKING.context_key(first), RANKING.context_key(same))
        empty = RANKING.parse_context({"intent": '{"tag_ids":[1,2]}', "eligibility": '{"video_ids":[]}'})
        self.assertNotEqual(RANKING.context_key(first), RANKING.context_key(empty))
        self.assertEqual(first["eligible_ids"], {"video": None, "image": None})



class ModuleNameContracts(OfflineTestCase):
    """Every name a shipped module LOADS must be defined or imported in that module.

    2026-09-13: a deploy preflight failed with a bare NameError because a module
    called a helper the test harness had injected. This test resolves names the
    way the interpreter will, with no harness namespace.
    """

    def test_no_free_names_in_shipped_modules(self):
        for module in SHIPPED_MODULES:
            tree = ast.parse(module.read_text())
            defined = set(dir(builtins)) | {'__name__', '__file__', '__doc__', '__spec__', '__package__'}
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    defined |= {(a.asname or a.name).split('.')[0] for a in node.names}
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    defined |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                elif isinstance(node, (ast.If, ast.Try)):
                    for inner in ast.walk(node):
                        if isinstance(inner, (ast.Import, ast.ImportFrom)):
                            defined |= {(a.asname or a.name).split('.')[0] for a in inner.names}
                        elif isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            defined.add(inner.name)
                        elif isinstance(inner, ast.Assign):
                            defined |= {n.id for t in inner.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            free = set()
            # Top-level functions only: a nested def or lambda shares its enclosing
            # scope, so it is walked as part of its parent, never on its own.
            top = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            top += [m for n in tree.body if isinstance(n, ast.ClassDef) for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for fn in top:
                local = set()
                for n in ast.walk(fn):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                        local.add(n.id)
                    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                        args = n.args
                        local |= {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
                        local |= {a.arg for a in (args.vararg, args.kwarg) if a}
                        if not isinstance(n, ast.Lambda):
                            local.add(n.name)
                    elif isinstance(n, ast.ExceptHandler) and n.name:
                        local.add(n.name)
                    elif isinstance(n, (ast.Import, ast.ImportFrom)):
                        local |= {(a.asname or a.name).split('.')[0] for a in n.names}
                    elif isinstance(n, ast.ClassDef):
                        local.add(n.name)
                for n in ast.walk(fn):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined and n.id not in local:
                        free.add((fn.name, n.id))
            self.assertEqual(sorted(free), [], str(module))


class CandidateContracts(OfflineTestCase):
    def test_shared_admission_keeps_independent_budgets_and_exact_duplicate_policy(self):
        tags = [("video", sid) for sid in range(1, 604)]
        vectors = [("video", 800), ("video", 801), ("video", 802)]
        sources, union = RANKING.admit_sources(
            {"tags": tags, "visual": vectors}, {"tags": 600, "visual": 2}, {"video": None},
            {("video", 1), ("video", 800)}, {("video", 2): "same-md5", ("video", 801): "same-md5"})
        self.assertEqual(len(sources["tags"]), 600)
        self.assertEqual(sources["visual"], [("video", 801), ("video", 802)])
        self.assertIn(("video", 802), union)
        self.assertNotIn(("video", 801), union)
        self.assertEqual(len(union), 601)

    def test_embedding_ties_use_ids_not_matrix_row_order(self):
        for ids in [[12, 2, 10, 5, 1], [1, 5, 10, 2, 12]]:
            matrix = np.array([[0.9 if i == 12 else 0.5] for i in ids], dtype=np.float16)
            self.assertEqual(RANKING.embedding_candidates(np.array(ids), matrix, np.array([1.0]), exclude={2}, limit=3), [12, 1, 5])

    def test_candidate_dot_uses_shared_chunked_boundary(self):
        matrix = object()
        query = np.array([1.0])
        with patch.object(RANKING_MODULE, "chunked_dot", return_value=np.array([0.5, 0.5])) as dot:
            self.assertEqual(RANKING.embedding_candidates(np.array([2, 1]), matrix, query, exclude=set(), limit=1), [1])
            dot.assert_called_once_with(matrix, query)

    def test_embedding_allowlist_applies_before_its_source_budget(self):
        ids, matrix = np.array([1, 2, 3]), np.ones((3, 1))
        self.assertEqual(RANKING.embedding_candidates(ids, matrix, np.array([1.0]), exclude=set(), limit=1, eligible_ids=(2, 3)), [2])
        self.assertEqual(RANKING.embedding_candidates(ids, matrix, np.array([1.0]), exclude=set(), limit=1, eligible_ids=()), [])


class ImageContracts(OfflineTestCase):
    def run_lane(self, sims, multipliers):
        # Lane rows as rank_page consumes them: (key, base score, fatigue penalty, affinity delta);
        # the delta reproduces each multiplier through affinity_scale at weight 0.5.
        rows = [(("image", i), sims[i], 1.0, (multipliers[i] - 1) / 0.5) for i in range(len(sims))]
        result = RANKING.rank_page([], rows, config=raw_config(images_share=1), target_shares={}, seed=1,
                                   allowed={}, excluded=set(), duplicate_groups={}, limit=1)
        return [{"id": row['key'][1], "score": row['score']} for row in result['items']]

    def test_affinity_reranks_before_lane_cut(self):
        self.assertEqual(self.run_lane([0.9, 0.85], [0.85, 1.15])[0]["id"], 1)


class ScoringPreservationContracts(OfflineTestCase):
    def test_negative_tags_caps_length_and_bodypart_damping(self):
        kw = dict(bodyparts_weight=0.3, max_tag_share=0.35, length_floor=120)
        score = RANKING.relevance({1: 600, 2: 600}, {1: 1, 2: -1}, {2: "bodyparts"}, 600, **kw)
        self.assertAlmostEqual(score, (210 - 63) / 600)
        self.assertAlmostEqual(RANKING.relevance({1: 30}, {1: 1}, {}, 30, **kw), 10.5 / 120)

    def test_embedding_normalization_stays_max_anchored(self):
        comps = [(1, {}, "acts", 0.5, 0.8, None, None, 1.0, None),
                 (2, {}, "acts", 0.0, 0.4, None, None, 1.0, None)]
        knobs = dict(embedding_weight=0.5, contributor_affinity_weight=0, taste_audio_weight=0, taste_mix_weight=0)
        self.assertEqual([s[0] for s in RANKING.score_components(comps, knobs)], [0.75, 0.25])


class SelectionContracts(OfflineTestCase):
    def test_seeded_equivalence_and_ties_both_selectors(self):
        def pick_sim(scored, *, pool_size, diversity, details):
            # The similar-items selection block: light MMR, no calibration, over the whole pool.
            want = max(pool_size, len(scored))
            return RANKING.select(scored, want=want, diversity=diversity, calibration=0, target_shares={}, details=details)
        for seed in range(250):
            scored = fixture(seed)
            for want in [0, 1, 20, 45]:
                kw = dict(want=want, diversity=0.35, calibration=0.25,
                          target_shares={"acts": 0.7, "bodyparts": 0.1})
                self.assertEqual(RANKING.select(scored, **kw), old_select(scored, cosine=RANKING.cosine, **kw))
                similar = pick_sim([(*s[:3], "other") for s in scored], pool_size=len(scored), diversity=0.35, details={})
                kw.update(want=want or 20, calibration=0, target_shares={})
                self.assertEqual(similar[:want or 20], old_select(scored, cosine=RANKING.cosine, **kw))
        tied = [(1.0, i, {1: 1.0}, "acts") for i in [7, 3, 9, 1]]
        self.assertEqual(RANKING.select(tied, want=4, diversity=1, calibration=0, target_shares={}), [7, 3, 9, 1])
        self.assertEqual(RANKING.select([], want=20, diversity=1, calibration=1, target_shares={}), [])

    def test_pairwise_work_is_incremental(self):
        cosine = Mock(wraps=RANKING.cosine)
        with patch.object(RANKING_MODULE, "cosine", cosine):
            RANKING.select(fixture(5, 60), want=20, diversity=0.35, calibration=0.25, target_shares={})
        self.assertLessEqual(cosine.call_count, 60 * 20)


if __name__ == "__main__":
    unittest.main()

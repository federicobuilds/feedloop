"""Profiles over slot facts: watch rows from real visit facts, coverage selection,
decayed tag profiles, the secondary-event weighting decision, contributor affinity
from trusted links, and the shared trial reward."""
import numpy as np
import pytest

from feedloop import profiles

K = lambda i, kind="video": (kind, i)


def test_watch_rows_derive_days_and_visits_from_real_facts_only():
    rows = {K(1): {"rating": None, "engagement_count": 0,
                   "watch": {"watched_s": 300.0, "last_at": 1000.0 - 2 * 86400.0, "visit_days": [3, 3, 5], "intervals": []}},
            K(2): {"rating": 80, "engagement_count": 0},
            K(3): {"rating": None, "engagement_count": 0, "watch": {"watched_s": 30.0, "last_at": 1000.0, "visit_days": []}}}
    watch = profiles.watch_rows(rows, now=1000.0)
    assert watch == {K(1): {"watched_s": 300.0, "days": 2.0, "visits": 2.0}, K(3): {"watched_s": 30.0, "days": 0.0, "visits": 1.0}}
    assert K(2) not in watch, "an explicit-only row never becomes a fabricated zero-second visit"


def test_coverage_prefers_watched_for_watched_items_and_full_for_explicit_only():
    features = {K(1): {"tag_seconds": {10: 100.0, 20: 50.0}, "watched_tag_seconds": {10: 30.0}},
                K(2): {"tag_seconds": {10: 100.0}, "watched_tag_seconds": None},
                K(3): {"tag_seconds": None, "watched_tag_seconds": {20: 12.0}},
                K(4): {"tag_seconds": {10: 0.0, 20: float("nan"), 30: -5.0}}}
    watched = profiles.coverage_for(features, [K(1), K(2), K(3), K(4), K(9)])
    assert watched == {K(1): {10: 30.0}, K(2): {10: 100.0}, K(3): {20: 12.0}}
    explicit = profiles.coverage_for(features, [K(1), K(3)], prefer_full=True)
    assert explicit == {K(1): {10: 100.0, 20: 50.0}, K(3): {20: 12.0}}


def test_build_profiles_decays_repeats_and_admits_explicit_items_at_sixty_seconds():
    watch = {K(1): {"watched_s": 600.0, "days": 0.0, "visits": 1.0},      # liked, fresh
             K(2): {"watched_s": 600.0, "days": 21.0, "visits": 3.0},     # liked, one half-life old, three visits
             K(3): {"watched_s": 90.0, "days": 0.0, "visits": 1.0}}       # 90 s of two hours: dislike
    durations = {K(1): 600.0, K(2): 600.0, K(3): 7200.0}
    features = {K(1): {"tag_seconds": {10: 600.0}, "watched_tag_seconds": {10: 300.0}},
                K(2): {"tag_seconds": {20: 600.0}, "watched_tag_seconds": {20: 300.0}},
                K(3): {"tag_seconds": {30: 3600.0}, "watched_tag_seconds": {30: 90.0}},
                K(4): {"tag_seconds": {10: 300.0, 40: 100.0}, "watched_tag_seconds": None}}
    liked, disliked, meta = profiles.build_profiles(watch, durations, features, half_life=21.0, min_watch=20.0, finished_ratio=.45,
                                                    abandon_ratio=.15, history_limit=600, ratings={K(4): 90}, rating_strength=1.0)
    assert liked[10] == pytest.approx(300.0 + 60.0 * 0.75 * 2.0), "fresh watched coverage plus the explicit item's 60 s share, boosted"
    assert liked[40] == pytest.approx(60.0 * 0.25 * 2.0)
    assert liked[20] == pytest.approx(300.0 * 0.5 * (1.0 + 0.25 * 2)), "one half-life decay, repeat weight for two extra visits"
    assert disliked == {30: pytest.approx(90.0)}
    assert (meta["liked_items"], meta["disliked_items"], meta["items_considered"]) == (3, 1, 4)
    assert meta["liked_watch_s"] == 1200.0 and meta["abandoned_watch_s"] == 90.0
    # history_limit bounds the watched items by recency while explicit items always survive
    _, _, bounded = profiles.build_profiles(watch, durations, features, half_life=21.0, min_watch=20.0, finished_ratio=.45,
                                            abandon_ratio=.15, history_limit=1, ratings={K(4): 90}, rating_strength=1.0)
    assert bounded["items_considered"] == 2


def test_secondary_event_extras_distribute_sixty_seconds_by_coverage_share():
    # 2026-09-16 orchestrator decision: 60 pseudo-seconds per verdicted secondary item,
    # split across its tags by coverage share (the ranking contract), not 60 per tag.
    events = {K(1, "image"): {"rating": 90, "engagement_count": 0},   # like, boost 2
              K(2, "image"): {"rating": 20, "engagement_count": 0},   # dislike, boost 1.5
              K(3, "image"): {"rating": 60, "engagement_count": 0},   # neutral: no event
              K(4, "image"): {"rating": None, "engagement_count": 1}} # engagement: like
    coverage = {K(1, "image"): {10: 3.0, 20: 1.0}, K(2, "image"): {30: 5.0}, K(3, "image"): {10: 1.0}, K(4, "image"): {}}
    means = (np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), {K(1, "image"): 0, K(4, "image"): 1})
    tags, vectors, count = profiles.secondary_event_extras(events, coverage, means)
    assert count == 3
    assert tags == [({10: 45.0, 20: 15.0}, 2.0, True), ({30: 60.0}, 1.5, False)]
    assert all(sum(t.values()) == pytest.approx(profiles.SECONDARY_EVENT_TAG_SECONDS) for t, _b, _l in tags)
    assert [(w, like) for _v, w, like in vectors] == [(600.0 * 2.0, True), (600.0 * 1.4, True)]
    np.testing.assert_array_equal(vectors[1][0], np.array([0.0, 1.0], dtype=np.float32))
    liked, disliked, meta = profiles.build_profiles({}, {}, {}, half_life=21.0, min_watch=20.0, finished_ratio=.45, abandon_ratio=.15,
                                                    history_limit=600, extra_tag_events=tags)
    assert liked == {10: 90.0, 20: 30.0} and disliked == {30: 90.0} and meta["secondary_events"] == 2


def test_rocchio_uses_watch_time_weights_and_beta_half_negative():
    m = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float16)
    index = {K(1): 0, K(2): 1, K(3): 2}
    watch = {K(1): {"watched_s": 3600.0, "days": 0.0, "visits": 1.0}, K(2): {"watched_s": 7200.0, "days": 0.0, "visits": 1.0},
             K(3): {"watched_s": 90.0, "days": 0.0, "visits": 1.0}}
    durations = {K(1): 3600.0, K(2): 7200.0, K(3): 7200.0}
    q = profiles.rocchio_over(m, index, watch, durations, half_life=21.0, finished_ratio=.45, abandon_ratio=.15, history_limit=600,
                              ratings={}, rating_strength=1.0)
    # both likes weigh the 3,600 s cap equally; the dislike pulls beta=.5 along -x
    expected = np.array([1.0, 1.0]) / np.sqrt(2) - 0.5 * np.array([-1.0, 0.0])
    np.testing.assert_allclose(q, expected / np.linalg.norm(expected), atol=1e-6)
    assert profiles.rocchio_over(m, index, {}, {}, half_life=21.0, finished_ratio=.45, abandon_ratio=.15, history_limit=600,
                                 ratings={}, rating_strength=1.0) is None
    assert profiles.rocchio_over(None, None, {}, {}, half_life=21.0, finished_ratio=.45, abandon_ratio=.15, history_limit=600,
                                 ratings={}, rating_strength=1.0, extras=[]) is None


def test_contributor_affinity_counts_secondary_items_as_one_unit_and_caps_primary_at_an_hour():
    links = {K(1): ["X"], K(2, "image"): ["X"], K(3, "image"): ["Y"], K(4): ["Z"]}
    watch = {K(1): {"watched_s": 7200.0}}
    aff, prior = profiles.contributor_affinity(links, liked_primary={K(1)}, seen_primary={K(1), K(4)},
                                               liked_secondary={K(2, "image")}, seen_secondary={K(2, "image"), K(3, "image")}, watch=watch)
    assert aff["X"] == {"affinity": pytest.approx((7.0 + 5 * prior) / (7.0 + 5)), "exposures": 7.0, "likes": 7.0, "primary": 1, "secondary": 1}
    assert aff["Y"]["exposures"] == 1.0 and aff["Y"]["likes"] == 0.0 and aff["Y"]["secondary"] == 1
    assert "Z" not in aff, "a primary item with no watch time is not an exposure"
    assert prior == pytest.approx(7.0 / 8.0)
    assert profiles.contributor_affinity({}, set(), set()) == ({}, 0.0)


def test_constants_carry_the_source_values():
    assert (profiles.LIKE_ABS_MIN_S, profiles.LIKE_ABS_MAX_S, profiles.LIKE_ABS_DURATION_SHARE) == (120.0, 240.0, 0.05)
    assert (profiles.D_DISLIKE_MIN_WATCH, profiles.SHORT_WATCH_ABS_MIN_S, profiles.D_FINISHED_RATIO, profiles.D_ABANDON_RATIO) == (60.0, 5.0, .45, .15)
    assert (profiles.ROCCHIO_BETA, profiles.SECONDARY_EVENT_VEC_WEIGHT, profiles.SECONDARY_EVENT_TAG_SECONDS) == (.5, 600.0, 60.0)
    assert (profiles.AFFINITY_SHRINK, profiles.AFFINITY_CAP, profiles.AFFINITY_UNIT_S) == (5.0, .15, 600.0)
    assert (profiles.TRIAL_FULL_S, profiles.TRIAL_RATING_FLOOR) == (3600.0, 600.0 / 3600.0)
    from feedloop import taste
    assert profiles.verdict is taste.verdict and profiles.item_preferences is taste.item_preferences, "one production verdict"

"""The verdict rules, each pinned to the decision that set it."""
import numpy as np

from feedloop import profiles


def test_like_bar_scales_and_clamps():
    assert profiles.like_bar(0) == 120.0            # unknown duration: the floor
    assert profiles.like_bar(600) == 120.0          # 5% of 10 min = 30s, clamped up
    assert profiles.like_bar(3600) == 180.0         # 5% of an hour
    assert profiles.like_bar(20000) == 240.0        # long file: clamped at 4 min


def _v(watched, duration, **kw):
    return profiles.verdict(watched, duration, finished_ratio=0.45, abandon_ratio=0.15, **kw)


def test_short_clip_likes_by_completion():
    assert _v(20, 40)[0] is True                # 50% of a 40s clip


def test_long_file_likes_at_the_bar_not_before():
    assert _v(200, 20000)[0] is False           # 200s < 240s bar, 1% completion
    assert _v(240, 20000)[0] is True


def test_dislike_window_runs_to_the_bar():
    # 2h file: bar is 240s, so a 200s sample under 15% completion is a dislike
    is_like, is_dis, _ = _v(200, 7200)
    assert (is_like, is_dis) == (False, True)
    # under the 60s floor it is a peek, never a dislike
    assert _v(50, 7200)[1] is False
    # at or past the bar it becomes a like, never a dislike
    assert _v(240, 7200)[0] is True


def test_three_stars_force_neutral():
    is_like, is_dis, boost = _v(3600, 3600, rating=60)
    assert (is_like, is_dis, boost) == (False, False, 1.0)


def test_high_rating_forces_like_even_when_barely_watched():
    is_like, is_dis, boost = _v(5, 3600, rating=90)
    assert (is_like, is_dis) == (True, False) and boost > 1.0


def test_low_rating_forces_dislike_even_when_finished():
    is_like, is_dis, _ = _v(3600, 3600, rating=20)
    assert (is_like, is_dis) == (False, True)


def test_engagement_outranks_everything():
    is_like, is_dis, boost = _v(0, 3600, rating=20, engagement_count=2)
    assert (is_like, is_dis) == (True, False)
    # rating 20 gives a 1.5 dislike boost, then the engagement multiplies by 1.8
    assert abs(boost - 1.5 * 1.8) < 1e-9


def test_engagement_compounding_is_capped_at_five():
    _, _, b5 = _v(0, 100, engagement_count=5)
    _, _, b9 = _v(0, 100, engagement_count=9)
    assert b5 == b9


def test_secondary_items_verdict_from_rating_and_engagement_only():
    assert _v(0, 0)[:2] == (False, False)
    assert _v(0, 0, rating=80)[:2] == (True, False)
    assert _v(0, 0, rating=40)[:2] == (False, True)
    assert _v(0, 0, engagement_count=1)[:2] == (True, False)


def test_watch_counts_gate():
    assert profiles.watch_counts({"watched_s": 25}, 3600, 20, 0.5) is True
    assert profiles.watch_counts({"watched_s": 10}, 3600, 20, 0.5) is False
    # short-clip rescue: 15s of a 20s clip counts (>= 5s and >= 50%)
    assert profiles.watch_counts({"watched_s": 15}, 20, 20, 0.5) is True
    assert profiles.watch_counts({"watched_s": 4}, 6, 20, 0.5) is False


def test_unwatched_explicit_preferences_are_admitted_without_inventing_watch():
    watch = {1: {"watched_s": 1, "days": 0, "visits": 1}}
    facts = profiles.item_preferences(watch, {}, ratings={2: 0, 3: 60, 4: 80}, engagement_counts={3: 1})
    assert not facts[1]["admitted"]
    assert facts[2]["is_dislike"]
    assert facts[3]["is_like"] and not facts[3]["is_dislike"]
    assert facts[4]["is_like"]
    assert set(watch) == {1}


def test_secondary_only_profile_and_explicit_primary_survive_empty_watch_history():
    options = dict(half_life=21, finished_ratio=.45, abandon_ratio=.15, history_limit=0, rating_strength=1)
    vector = np.array([1, 0], dtype=np.float32)
    secondary = profiles.rocchio_over(None, None, {}, {}, ratings={}, extras=[(vector, 180, True)], **options)
    primary = profiles.rocchio_over(np.array([[1, 0]], dtype=np.float16), {7: 0}, {}, {}, ratings={7: 80}, **options)
    np.testing.assert_array_equal(secondary, vector)
    np.testing.assert_array_equal(primary, vector)


def test_reset_fences_inflight_feature_cache_publication():
    cache = profiles.TTLCache("regression", 60)
    def build():
        cache.clear()
        return "finished-old-build"
    assert profiles.feature_cached(cache, lambda: (("visual", 1),), build) == "finished-old-build"
    assert cache.value is None


def test_bounded_scoring_matches_full_precision():
    matrix = np.arange(15003, dtype=np.float32).reshape(-1, 3).astype(np.float16)
    vector = np.array([.1, -.2, .3], dtype=np.float32)
    np.testing.assert_allclose(profiles.chunked_dot(matrix, vector), matrix.astype(np.float32) @ vector, atol=.001)

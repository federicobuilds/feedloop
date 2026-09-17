import json

import feedloop.taste as tm


def test_place_explore_lands_inside_the_requested_page():
    core = list(range(100, 130))                      # limit 10 + offset 20 = 30 ranked
    out = tm.place_explore(core, [901, 902], offset=20, limit=10)
    page = out[20:30]
    assert page[-2:] == [901, 902]                    # explore items on THIS page
    assert out[:20] == core[:20]                      # earlier pages untouched
    assert len(out) == 30


def test_place_explore_first_page_regression():
    # the bug: tail replacement put explore items past the window on page 1
    core = list(range(20))                            # limit 10, offset 0, core is 20 long
    out = tm.place_explore(core, [777, 888], offset=0, limit=10)
    assert 777 in out[:10] and 888 in out[:10]


def test_place_explore_noop_without_explore():
    assert tm.place_explore([1, 2, 3], [], 0, 10) == [1, 2, 3]


def test_team_draft_alternates_and_dedupes():
    out, arm = tm.team_draft([1, 2, 3, 4], [3, 5, 6], want=6, first_a=True)
    assert out[0] == 1 and arm[1] == "base"
    assert out[1] == 3 and arm[3] == "cand"           # cand's first pick
    assert 3 not in out[2:]                           # never drafted twice
    assert len(out) == len(set(out)) == 6


def test_shannon_entropy():
    assert tm.shannon_entropy({"a": 10}) == 0.0
    assert abs(tm.shannon_entropy({"a": 1, "b": 1}) - 0.6931) < 1e-3


def test_cooldown_tiers():
    kw = dict(cooldown_days=45.0, recovery_days=120.0)
    assert tm.cooldown_multiplier(7.4, judgeable=False, **kw) == 0.0        # peek still resting
    assert tm.cooldown_multiplier(7.6, judgeable=False, **kw) > 0.0         # peek back after 7.5d
    assert tm.cooldown_multiplier(22.4, **kw) == 0.0                        # consumed: half
    assert tm.cooldown_multiplier(22.6, **kw) > 0.0
    assert tm.cooldown_multiplier(44.9, is_dislike=True, **kw) == 0.0       # disliked: full
    assert tm.cooldown_multiplier(45.0 + 120.0, is_dislike=True, **kw) == 1.0


def test_welch_interval():
    # identical arms: diff 0; distinct arms: diff positive and significant
    d, hw = tm.welch_interval(60, 30.0, 30.0, 60, 30.0, 30.0, 1.645)
    assert d == 0.0 and hw > 0
    d, hw = tm.welch_interval(60, 6.0, 0.6, 60, 30.0, 15.0, 1.645)
    assert d > hw


def test_tagging_pending_parses_the_real_status_shape():
    st = json.loads('{"ts":"x","counts":{"applied":140,"failed":3,"extracting":2,"uploaded":5},"eta_hours":1.2}')
    assert tm.tagging_pending(st) == 7
    assert tm.tagging_pending({"counts": {"applied": 9}}) == 0
    assert tm.tagging_pending({}) == 0


def test_embed_pending_parses_the_real_status_shape():
    st = json.loads('{"ts":"x","applied":4386,"pending":12,"extracting":1,"failed":9}')
    assert tm.embed_pending(st) == 13
    assert tm.embed_pending({"ts": "x", "applied": 1}) == 0

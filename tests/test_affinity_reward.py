from feedloop import profiles


def test_trial_reward_watch_seconds_currency():
    assert profiles.trial_reward(3600) == 1.0
    assert profiles.trial_reward(7200) == 1.0                 # capped
    assert abs(profiles.trial_reward(15) - 15 / 3600) < 1e-9  # a cheap like is ~0.004
    assert profiles.trial_reward(0, engagement_count=1) == 1.0  # engagement is a full point
    assert abs(profiles.trial_reward(0, rating=90) - profiles.TRIAL_RATING_FLOOR) < 1e-9


def test_affinity_multiplier_is_capped_both_ways():
    aff = {"a": {"affinity": 1.0}, "b": {"affinity": 0.0}}
    assert profiles.affinity_multiplier(["a"], aff, prior=0.5, weight=10.0) == 1.0 + profiles.AFFINITY_CAP
    assert profiles.affinity_multiplier(["b"], aff, prior=0.5, weight=10.0) == 1.0 - profiles.AFFINITY_CAP
    assert profiles.affinity_multiplier([], aff, 0.5, 1.0) == 1.0
    assert profiles.affinity_multiplier(["zzz"], aff, 0.5, 1.0) == 1.0
    assert profiles.affinity_multiplier(["a"], aff, 0.5, 0.0) == 1.0


def test_contributor_affinity_shrinks_toward_prior():
    # two items for contributor X: one liked for an hour, one skipped for 30s
    k = lambda i: ("video", i)
    links = {k(1): ["X"], k(2): ["X"], k(3): ["Y"]}
    watch = {k(1): {"watched_s": 3600}, k(2): {"watched_s": 30}, k(3): {"watched_s": 600}}
    aff, prior = profiles.contributor_affinity(links, liked_primary={k(1)}, seen_primary={k(1), k(2), k(3)}, watch=watch)
    # X: likes 6 units (3600/600), exposures 6.05; Y: 0 likes over 1 unit
    assert aff["X"]["likes"] == 6.0 and abs(aff["X"]["exposures"] - 6.05) < 1e-6
    assert 0 < aff["Y"]["affinity"] < prior + 1e-9      # shrunk up toward the prior from 0
    assert aff["X"]["affinity"] < 1.0                   # shrunk down from ~0.99
    assert aff["X"]["primary"] == 2 and aff["Y"]["primary"] == 1


def test_thirty_second_clip_is_a_twentieth_of_a_unit():
    links = {("video", 1): ["X"]}
    aff, _ = profiles.contributor_affinity(links, {("video", 1)}, {("video", 1)}, watch={("video", 1): {"watched_s": 30}})
    assert aff["X"]["exposures"] == 0.05

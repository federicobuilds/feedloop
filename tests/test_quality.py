"""Recommendation quality over a clustered synthetic library: the native Feed follows likes
without collapsing onto one cluster, lets a dislike push its neighbours down, never repeats an
item on a page, and lets recent taste outweigh old taste."""
import random

import numpy as np
import pytest

from feedloop import profiles
from feedloop.engine import Engine, initialize_stores
from fl2_helpers import MemoryCatalog, MemorySignals, MemorySpaces, catalog_row
from test_engine import clock  # noqa: F401  (clock is a fixture)

CLUSTERS, PER_CLUSTER, DAY = 4, 25, 86400.0
SHARED_TAG = 90
NOW = 100.0 * DAY
REQUEST = {"limit": 24, "images": False, "surface": "feed", "session_id": "q", "request_id": "q1", "client_request_id": "q1"}


def cluster_of(item_id):
    return (item_id - 1) // PER_CLUSTER


def library():
    rows, features = [], {}
    for item_id in range(1, CLUSTERS * PER_CLUSTER + 1):
        c = cluster_of(item_id)
        # one tag of the item's own cluster and one tag every cluster shares, as real libraries overlap
        tags = [10 * c + 1 + item_id % 3, SHARED_TAG + item_id % 4]
        rows.append(catalog_row("video", item_id, duration=600.0, tags=[f"tag {t}" for t in tags]))
        features[("video", item_id)] = {"tag_seconds": {tags[0]: 300.0, tags[1]: 200.0}, "watched_tag_seconds": None,
                                        "tag_categories": {t: "acts" for t in tags}}
    catalog = MemoryCatalog(rows, features, {t: f"tag {t}" for t in [*range(SHARED_TAG, SHARED_TAG + 4), *(10 * c + k for c in range(CLUSTERS) for k in (1, 2, 3))]})
    rng = np.random.default_rng(3)
    # overlapping clusters: every centroid is positive, so every item is somewhat similar to every like
    centroids = np.abs(rng.normal(size=(CLUSTERS, 16)))
    keys = sorted(catalog.rows)
    m = np.stack([centroids[cluster_of(i)] + 0.3 * rng.normal(size=16) for _kind, i in keys]).astype(np.float32)
    return catalog, MemorySpaces({"visual": (keys, m)})


def watched(days_ago, seconds=600.0):
    return {"rating": None, "engagement_count": 0,
            "watch": {"watched_s": seconds, "last_at": NOW - days_ago * DAY, "visit_days": [0], "intervals": [(0, seconds)]}}


def page(tmp_path, now, fake, signal_rows, limit=24, **config):
    catalog, spaces = library()
    tmp_path.mkdir()
    ledger_path, tuner_path = str(tmp_path / "events.sqlite"), str(tmp_path / "tuner.sqlite")
    now[0] = NOW - 10.0
    initialize_stores(ledger_path=ledger_path, tuner_path=tuner_path, cutover_ts=NOW - 20.0, clock=fake)
    eng = Engine(catalog=catalog, signals=MemorySignals(signal_rows, observed_at=NOW - 5.0), spaces=spaces,
                 ledger_path=ledger_path, tuner_path=tuner_path, clock=fake, automatic_tuning=False,
                 space_roles={"visual": "visual", "semantic": "semvisual", "voice": "audioembed", "sound": "audiomix"},
                 config={"control_rate": 0.0, "cooldown_days": 0.0, **config})
    now[0] = NOW
    state = random.getstate()
    try:
        random.seed(11)
        result = eng.feed({**REQUEST, "limit": limit}, record_delivery=False)
    finally:
        random.setstate(state)
    assert result["status"] == "ok", result
    return [item["id"] for item in result["items"]]


def likes(ids, days_ago=1.0):
    return {("video", i): watched(days_ago) for i in ids}


def position(ids, wanted):
    """Mean page position of the wanted items; an item off the page counts as the page length."""
    return float(np.mean([ids.index(i) if i in ids else len(ids) for i in wanted]))


def test_likes_lead_the_page_without_collapsing_onto_one_cluster(tmp_path, clock):
    now, fake = clock
    ids = page(tmp_path / "a", now, fake, likes(range(1, 6)))
    top = [cluster_of(i) for i in ids[:10]]
    print("clusters", [cluster_of(i) for i in ids])
    assert top.count(0) >= 6, top
    assert any(cluster_of(i) != 0 for i in ids), ids


def test_a_dislike_pushes_its_neighbours_down(tmp_path, clock):
    now, fake = clock
    # two cluster-B likes put B on the page, then one B dislike must pull the rest of B down
    b = PER_CLUSTER + 1
    rows = likes([1, 2, 3, 4, 5, b, b + 1])
    neighbours = range(b + 3, 2 * PER_CLUSTER + 1)
    before = page(tmp_path / "a", now, fake, rows, limit=60)
    after = page(tmp_path / "b", now, fake, {**rows, ("video", b + 2): {"rating": 20, "engagement_count": 0}}, limit=60)
    print("before", position(before, neighbours), "after", position(after, neighbours))
    assert b + 2 not in after
    assert position(after, neighbours) > position(before, neighbours), (before, after)


def test_a_dislike_re_rated_positively_returns_to_the_page(tmp_path, clock):
    now, fake = clock
    b = PER_CLUSTER + 1
    rows = likes([1, 2, 3, 4, 5, b, b + 1])
    disliked = page(tmp_path / "a", now, fake, {**rows, ("video", b + 2): {"rating": 20, "engagement_count": 0}}, limit=60)
    assert b + 2 not in disliked
    re_rated = page(tmp_path / "b", now, fake, {**rows, ("video", b + 2): {"rating": 80, "engagement_count": 0}}, limit=60)
    assert b + 2 in re_rated, re_rated


def test_the_page_never_repeats_an_item(tmp_path, clock):
    now, fake = clock
    ids = page(tmp_path / "a", now, fake, {**likes(range(1, 6)), **likes(range(51, 54), days_ago=40.0)})
    assert len(ids) == len(set(ids)) and len(ids) > 0, ids


def test_recent_likes_outweigh_old_likes(tmp_path, clock):
    now, fake = clock
    old, recent = range(1, 6), range(2 * PER_CLUSTER + 1, 2 * PER_CLUSTER + 6)
    ids = page(tmp_path / "a", now, fake, {**likes(old, days_ago=90.0), **likes(recent, days_ago=1.0)})
    top = [cluster_of(i) for i in ids[:10]]
    print("clusters", [cluster_of(i) for i in ids])
    assert top.count(2) > top.count(0), top


def test_a_full_watch_teaches_more_than_an_equal_length_partial_watch():
    m = np.eye(2, dtype=np.float32)
    watch = {1: {"watched_s": 300.0, "days": 1.0, "visits": 1.0}, 2: {"watched_s": 300.0, "days": 1.0, "visits": 1.0}}
    q = profiles.rocchio_over(m, {1: 0, 2: 1}, watch, {1: 300.0, 2: 3000.0}, half_life=21.0, finished_ratio=.45,
                              abandon_ratio=.15, history_limit=600, ratings={}, rating_strength=1.0)
    assert q[0] > q[1] > 0, q
    rated = profiles.rocchio_over(m, {1: 0, 2: 1}, watch, {1: 300.0, 2: 3000.0}, half_life=21.0, finished_ratio=.45,
                                  abandon_ratio=.15, history_limit=600, ratings={1: 80, 2: 80}, rating_strength=1.0)
    assert rated[0] == pytest.approx(rated[1]), "explicit ratings are not scaled by completion"


def test_embedding_mmr_mixes_other_clusters_into_the_top_ten(tmp_path, clock):
    # the default diversity (0.7 since 2026-09-29) with embedding redundancy; at 0.35 no redundancy
    # term could lift items whose relevance is about half the leaders', and tag cosine alone stays all-A
    now, fake = clock
    ids = page(tmp_path / "a", now, fake, likes(range(1, 6)))
    top = [cluster_of(i) for i in ids[:10]]
    print("clusters", [cluster_of(i) for i in ids])
    assert sum(c != 0 for c in top) >= 2 and top.count(0) > len(top) / 2, top

"""Preference profiles over slot-supplied facts: verdict helpers, decayed tag
profiles, the Rocchio embedding profile, secondary-kind explicit events,
contributor affinity and the shared trial reward.

The one verdict lives in feedloop.taste; this module only re-exports it.
Watch rows are ``{"watched_s", "days", "visits"}`` derived from a Signals row's
actual last-view time and distinct UTC visit days (``watch_rows``), never from
counts alone.
"""
from __future__ import annotations

from collections import defaultdict
import threading
import time
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from feedloop.taste import verdict, item_preferences, chunked_dot

LIKE_ABS_MIN_S = 120.0
LIKE_ABS_MAX_S = 240.0
LIKE_ABS_DURATION_SHARE = 0.05
D_DISLIKE_MIN_WATCH = 60.0
D_SHORT_WATCH_RATIO = 0.5
SHORT_WATCH_ABS_MIN_S = 5.0
D_FINISHED_RATIO = 0.45
D_ABANDON_RATIO = 0.15
ROCCHIO_BETA = 0.5
SECONDARY_EVENT_TAG_SECONDS = 60.0
SECONDARY_EVENT_VEC_WEIGHT = 600.0
AFFINITY_SHRINK = 5.0
AFFINITY_CAP = 0.15
AFFINITY_UNIT_S = 600.0
TRIAL_FULL_S = 3600.0
TRIAL_RATING_FLOOR = AFFINITY_UNIT_S / TRIAL_FULL_S


class TTLCache:
    """One cache shape: a value served while younger than ttl, pinned to an
    external signature, fenced by a generation so a reset during a build
    prevents that build from publishing."""

    def __init__(self, name: str, ttl: float):
        self.name, self.ttl = name, ttl
        self.ts, self.sig, self.value = 0.0, None, None
        self.generation = 0
        self.lock = threading.RLock()
        self.build_lock = threading.RLock()

    def fresh(self, now=None) -> bool:
        return self.value is not None and ((time.time() if now is None else now) - self.ts) < self.ttl

    def touch(self, now=None):
        self.ts = now or time.time()

    def set(self, value, sig=None, now=None, *, generation=None):
        with self.lock:
            if generation is None or generation == self.generation:
                self.value, self.sig, self.ts = value, sig, (time.time() if now is None else now)
        return value

    def clear(self):
        with self.lock:
            self.generation += 1
            self.ts, self.sig, self.value = 0.0, None, None


def feature_cached(cache: TTLCache, signature: Callable[[], Any], builder: Callable[[], Any]):
    """Never publish a build across a feature revision change or an explicit reset.

    ``signature`` returns the committed revisions the value depends on, or None
    when any of them is unavailable (which disables publication)."""
    with cache.build_lock:
        generation = cache.generation
        sig = signature()
        with cache.lock:
            if sig is not None and cache.sig == sig and cache.fresh() and generation == cache.generation:
                return cache.value
        value = builder()
        if sig is not None and signature() == sig:
            cache.set(value, sig=sig, generation=generation)
        return value


def like_bar(duration: float) -> float:
    """Absolute watched seconds that count as a like: 5% clamped to [120, 240]."""
    if duration <= 0:
        return LIKE_ABS_MIN_S
    return min(max(LIKE_ABS_DURATION_SHARE * duration, LIKE_ABS_MIN_S), LIKE_ABS_MAX_S)


def watch_counts(w: Mapping[str, float], duration: float, min_watch: float, short_watch_ratio: float) -> bool:
    """Whether a watch row carries enough signal to be judged at all."""
    if w["watched_s"] >= min_watch:
        return True
    return (duration > 0 and 0 < short_watch_ratio < 1 and w["watched_s"] >= SHORT_WATCH_ABS_MIN_S
            and (w["watched_s"] / duration) >= short_watch_ratio)


def profile_ids(watch, facts, history_limit):
    watched = [sid for sid in watch if facts[sid]["watch_eligible"]]
    watched.sort(key=lambda sid: watch[sid]["days"])
    return list(dict.fromkeys(watched[:max(0, history_limit)] + [sid for sid, f in facts.items() if f["explicit"]]))


def watch_rows(rows: Mapping[Any, Mapping[str, Any]], *, now: float) -> Dict[Any, Dict[str, float]]:
    """Signals rows -> {key: {watched_s, days, visits}} for items with real watch history."""
    out = {}
    for key, row in rows.items():
        watch = row.get("watch")
        if not watch:
            continue
        out[key] = {"watched_s": float(watch["watched_s"]),
                    "days": (now - float(watch["last_at"])) / 86400.0,
                    "visits": float(len(set(watch.get("visit_days") or ())) or 1)}
    return out


def coverage_for(features: Mapping[Any, Mapping[str, Any]], keys: Sequence[Any], *, prefer_full=False):
    """Watched coverage first for watched items, full coverage first for explicit-only
    items, falling back to the other when one is absent. Only positive finite seconds."""
    def clean(mapping):
        return {int(t): float(v) for t, v in (mapping or {}).items()
                if v is not None and float(v) > 0 and np.isfinite(float(v))} or None
    out = {}
    for key in dict.fromkeys(keys):
        feature = features.get(key) or {}
        full, watched = clean(feature.get("tag_seconds")), clean(feature.get("watched_tag_seconds"))
        tags = (full or watched) if prefer_full else (watched or full)
        if tags:
            out[key] = tags
    return out


def build_profiles(watch, durations, features, *, half_life, min_watch, finished_ratio, ratings=None,
                   rating_strength=1.0, abandon_ratio, history_limit, dislike_min_watch=D_DISLIKE_MIN_WATCH,
                   short_watch_ratio=D_SHORT_WATCH_RATIO, engagement_counts=None, extra_tag_events=None, facts=None):
    """Recency-decayed like/dislike tag profiles from watch behaviour.

    extra_tag_events carries verdicted non-watch items: (tag_seconds, boost, is_like)."""
    if facts is None:
        facts = item_preferences(watch, durations, ratings=ratings, engagement_counts=engagement_counts,
                                 min_watch=min_watch, short_watch_ratio=short_watch_ratio, finished_ratio=finished_ratio,
                                 abandon_ratio=abandon_ratio, dislike_min_watch=dislike_min_watch, rating_strength=rating_strength)
    considered = profile_ids(watch, facts, history_limit)
    watched_ids = [sid for sid in considered if facts[sid]["watch_eligible"]]
    explicit_only = [sid for sid in considered if not facts[sid]["watch_eligible"]]
    tag_seconds = coverage_for(features, watched_ids)
    for sid, tags in coverage_for(features, explicit_only, prefer_full=True).items():
        total = sum(v for v in tags.values() if v > 0)
        if total > 0:
            tag_seconds[sid] = {t: SECONDARY_EVENT_TAG_SECONDS * v / total for t, v in tags.items() if v > 0}
    liked, disliked = defaultdict(float), defaultdict(float)
    n_like = n_dislike = 0
    like_seconds = dislike_seconds = 0.0
    for sid in considered:
        tags = tag_seconds.get(sid)
        if not tags:
            continue
        fact = facts[sid]
        w = watch.get(sid)
        decay = (0.5 ** (w["days"] / half_life) if fact["watch_eligible"] and half_life > 0 else 1.0)
        repeat = 1.0 + 0.25 * max(0.0, w["visits"] - 1) if fact["watch_eligible"] else 1.0
        if fact["is_like"]:
            n_like += 1
            like_seconds += w["watched_s"] if w is not None else 0.0
            for tag, secs in tags.items():
                liked[tag] += secs * decay * repeat * fact["boost"]
        elif fact["is_dislike"]:
            n_dislike += 1
            dislike_seconds += w["watched_s"] if w is not None else 0.0
            for tag, secs in tags.items():
                disliked[tag] += secs * decay * fact["boost"]
    n_events = 0
    for tags_map, boost, is_like in (extra_tag_events or []):
        if not tags_map:
            continue
        n_events += 1
        target = liked if is_like else disliked
        if is_like:
            n_like += 1
        else:
            n_dislike += 1
        for tag, secs in tags_map.items():
            target[tag] += secs * boost
    meta = {"secondary_events": n_events, "items_considered": len(considered), "liked_items": n_like,
            "disliked_items": n_dislike, "liked_watch_s": round(like_seconds, 1),
            "abandoned_watch_s": round(dislike_seconds, 1), "half_life_days": half_life}
    return dict(liked), dict(disliked), meta


def rocchio_over(m, index, watch, durations, *, half_life, finished_ratio, abandon_ratio, history_limit, ratings,
                 rating_strength, dislike_min_watch=D_DISLIKE_MIN_WATCH, engagement_counts=None, extras=None,
                 facts=None, min_watch=20.0, short_watch_ratio=D_SHORT_WATCH_RATIO):
    """Rocchio taste vector: decayed, watch-time-weighted liked centroid minus beta
    times the disliked centroid, in whatever space ``m`` spans."""
    extras = list(extras or [])
    dim = m.shape[1] if m is not None else next((len(vec) for vec, _w, _il in extras if vec is not None), 0)
    if not dim:
        return None
    if facts is None:
        facts = item_preferences(watch, durations, ratings=ratings, engagement_counts=engagement_counts,
                                 min_watch=min_watch, short_watch_ratio=short_watch_ratio, finished_ratio=finished_ratio,
                                 abandon_ratio=abandon_ratio, dislike_min_watch=dislike_min_watch, rating_strength=rating_strength)
    liked = np.zeros(dim, dtype=np.float32)
    disliked = np.zeros(dim, dtype=np.float32)
    n_like = n_dis = 0
    for sid in profile_ids(watch, facts, history_limit):
        i = (index or {}).get(sid)
        if i is None or m is None:
            continue
        fact = facts[sid]
        if fact["watch_eligible"]:
            w = watch[sid]
            weight = (0.5 ** (w["days"] / half_life) if half_life > 0 else 1.0) * min(w["watched_s"], 3600.0)
        else:
            weight = SECONDARY_EVENT_VEC_WEIGHT
        if fact["is_like"]:
            liked += weight * fact["boost"] * m[i].astype(np.float32)
            n_like += 1
        elif fact["is_dislike"]:
            disliked += weight * fact["boost"] * m[i].astype(np.float32)
            n_dis += 1
    for vec, weight, is_like in extras:
        if vec is None:
            continue
        if is_like:
            liked += weight * vec.astype(np.float32)
            n_like += 1
        else:
            disliked += weight * vec.astype(np.float32)
            n_dis += 1
    if n_like == 0 or not np.isfinite(liked).all() or float(np.linalg.norm(liked)) <= 1e-8:
        return None
    q = liked / max(float(np.linalg.norm(liked)), 1e-8)
    if n_dis:
        d = disliked / max(float(np.linalg.norm(disliked)), 1e-8)
        q = q - ROCCHIO_BETA * d
        q = q / max(float(np.linalg.norm(q)), 1e-8)
    return q if np.isfinite(q).all() and float(np.linalg.norm(q)) > 1e-8 else None


def secondary_event_extras(events, coverage, means, *, rating_strength=1.0):
    """(tag_extras, vec_extras, n_events) from explicitly rated/engaged secondary items.

    events: {key: {"rating", "engagement_count"}}; coverage: {key: {tag: seconds}}
    (full coverage); means: (matrix, {key: row}) or None. 2026-09-16 orchestrator
    decision: each verdicted item distributes SECONDARY_EVENT_TAG_SECONDS in total
    across its tags by coverage share, matching the ranking contract."""
    tag_extras, vec_extras, n = [], [], 0
    m, index = means if means is not None else (None, {})
    for key, rec in events.items():
        is_like, is_dis, boost = verdict(0.0, 0.0, finished_ratio=D_FINISHED_RATIO, abandon_ratio=D_ABANDON_RATIO,
                                         rating=rec.get("rating"), rating_strength=rating_strength,
                                         engagement_count=rec.get("engagement_count"))
        if not is_like and not is_dis:
            continue
        n += 1
        tags = coverage.get(key) or {}
        total = sum(v for v in tags.values() if v > 0)
        if total > 0:
            tag_extras.append(({t: SECONDARY_EVENT_TAG_SECONDS * v / total for t, v in tags.items() if v > 0}, boost, is_like))
        if m is not None and index.get(key) is not None:
            vec_extras.append((m[index[key]].astype(np.float32), SECONDARY_EVENT_VEC_WEIGHT * boost, is_like))
    return tag_extras, vec_extras, n


def contributor_affinity(links, liked_primary, seen_primary, liked_secondary=None, seen_secondary=None, watch=None):
    """({contributor: {affinity, exposures, likes, primary, secondary}}, prior).

    links: trusted {key: [contributor_id]}. A unit is AFFINITY_UNIT_S of real watch
    time capped at an hour per primary item; an explicitly rated secondary item is one
    unit. Shrinkage k=AFFINITY_SHRINK toward the user's own global like rate."""
    liked_secondary, seen_secondary, watch = liked_secondary or set(), seen_secondary or set(), watch or {}
    stats = {}
    for key in seen_primary:
        w = watch.get(key) or {}
        units = min(float(w.get("watched_s", 0.0)), 3600.0) / AFFINITY_UNIT_S
        if units <= 0:
            continue
        for ident in links.get(key, ()):
            rec = stats.setdefault(ident, {"exposures": 0.0, "likes": 0.0, "primary": 0, "secondary": 0})
            rec["exposures"] += units
            rec["primary"] += 1
            if key in liked_primary:
                rec["likes"] += units
    for key in seen_secondary:
        for ident in links.get(key, ()):
            rec = stats.setdefault(ident, {"exposures": 0.0, "likes": 0.0, "primary": 0, "secondary": 0})
            rec["exposures"] += 1.0
            rec["secondary"] += 1
            if key in liked_secondary:
                rec["likes"] += 1.0
    total_exp = sum(r["exposures"] for r in stats.values())
    prior = (sum(r["likes"] for r in stats.values()) / total_exp) if total_exp else 0.0
    out = {}
    for ident, r in stats.items():
        out[ident] = {"affinity": (r["likes"] + AFFINITY_SHRINK * prior) / (r["exposures"] + AFFINITY_SHRINK),
                      "exposures": round(r["exposures"], 2), "likes": round(r["likes"], 2),
                      "primary": r["primary"], "secondary": r["secondary"]}
    return out, prior


def affinity_multiplier(idents, aff_map, prior, weight):
    """Hard-capped score multiplier: a wrong link can nudge, never steer."""
    if not idents or not aff_map or weight <= 0:
        return 1.0
    vals = [aff_map[i]["affinity"] for i in idents if i in aff_map]
    if not vals:
        return 1.0
    delta = weight * (sum(vals) / len(vals) - prior)
    return 1.0 + max(-AFFINITY_CAP, min(AFFINITY_CAP, delta))


def trial_reward(watched_s, rating=None, engagement_count=None):
    """[0,1] reward for a LIKED served item: an hour = 1, engagement = 1, a high rating floors at one unit."""
    r = min(max(watched_s, 0.0), TRIAL_FULL_S) / TRIAL_FULL_S
    if engagement_count and engagement_count > 0:
        return 1.0
    if rating is not None and rating >= 80:
        r = max(r, TRIAL_RATING_FLOOR)
    return r


__all__ = [
    "TTLCache", "feature_cached", "like_bar", "watch_counts", "profile_ids", "watch_rows", "coverage_for",
    "build_profiles", "rocchio_over", "secondary_event_extras", "contributor_affinity", "affinity_multiplier",
    "trial_reward", "verdict", "item_preferences", "chunked_dot",
    "LIKE_ABS_MIN_S", "LIKE_ABS_MAX_S", "LIKE_ABS_DURATION_SHARE", "D_DISLIKE_MIN_WATCH", "D_SHORT_WATCH_RATIO",
    "SHORT_WATCH_ABS_MIN_S", "D_FINISHED_RATIO", "D_ABANDON_RATIO", "ROCCHIO_BETA", "SECONDARY_EVENT_TAG_SECONDS",
    "SECONDARY_EVENT_VEC_WEIGHT", "AFFINITY_SHRINK", "AFFINITY_CAP", "AFFINITY_UNIT_S", "TRIAL_FULL_S", "TRIAL_RATING_FLOOR",
]

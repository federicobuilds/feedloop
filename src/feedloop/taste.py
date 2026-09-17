"""Pure functions of the taste recommender, kept import-free so they can be
unit-tested without the backend.

2026-09-03 audit D5: the explore-slot placement bug survived a week because
nothing could exercise it outside a live backend. Everything here has no
side effects and no host imports; ranking.py imports from this module.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_KINDS = ("video", "image")


def verdict(watched_s, duration, *, finished_ratio, abandon_ratio,
            dislike_min_watch=60.0, rating=None, rating_strength=1.0, engagement_count=None):
    """Core verdict math, extracted unchanged for import-safe raw-input ranking."""
    bar = 120.0 if duration <= 0 else min(max(0.05 * duration, 120.0), 240.0)
    if duration <= 0:
        is_like, is_dis = watched_s >= bar, False
    else:
        completion = watched_s / duration
        is_like = completion >= finished_ratio or watched_s >= bar
        is_dis = not is_like and completion <= abandon_ratio and dislike_min_watch <= watched_s < bar
    boost = 1.0
    if rating is not None and rating_strength > 0:
        if rating >= 80:
            is_like, is_dis, boost = True, False, 1.0 + rating_strength
        elif rating <= 40:
            is_like, is_dis, boost = False, True, 1.0 + 0.5 * rating_strength
        else:
            is_like, is_dis = False, False
    if engagement_count and engagement_count > 0:
        is_like, is_dis, boost = True, False, boost * (1.0 + 0.4 * min(engagement_count, 5))
    return is_like, is_dis, boost


def item_preferences(watch, durations, *, ratings=None, engagement_counts=None,
                      min_watch=20.0, short_watch_ratio=0.5, finished_ratio=0.45,
                      abandon_ratio=0.15, dislike_min_watch=60.0, rating_strength=1.0):
    ratings, engagement_counts = ratings or {}, engagement_counts or {}
    facts = {}
    for sid in dict.fromkeys([*watch, *ratings, *engagement_counts]):
        w = watch.get(sid)
        duration = durations.get(sid, 0.0)
        il, idis, boost = verdict(
            w["watched_s"] if w is not None else 0.0, duration,
            finished_ratio=finished_ratio, abandon_ratio=abandon_ratio,
            dislike_min_watch=dislike_min_watch, rating=ratings.get(sid),
            rating_strength=rating_strength, engagement_count=engagement_counts.get(sid))
        explicit = ((ratings.get(sid) is not None and rating_strength > 0) or engagement_counts.get(sid, 0) > 0)
        eligible = w is not None and (w["watched_s"] >= min_watch or (
            duration > 0 and 0 < short_watch_ratio < 1 and w["watched_s"] >= 5.0
            and w["watched_s"] / duration >= short_watch_ratio))
        admitted = explicit or eligible
        facts[sid] = {"admitted": admitted, "is_like": admitted and il,
                      "is_dislike": admitted and idis, "boost": boost,
                      "watch_eligible": eligible, "explicit": explicit}
    return facts


def chunked_dot(matrix, query, chunk_rows=4096):
    """Core dot-product contract with a bounded float32 conversion buffer."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
        raise ValueError("incompatible matrix and query shapes")
    scores = np.empty(matrix.shape[0], dtype=np.float32)
    query = query.astype(np.float32, copy=False)
    for start in range(0, len(scores), chunk_rows):
        scores[start:start + chunk_rows] = matrix[start:start + chunk_rows].astype(np.float32) @ query
    return scores


def eligible_ranked_items(ranked, eligible_ids, excluded, duplicate_groups,
                          already_selected=(), limit=None):
    """Core eligibility contract; only verified, kind-scoped groups are supplied."""
    if limit is not None and limit <= 0:
        return []
    seen = set(already_selected)
    groups = {(key[0], duplicate_groups[key]) for key in seen if duplicate_groups.get(key) is not None}
    out = []
    for key in ranked:
        kind, item_id = key
        allowed = eligible_ids.get(kind)
        group = duplicate_groups.get(key)
        if key in excluded or key in seen or (allowed is not None and item_id not in allowed):
            continue
        if group is not None and (kind, group) in groups:
            continue
        seen.add(key)
        if group is not None:
            groups.add((kind, group))
        out.append(key)
        if limit is not None and len(out) >= limit:
            break
    return out


def blend_emb(visual_sim, audio_sim, audio_weight, mix_sim=None, mix_weight=0.0):
    vis_weight = max(0.0, 1.0 - max(audio_weight, 0.0) - max(mix_weight, 0.0))
    terms = [(w, v) for w, v in (
        (vis_weight, visual_sim), (max(audio_weight, 0.0), audio_sim),
        (max(mix_weight, 0.0), mix_sim)) if v is not None and w > 0]
    if not terms:
        return None
    wsum = sum(w for w, _ in terms)
    return sum(w * v for w, v in terms) / wsum if wsum else None


def team_draft(a: Sequence[int], b: Sequence[int], want: int,
               first_a: bool | None = None) -> Tuple[List[int], Dict[int, str]]:
    """Team-draft interleave of two ranked id lists.

    Both arms alternately draft their best not-yet-picked item; the page
    is the merged draft and each slot remembers which arm nominated it.
    `first_a` fixes the coin flip for tests."""
    out: List[int] = []
    arm_of: Dict[int, str] = {}
    seen = set()
    ia = ib = 0
    a_turn = (random.random() < 0.5) if first_a is None else first_a
    while len(out) < want and (ia < len(a) or ib < len(b)):
        lst, idx, arm = (a, ia, "base") if a_turn else (b, ib, "cand")
        while idx < len(lst) and lst[idx] in seen:
            idx += 1
        if idx < len(lst):
            sid = lst[idx]
            out.append(sid)
            seen.add(sid)
            arm_of[sid] = arm
            idx += 1
        if a_turn:
            ia = idx
        else:
            ib = idx
        if (a_turn and ib < len(b)) or (not a_turn and ia < len(a)):
            a_turn = not a_turn
    return out, arm_of


def shannon_entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values()) or 1
    ent = 0.0
    for n in counts.values():
        if n > 0:
            p = n / total
            ent -= p * math.log(p)
    return ent


def cooldown_multiplier(days: float, *, cooldown_days: float, recovery_days: float,
                        judgeable: bool = True, is_dislike: bool = False) -> float:
    """0 while resting, then a linear fade back to 1.0.

    Verdict-aware (2026-09-02): a peek rests a sixth, anything consumed or
    sampled rests half, only a dislike serves the full term."""
    if not judgeable:
        effective = cooldown_days / 6.0
    elif is_dislike:
        effective = cooldown_days
    else:
        effective = cooldown_days * 0.5
    if days < effective:
        return 0.0
    if recovery_days <= 0:
        return 1.0
    return min(1.0, (days - effective) / recovery_days)


def place_explore(core: List[int], explore_ids: List[int], offset: int, limit: int) -> List[int]:
    """Put explore picks INSIDE the requested page window.

    The ranked list is `limit + offset` long. Replacing its tail (the old
    behavior) put explore items past the window on every page but the last
    and dropped two ranked items per page (2026-09-03 audit). Now the last
    len(explore) slots of [offset, offset+limit) are replaced, and later
    pages are untouched."""
    if not explore_ids:
        return list(core)
    page_len = limit or 20
    win_end = min(len(core), offset + page_len)
    cut = max(offset, win_end - len(explore_ids))
    return core[:cut] + list(explore_ids) + core[win_end:]


def welch_interval(n_a: int, sum_a: float, sq_a: float,
                   n_b: int, sum_b: float, sq_b: float, z: float) -> Tuple[float, float]:
    """(mean_b - mean_a, half-width) for two independent samples given their
    running moments. The tuner's decision is |diff| > half-width."""
    ma, mb = sum_a / n_a, sum_b / n_b
    va = max((sq_a - sum_a * sum_a / n_a) / max(n_a - 1, 1), 0.0)
    vb = max((sq_b - sum_b * sum_b / n_b) / max(n_b - 1, 1), 0.0)
    se = math.sqrt(max(va / n_a + vb / n_b, 1e-12))
    return mb - ma, z * se


def tagging_pending(status: Mapping) -> int:
    """Pending count from a tagging status.json (tagging stack)."""
    counts = status.get("counts") or {}
    return sum(v for k, v in counts.items() if k not in ("applied", "failed"))


def embed_pending(status: Mapping) -> int:
    """Pending count from an embed_status.json (embedder stack)."""
    return sum(v for k, v in status.items()
               if k not in ("ts", "applied", "failed", "done") and isinstance(v, int))

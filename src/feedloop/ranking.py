"""Import-safe ranking request contracts, shared by serving and frozen evaluation.

Transient values are serialized text because the host discards undeclared
config keys. Parsing is separate from ranking; delivery IDs never enter a
content cache key or imply that an item was viewed.
"""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from feedloop.taste import (
    DEFAULT_KINDS, item_preferences, chunked_dot, eligible_ranked_items, cooldown_multiplier,
    team_draft, place_explore, blend_emb,
)


SUPPORTED_VARIANTS = ("current", "audit_old", "admission_only")
AUDITED_ADMISSION = {
    "revision": "11b5266068fc46425eacc4662e4bd38152a55061",
    "path": "recommender.py",
    "sha256": "ca23a3976c36cbdd7643243f17e7f03c1ec0d381197cd669b000689154f8aa4f",
    "scope": "primary-kind admission only; shared current profiles, eligibility, features and scoring",
    "tie_policy": "canonical ID order at the supplied query boundary",
}
VARIANT_CONTRACT = {
    "current": {"policy": "union", "normalization": "corrected_candidates", "historical_ranker_reproduction": False},
    "audit_old": {"policy": "audited_tag_gated_look_then_cut", "normalization": "paired_candidate_union",
                  "historical_ranker_reproduction": False, "baseline": AUDITED_ADMISSION},
    "admission_only": {"policy": "union", "normalization": "paired_candidate_union",
                       "historical_ranker_reproduction": False, "baseline": AUDITED_ADMISSION},
}
REQUIRED_CONFIG = (
    "half_life_days", "min_watch_seconds", "finished_ratio", "abandon_ratio",
    "dislike_min_watch_seconds", "short_watch_ratio", "history_limit", "rating_strength",
    "dislike_strength", "profile_tags", "candidate_pool", "bodyparts_weight", "max_tag_share",
    "length_floor_seconds", "embedding_weight", "taste_audio_weight", "taste_mix_weight",
    "diversity", "calibration", "cooldown_days", "recovery_days", "impression_discount",
    "contributor_affinity_weight", "image_events_enabled", "include_images", "images_share",
    "explore_slots", "control_rate", "experiment", "vector_spaces",
)


TRANSIENT_CONFIG = [
    {"name": name, "label": label, "type": "text", "default": "", "persist": False}
    for name, label in (
        ("session_id", "Session ID"),
        ("request_id", "Request ID"),
        ("client_request_id", "Client Request ID"),
        ("intent", "Temporary Intent JSON"),
        ("eligibility", "Eligibility JSON"),
        ("filter_identity", "Filter Identity"),
        ("cursor", "Ranking Cursor JSON"),
    )
]


def _ids(value, field):
    if not isinstance(value, (list, tuple)) or len(value) > 2048:
        raise ValueError(f"invalid {field}")
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"invalid {field}")
    return tuple(sorted(set(value)))


def _json_object(value, field, allowed, decode):
    if value is None or value == "":
        return {}
    if not isinstance(value, str) or len(value) > 65536:
        raise ValueError(f"invalid {field}")
    if decode is None:
        raise ValueError("serialized context requires a decoder at the serving boundary")

    def unique(pairs):
        out = {}
        for key, item in pairs:
            if key in out:
                raise ValueError(f"invalid {field}")
            out[key] = item
        return out

    try:
        result = decode(value, object_pairs_hook=unique)
    except (ValueError, RecursionError):
        raise ValueError(f"invalid {field}") from None
    if not isinstance(result, dict) or not set(result) <= allowed:
        raise ValueError(f"invalid {field}")
    return result


def parse_context(config, *, seed_ids=(), decode=None, resolve_eligibility=None, kinds=DEFAULT_KINDS):
    """Validate normalized host config; do not read ignored top-level fields.

    kinds names every item kind, primary first. intent: seed_<primary>_ids,
    tag_ids and exclude_<kind>_ids lists. eligibility: <kind>_ids lists; absent
    or null is unrestricted, while an empty list allows no items of that kind.
    """
    out = {}
    for field in ("session_id", "request_id", "client_request_id", "filter_identity"):
        value = config.get(field)
        if value is None or value == "":
            value = None
        elif (not isinstance(value, str) or not 1 <= len(value) <= 128 or not value.isascii()
              or not value[0].isalnum() or any(not (c.isalnum() or c in "._:-") for c in value)):
            raise ValueError(f"invalid {field}")
        out[field] = value
    seed_field = f"seed_{kinds[0]}_ids"
    fields = {seed_field, "tag_ids"} | {f"exclude_{kind}_ids" for kind in kinds}
    intent = _json_object(config.get("intent"), "intent", fields, decode)
    out["intent"] = {field: _ids(intent.get(field, []), f"intent.{field}") for field in sorted(fields)}
    out["intent"][seed_field] = tuple(sorted(set(out["intent"][seed_field]) | set(_ids(seed_ids, "seed_ids"))))
    eligibility = _json_object(config.get("eligibility"), "eligibility", {kind + "_ids" for kind in kinds} | {"snapshot_id"}, decode)
    if "snapshot_id" in eligibility:
        identity = eligibility["snapshot_id"]
        if (set(eligibility) != {"snapshot_id"} or not isinstance(identity, str) or
                not identity.startswith("elig:v1:") or len(identity) != 72 or
                any(c not in "0123456789abcdef" for c in identity[8:]) or not callable(resolve_eligibility)):
            raise ValueError("invalid eligibility snapshot")
        members = resolve_eligibility(identity)
        if not isinstance(members, dict) or set(members) != set(kinds):
            raise ValueError("invalid eligibility membership")
        out["eligible_ids"] = {}
        for kind, ids in members.items():
            if (not isinstance(ids, (list, tuple)) or
                    any(type(i) is not int or not 0 < i <= 9223372036854775807 for i in ids) or
                    len(ids) != len(set(ids))):
                raise ValueError("invalid eligibility membership")
            out["eligible_ids"][kind] = tuple(sorted(ids))
        out["eligibility_spec"] = {"snapshot_id": identity}
    else:
        out["eligible_ids"] = {
            kind: None if eligibility.get(kind + "_ids") is None else _ids(eligibility[kind + "_ids"], "eligibility." + kind)
            for kind in kinds
        }
        out["eligibility_spec"] = {kind + "_ids": None if ids is None else list(ids)
                                   for kind, ids in out["eligible_ids"].items()}
    cursor = _json_object(config.get("cursor"), "cursor", {"generation_id", "offset", "after"}, decode)
    if cursor and (set(cursor) != {"generation_id", "offset", "after"} or
                   not isinstance(cursor["generation_id"], str) or len(cursor["generation_id"]) != 64 or
                   any(c not in "0123456789abcdef" for c in cursor["generation_id"]) or
                   type(cursor["offset"]) is not int or cursor["offset"] < 0 or
                   not isinstance(cursor["after"], str) or not cursor["after"].startswith(tuple(kind + ":" for kind in kinds)) or
                   not cursor["after"].split(":", 1)[1].isdigit()):
        raise ValueError("invalid cursor")
    out["cursor"] = cursor or None
    return out


def context_key(context):
    """Content identity excludes delivery identity but includes empty allowlists."""
    return (
        tuple(sorted(context["intent"].items())),
        (("snapshot_id", context["eligibility_spec"]["snapshot_id"]),) if "snapshot_id" in context["eligibility_spec"]
        else tuple(sorted(context["eligible_ids"].items())),
        context["filter_identity"],
    )


def admit_sources(ordered, budgets, eligible_ids, excluded, duplicate_groups, *, already_selected=()):
    """Apply eligibility before independent source cuts, then deduplicate the union."""
    sources = {name: eligible_ranked_items(items, eligible_ids, excluded, duplicate_groups,
                                         already_selected=already_selected, limit=budgets[name])
               for name, items in ordered.items()}
    pool = eligible_ranked_items((key for items in sources.values() for key in items),
                                 eligible_ids, excluded, duplicate_groups, already_selected=already_selected)
    return sources, pool


def embedding_candidates(ids, matrix, query, *, exclude=(), eligible_ids=None, limit, kind=DEFAULT_KINDS[0]):
    if ids is None or matrix is None or query is None or limit <= 0 or eligible_ids == ():
        return []
    scores = chunked_dot(matrix, query)
    order = np.lexsort((np.asarray(ids, dtype=np.int64), -scores))
    rows = ((kind, int(ids[pos])) for pos in order if math.isfinite(float(scores[pos])))
    return [sid for _kind, sid in eligible_ranked_items(
        rows, {kind: None if eligible_ids is None else set(eligible_ids)},
         {(kind, sid) for sid in exclude}, {}, limit=limit)]


def merge_look_scores(paired, channels):
    """Keep paired scores exact; fill missing scores from available independent channels."""
    result = dict(paired)
    totals, counts = defaultdict(float), defaultdict(int)
    for scores in channels:
        for key, value in scores.items():
            if key not in paired:
                totals[key] += value
                counts[key] += 1
    result.update((key, value / counts[key]) for key, value in totals.items())
    return result


def share_vector(raw, duration, idf, cats, *, bodyparts_weight, max_tag_share):
    denom = duration if duration > 0 else sum(raw.values())
    if denom <= 0:
        return {}
    return {tag: min(seconds / denom, max_tag_share) * idf.get(tag, 1.0) *
            (bodyparts_weight if cats.get(tag) == "bodyparts" else 1.0)
            for tag, seconds in raw.items() if seconds > 0}


def face_overlap(seed_ids, candidate_ids):
    if not seed_ids or not candidate_ids:
        return None
    return len(seed_ids & candidate_ids) / max(1, min(len(seed_ids), len(candidate_ids)))


def embedding_similarity(seed, candidate, sem_share, audio_weight, mix_weight=0.0):
    look_share = max(0.0, 1.0 - audio_weight - mix_weight)
    weights = {"visual": (1.0 - sem_share) * look_share, "semvisual": sem_share * look_share,
               "audioembed": audio_weight, "audiomix": mix_weight}
    terms = [(weight, float(np.dot(seed[model], candidate[model])))
             for model, weight in weights.items() if weight > 0 and model in seed and model in candidate]
    total = sum(weight for weight, _score in terms)
    return sum(weight * score for weight, score in terms) / total if total else None


def mean_seed_vectors(seed_ids, embeddings):
    merged = {}
    for sid in seed_ids:
        for model, vector in embeddings.get(sid, {}).items():
            merged[model] = merged[model] + vector if model in merged else vector.copy()
    return {model: vector / np.linalg.norm(vector) for model, vector in merged.items()
            if np.isfinite(vector).all() and np.linalg.norm(vector) > 0}


def similar_components(comps, *, embed_weight, face_weight, details=None):
    scored = []
    for sid, vector, tag_sim, emb_sim, face_sim, cooldown in comps:
        tag_term = tag_sim if emb_sim is None else (1.0 - embed_weight) * tag_sim
        embedding_term = 0.0 if emb_sim is None else embed_weight * emb_sim
        face_scale = 1.0 if face_sim is None else 1.0 - face_weight
        face_term = 0.0 if face_sim is None else face_weight * face_sim
        score = (face_scale * (tag_term + embedding_term) + face_term) * cooldown
        if details is not None:
            details.setdefault(sid, {}).update(score=score, tag_term=tag_term, embedding_term=embedding_term,
                                               face_blend_multiplier=face_scale, face_term=face_term,
                                               history_multiplier=cooldown)
        if score > 0:
            scored.append((score, sid, vector, "other"))
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored


def seed_query(query, matrix, index, seed_ids):
    """Add a unit seed centroid to a private query; never edit stored vectors."""
    if matrix is None or not seed_ids:
        return query
    rows = [matrix[index[sid]].astype(np.float32) for sid in seed_ids if sid in index]
    if not rows:
        return query
    centroid = np.mean(rows, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0:
        return query
    combined = centroid / norm
    if query is not None:
        combined = combined + query
    norm = float(np.linalg.norm(combined))
    return combined / norm if norm > 0 else None


def cosine(a, b, *, contributions=None):
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    terms = [(k, v * large.get(k, 0.0)) for k, v in small.items()] if contributions is not None else None
    dot = sum(value for _, value in terms) if terms is not None else sum(v * large.get(k, 0.0) for k, v in small.items())
    if dot <= 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if contributions is not None and na and nb:
        contributions.extend({"tag_id": tag, "contribution": value / (na * nb)} for tag, value in terms if value)
    return dot / (na * nb) if na and nb else 0.0


def weights_from_profiles(liked, disliked, df, corpus, *, profile_tags, dislike_strength):
    if not liked:
        return {}, {"reason": "no positive signal yet"}
    total_like = sum(liked.values()) or 1.0
    total_dislike = sum(disliked.values()) or 0.0
    corpus = max(corpus, 1)
    weights = {}
    for tag in sorted(set(liked) | set(disliked)):
        idf = math.log(1.0 + corpus / max(df.get(tag, 1), 1))
        diff = liked.get(tag, 0.0) / total_like - (disliked.get(tag, 0.0) / total_dislike if total_dislike > 0 else 0.0)
        if diff < 0:
            diff *= dislike_strength
        score = diff * idf
        if abs(score) > 1e-9:
            weights[tag] = score
    ranked = sorted(weights.items(), key=lambda item: (-abs(item[1]), item[0]))
    kept = dict(ranked[:profile_tags])
    return kept, {"profile_tags": len(kept), "positive_tags": sum(v > 0 for v in kept.values()),
                  "negative_tags": sum(v < 0 for v in kept.values()), "corpus_items": corpus}


def relevance(vector, weights, tag_category, duration, *, bodyparts_weight,
              max_tag_share, length_floor, contributions=None):
    cap = max_tag_share * duration if duration > 0 else None
    total = 0.0
    denominator = max(duration, length_floor)
    for tag, secs in vector.items():
        weight = weights.get(tag)
        if not weight:
            continue
        capped = min(secs, cap) if cap else secs
        multiplier = bodyparts_weight if tag_category.get(tag) == "bodyparts" else 1.0
        term = weight * capped * multiplier
        total += term
        if contributions is not None:
            contributions.append({"tag_id": tag, "seconds": secs, "capped_seconds": capped,
                                  "weight": weight, "category_multiplier": multiplier,
                                  "contribution": term / denominator})
    return total / denominator


def category_shares(weights, tag_category, bodyparts_weight):
    sums = defaultdict(float)
    for tag, weight in weights.items():
        if weight <= 0:
            continue
        category = tag_category.get(tag, "other")
        sums[category] += weight * (bodyparts_weight if category == "bodyparts" else 1.0)
    total = sum(sums.values())
    return {category: value / total for category, value in sums.items()} if total else {}


def dominant_category(vector, tag_category, weights, bodyparts_weight):
    sums = defaultdict(float)
    for tag, seconds in vector.items():
        weight = weights.get(tag, 0.0)
        if weight <= 0:
            continue
        category = tag_category.get(tag, "other")
        sums[category] += seconds * weight * (bodyparts_weight if category == "bodyparts" else 1.0)
    return max(sums, key=sums.get) if sums else "other"


def score_components(comps, knobs, *, details=None):
    """The serving scorer, with explanations captured at the actual operations."""
    blends = {row[0]: blend_emb(row[4], row[5], knobs["taste_audio_weight"],
                               row[6], knobs["taste_mix_weight"]) for row in comps}
    have = [value for value in blends.values() if value is not None]
    emb_hi = max(have) if have else 0.0
    scored = []
    for sid, vec, cat, tag_norm, visual, voice, sound, mult_cd, aff_delta in comps:
        eb = blends.get(sid)
        effective_weight = knobs["embedding_weight"] if eb is not None and emb_hi > 0 else 0.0
        enorm = max(eb, 0.0) / emb_hi if eb is not None and emb_hi > 0 else 0.0
        tag_term = (1.0 - effective_weight) * tag_norm
        embedding_term = effective_weight * enorm
        rel = tag_term + embedding_term
        mult_aff = affinity_scale(aff_delta, knobs["contributor_affinity_weight"])
        rel *= mult_cd * mult_aff
        if details is not None:
            details.setdefault(sid, {}).update(
                tag_normalized=tag_norm,
                visual_similarity=visual, voice_similarity=voice, sound_similarity=sound,
                embedding_score=eb, embedding_max=emb_hi, embedding_normalized=enorm,
                tag_term=tag_term, embedding_term=embedding_term,
                history_multiplier=mult_cd, affinity_multiplier=mult_aff, score=rel)
        if rel > 0:
            scored.append((rel, sid, vec, cat))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def affinity_scale(delta, weight):
    return 1.0 + max(-0.15, min(0.15, weight * delta)) if delta is not None and weight > 0 else 1.0


def select(scored, *, want, diversity, calibration, target_shares, details=None):
    """Incremental MMR; strict comparisons preserve input tie order."""
    chosen = []
    last_vector = {}
    counts = defaultdict(int)
    pool = scored[:]
    max_sims = [0.0] * len(pool)
    scale = abs(scored[0][0] if scored else 1.0) or 1.0
    while pool and len(chosen) < want:
        best_idx, best_val, best_trace = 0, -1e18, None
        for idx, (rel, sid, vec, cat) in enumerate(pool):
            if chosen:
                sim = cosine(vec, last_vector)
                if sim > max_sims[idx]:
                    max_sims[idx] = sim
            sim = max_sims[idx]
            have = counts[cat] / (len(chosen) or 1)
            deficit = max(0.0, target_shares.get(cat, 0.0) - have)
            val = rel / scale - diversity * sim + calibration * deficit
            if val > best_val:
                best_idx, best_val = idx, val
                if details is not None:
                    best_trace = {"relevance_normalized": rel / scale, "scale": scale,
                                  "maximum_similarity": sim, "diversity_penalty": diversity * sim,
                                  "category_deficit": deficit, "calibration_bonus": calibration * deficit,
                                  "value": val, "ranked_position": len(chosen)}
        rel, sid, vec, cat = pool.pop(best_idx)
        max_sims.pop(best_idx)
        if details is not None:
            details.setdefault(sid, {})["selection"] = best_trace
        chosen.append(sid)
        last_vector = dict(vec)
        counts[cat] += 1
    return chosen


def _timestamp(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return _instant(value).timestamp()


def _instant(value):
    """UTC wire timestamps only; never infer timezone or truncate precision."""
    if not isinstance(value, str) or len(value) < 20 or not value.endswith("Z"):
        raise ValueError("ranking timestamps must be UTC")
    body = value[:-1]
    if (body[4] != "-" or body[7] != "-" or body[10] != "T" or body[13] != ":" or body[16] != ":" or
            (len(body) > 19 and (body[19] != "." or not 1 <= len(body[20:]) <= 6 or
                                any(c not in "0123456789" for c in body[20:])))):
        raise ValueError("invalid ranking timestamp")
    try:
        return datetime.fromisoformat(body + "+00:00").astimezone(timezone.utc)
    except ValueError:
        raise ValueError("invalid ranking timestamp") from None


def ordered_evidence(evidence, kinds=DEFAULT_KINDS):
    """Validate every record before filtering; order by instants, then event ID."""
    rows, ids = [], set()
    for event in evidence:
        if not isinstance(event, dict) or not isinstance(event.get("event_id"), str) or not event["event_id"]:
            raise ValueError("invalid evidence event_id")
        if event["event_id"] in ids:
            raise ValueError("duplicate evidence event_id")
        ids.add(event["event_id"])
        try:
            known = _instant(event["known_at"])
            if event.get("type") == "initial_state":
                if event["occurred_at"] is not None:
                    raise ValueError("initial state has no occurrence")
                at = known
                if not isinstance(event.get("value"), dict):
                    raise ValueError("invalid initial state")
                if "watch" in event["value"]:
                    watch = event["value"]["watch"]
                    if (event.get("kind") != kinds[0] or not isinstance(watch, dict) or
                            set(watch) != {"watched_s", "last_at", "visit_days"} or
                            type(watch["watched_s"]) not in (int, float) or
                            not math.isfinite(watch["watched_s"]) or watch["watched_s"] < 0):
                        raise ValueError("invalid initial watch")
                    last = _instant(watch["last_at"])
                    days = watch["visit_days"]
                    if (last > known or not isinstance(days, list) or
                            any(type(day) is not int or day > int(last.timestamp() // 86400) for day in days) or
                            days != sorted(set(days))):
                        raise ValueError("invalid initial watch")
            else:
                at = _instant(event["occurred_at"])
            if known < at:
                raise ValueError("knowledge precedes occurrence")
            if event.get("type") == "watch":
                start, end = _instant(event["start_at"]), _instant(event["end_at"])
                if start > end or end != at:
                    raise ValueError("invalid watch interval")
        except (KeyError, ValueError, TypeError, OverflowError):
            raise ValueError("invalid evidence timestamp") from None
        rows.append((at, known, event["event_id"], event))
    return sorted(rows, key=lambda row: row[:3])


def _key(row, kinds):
    if row.get("kind") not in kinds or type(row.get("id")) is not int or row["id"] < 0:
        raise ValueError("invalid ranking item key")
    return row["kind"], row["id"]


def _unit(vector):
    row = np.asarray(vector, dtype=np.float32)
    if row.ndim != 1 or not len(row) or not np.isfinite(row).all():
        raise ValueError("invalid ranking vector")
    norm = float(np.linalg.norm(row))
    if norm <= 0:
        raise ValueError("zero ranking vector")
    return row / norm


def interleave_kinds(primaries, images, want, share):
    """Build one deterministic prefix before pagination, never append page-zero images."""
    out = []
    i = j = 0
    while len(out) < want and (i < len(primaries) or j < len(images)):
        if j < len(images) and (i >= len(primaries) or j < int((len(out) + 1) * share)):
            out.append(images[j])
            j += 1
        elif i < len(primaries):
            out.append(primaries[i])
            i += 1
        else:
            break
    return out


def rank(inputs, *, context, config, seed, variant="current", kinds=DEFAULT_KINDS):
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError("unsupported ranking variant")
    if variant != "current" and config.get("experiment") is not None:
        raise ValueError("admission comparison cannot overlap a scoring-knob experiment")
    return _rank(inputs, context=context, config=config, seed=seed, variant=variant,
                 admission_policy=VARIANT_CONTRACT[variant]["policy"], kinds=kinds)


def audited_admission(tag_scores, look_ids, *, weights_available, pool_size, limit, offset):
    """11b5266 gate: require tags, append new look candidates, then rough-score cut.

    This is NOT tag-only: look candidates survive if the global cut has room.
    The caller supplies the same eligible, canonical query order to both arms.
    """
    if not weights_available or not tag_scores:
        return []
    tags = sorted(tag_scores, key=lambda key: (-tag_scores[key], key))
    look = [key for key in look_ids if key not in tag_scores][:max(40, pool_size // 4)]
    return sorted(tags + look, key=lambda key: -tag_scores.get(key, 0.0))[:max(pool_size, (limit or pool_size) + offset + 40)]


def fingerprint_groups(rows, kinds=DEFAULT_KINDS):
    """Exact whole-item identity requires an MD5 for every attached file."""
    groups = {}
    for row in rows:
        values = []
        files = row.get("files")
        if not isinstance(files, list) or not files:
            continue
        for file in files:
            hashes = {fp["value"].lower() for fp in file.get("fingerprints", [])
                      if isinstance(fp, dict) and str(fp.get("type", "")).lower() == "md5"
                      and isinstance(fp.get("value"), str) and len(fp["value"]) == 32
                      and all(c in "0123456789abcdefABCDEF" for c in fp["value"])}
            if len(hashes) != 1:
                break
            values.append(next(iter(hashes)))
        if len(values) == len(files):
            groups[_key(row, kinds)] = "md5:" + ":".join(sorted(values))
    return groups


def rank_page(comps, image_comps, *, config, target_shares, seed, allowed, excluded,
              duplicate_groups, seeds=(), explanations=None, admitted=None,
              fallback=(), fallback_reasons=(), explore=(), control=(),
              page_size=20, offset=0, limit=20, kinds=DEFAULT_KINDS):
    """One scoring, selection, mixed-kind pagination path for serving and replay.

    Counts and cursors refer to the complete bounded generation, not a page's
    item count. Callers gather facts only; they must not rescore this output.
    """
    allowed = {kind: None if ids is None else frozenset(ids) for kind, ids in allowed.items()}
    rng = random.Random(seed)
    details = {key: dict(value) for key, value in (explanations or {}).items()}
    scoring_seconds = 0.0
    def eligible(order, already=()):
        return eligible_ranked_items(order, allowed, excluded, duplicate_groups,
                                     already_selected=tuple(seeds) + tuple(already))
    def arm(knobs, trace):
        nonlocal scoring_seconds
        started = time.perf_counter()
        scored = [row for row in score_components(comps, knobs, details=trace)
                  if admitted is None or row[1] in admitted]
        scores = {row[1]: row[0] for row in scored}
        images = []
        for key, base, penalty, delta in image_comps:
            multiplier = affinity_scale(delta, knobs["contributor_affinity_weight"])
            score = base * multiplier * penalty
            if not math.isfinite(score) or score <= 0:
                continue
            scores[key] = score
            trace.setdefault(key, {}).update(sources=["images"], base_score=base, affinity_multiplier=multiplier,
                                            history_multiplier=penalty, score=score)
            images.append(key)
        images.sort(key=lambda key: (-scores[key], key))
        scoring_seconds += time.perf_counter() - started
        primaries = select(scored, want=len(scored), diversity=knobs["diversity"],
                           calibration=knobs["calibration"], target_shares=target_shares, details=trace)
        return eligible(interleave_kinds(primaries, images, len(primaries) + len(images), config["images_share"])), scores
    chosen, scores = arm(config, details)
    experiment = config.get("experiment")
    if experiment is not None:
        candidate_details = {key: dict(value) for key, value in details.items()}
        candidate, candidate_scores = arm({**config, experiment["knob"]: experiment["candidate"]}, candidate_details)
        chosen, arms = team_draft(chosen, candidate, len(chosen) + len(candidate), first_a=rng.random() < 0.5)
        for key, source in arms.items():
            if source == "cand":
                details[key], scores[key] = candidate_details[key], candidate_scores[key]
            details[key]["arm"] = source
    fallback_active = not chosen
    if fallback_active:
        candidates = eligible(sorted(fallback))
        rng.shuffle(candidates)
        candidates = candidates[:config["candidate_pool"]]
        chosen = interleave_kinds([k for k in candidates if k[0] == kinds[0]],
                                  [k for k in candidates if k[0] != kinds[0]], len(candidates), config["images_share"])
        for key in chosen:
            scores[key] = 0.0
            details[key] = {**details.get(key, {}), "sources": ["fallback"], "score": 0.0,
                            "fallback": "no_positive_feature_match", "strength": "weak",
                            "fallback_reasons": list(fallback_reasons), "selection": {"method": "eligible_fallback"}}
    elif config["explore_slots"]:
        extras, controls = eligible(sorted(explore), chosen), eligible(sorted(control), chosen)
        rng.shuffle(extras)
        rng.shuffle(controls)
        for start in range(0, len(chosen), max(page_size, 1)):
            picks = []
            control_key = None
            if controls and rng.random() < config["control_rate"]:
                control_key = controls.pop(0)
                picks.append(control_key)
            picks = eligible(picks + extras, chosen)[:min(config["explore_slots"], page_size)]
            for key in picks:
                source = "control" if key == control_key else "explore"
                scores[key] = 0.0
                details[key] = {"sources": [source], source: True, "score": 0.0,
                                "selection": {"method": "random_eligible"}}
            chosen = place_explore(chosen, picks, start, page_size)
            extras, controls = eligible(extras, chosen), eligible(controls, chosen)
    chosen = eligible(chosen)
    items = []
    for position, key in enumerate(chosen):
        details[key]["position"] = position
        items.append({"key": key, "score": scores[key], "position": position, "explanation": details[key]})
    page, total, more = page_items(items, offset=offset, limit=limit)
    return {"items": page, "all_items": items, "total": len(items), "has_more": more,
            "timings": {"scoring": scoring_seconds},
            "next_offset": offset + len(page) if more else None, "fallback_active": fallback_active and bool(items)}


def page_items(items, *, offset, limit):
    if type(offset) is not int or type(limit) is not int or offset < 0 or limit < 0:
        raise ValueError("invalid ranking pagination")
    page = items[offset:offset + limit]
    return page, len(items), bool(limit and offset + len(page) < len(items))


def compare_admission(inputs, *, context, config, seed, kinds=DEFAULT_KINDS):
    """Change only the candidate gate; this is not full historical-ranker replay."""
    if config.get("experiment") is not None:
        raise ValueError("admission comparison cannot overlap a scoring-knob experiment")
    return {
        "mechanism": "same_core_admission_replay", "historical_ranker_reproduction": False,
        "baseline": dict(AUDITED_ADMISSION),
        "control": rank(inputs, context=context, config=config, seed=seed, variant="audit_old", kinds=kinds),
        "treatment": rank(inputs, context=context, config=config, seed=seed, variant="admission_only", kinds=kinds),
    }


def _cutoff_state(catalog, events, *, now, cutoff, config, kinds):
    state, visible_days = {}, defaultdict(set)
    for instant, known_instant, _event_id, event in events:
        key = _key(event, kinds)
        if key not in catalog or known_instant.timestamp() >= cutoff:
            continue
        at = instant.timestamp()
        if at >= cutoff:
            continue
        rec = state.setdefault(key, {"rating": None, "engagement_count": 0, "watched_s": 0.0, "last": None, "days": set()})
        kind, value = event["type"], event["value"]
        if kind == "initial_state":
            rec["rating"] = value.get("rating")
            rec["engagement_count"] = value.get("engagement_count", 0)
            if "watch" in value:
                initial = value["watch"]
                rec["watched_s"] = initial["watched_s"]
                rec["last"] = _timestamp(initial["last_at"])
                rec["days"] = set(initial["visit_days"])
        elif kind == "rating":
            rec["rating"] = value
        elif kind == "engagement_delta":
            rec["engagement_count"] = max(0, rec["engagement_count"] + value)
        elif kind == "watch":
            start = _timestamp(event["start_at"])
            end = _timestamp(event["end_at"])
            if key[0] != kinds[0] or value < 0 or start > end or end != at or value > end - start + 1e-9:
                raise ValueError("invalid watch interval")
            rec["watched_s"] += value
            rec["last"] = max(at, rec["last"] or at)
            rec["days"].add(int(at // 86400))
        elif kind == "visible" and now - at <= 14 * 86400:
            visible_days[key].add(int(at // 86400))
    watch = {key: {"watched_s": rec["watched_s"], "days": (now - rec["last"]) / 86400,
                   "visits": float(len(rec["days"]))} for key, rec in sorted(state.items()) if rec["last"] is not None}
    ratings = {key: rec["rating"] for key, rec in sorted(state.items()) if rec["rating"] is not None}
    engagement_counts = {key: rec["engagement_count"] for key, rec in sorted(state.items()) if rec["engagement_count"] > 0}
    facts = item_preferences(watch, {key: float(row["duration_s"]) for key, row in catalog.items()},
                             ratings=ratings, engagement_counts=engagement_counts,
                             min_watch=config["min_watch_seconds"], short_watch_ratio=config["short_watch_ratio"],
                             finished_ratio=config["finished_ratio"], abandon_ratio=config["abandon_ratio"],
                             dislike_min_watch=config["dislike_min_watch_seconds"], rating_strength=config["rating_strength"])
    return state, watch, ratings, engagement_counts, facts, visible_days


def _hard_eligibility(catalog, context, config, watch, facts, kinds):
    intent = context["intent"]
    seeds = [(kinds[0], int(sid)) for sid in intent.get(f"seed_{kinds[0]}_ids", ())]
    if intent.get("seed") is not None:
        seeds.append(_key(intent["seed"], kinds))
    seeds = sorted(set(seeds))
    allowed = {kind: set() for kind in kinds}
    raw_allowed = context["eligible_ids"]
    if isinstance(raw_allowed, dict):
        if set(raw_allowed) - set(kinds):
            raise ValueError("eligibility must contain resolved membership")
        allowed = {kind: None if raw_allowed.get(kind) is None else set(raw_allowed[kind]) for kind in allowed}
    else:
        for item in raw_allowed:
            kind, id_ = _key(item, kinds)
            allowed[kind].add(id_)
    excluded = set(seeds)
    for kind in kinds:
        excluded.update((kind, id_) for id_ in intent.get(f"exclude_{kind}_ids", ()))
    exclusions, penalties = [], {}
    for key in sorted(catalog):
        fact = facts.get(key, {})
        penalty = 1.0
        if key in watch:
            penalty = cooldown_multiplier(watch[key]["days"], cooldown_days=config["cooldown_days"],
                                          recovery_days=config["recovery_days"], judgeable=fact.get("watch_eligible", False),
                                          is_dislike=fact.get("is_dislike", False))
        penalties[key] = penalty
        reason = None
        if not catalog[key].get("eligible", True) or key[0] not in context["kinds"]:
            reason = "catalog_or_kind_policy"
        elif key[0] != kinds[0] and not config["include_images"]:
            reason = "images_disabled"
        elif key in excluded:
            reason = "intent_exclusion"
        elif allowed[key[0]] is not None and key[1] not in allowed[key[0]]:
            reason = "eligibility"
        elif penalty <= 0:
            reason = "cooldown"
        elif key[0] != kinds[0] and config["image_events_enabled"] and fact.get("is_dislike"):
            reason = "explicit_image_dislike"
        if reason:
            excluded.add(key)
            exclusions.append({"kind": key[0], "id": key[1], "reason": reason})
    return allowed, excluded, seeds, penalties, exclusions


def _validate_config(config):
    missing = set(REQUIRED_CONFIG) - config.keys()
    if missing:
        raise ValueError("unresolved ranking config: " + ", ".join(sorted(missing)))
    for field in ("image_events_enabled", "include_images"):
        if type(config[field]) is not bool:
            raise ValueError("invalid ranking boolean: " + field)
    for field in ("history_limit", "profile_tags", "candidate_pool", "explore_slots"):
        if type(config[field]) is not int or config[field] < (0 if field == "explore_slots" else 1):
            raise ValueError("invalid ranking count: " + field)
    for field in set(REQUIRED_CONFIG) - {"image_events_enabled", "include_images", "experiment", "vector_spaces"}:
        if type(config[field]) not in (int, float) or not math.isfinite(config[field]) or config[field] < 0:
            raise ValueError("invalid ranking number: " + field)
    if config["length_floor_seconds"] <= 0 or not 0 <= config["images_share"] <= 1 or not 0 <= config["control_rate"] <= 1:
        raise ValueError("invalid ranking scale")


def shared_hard_eligibility(inputs, *, context, config, kinds=DEFAULT_KINDS) -> list[str]:
    """Common policy admission for baselines, without scoring or duplicate selection."""
    _validate_config(config)
    if set(inputs) != {"catalog", "features", "evidence"}:
        raise ValueError("ranking accepts raw catalog/features/evidence only")
    catalog = {_key(row, kinds): row for row in inputs["catalog"]}
    if len(catalog) != len(inputs["catalog"]):
        raise ValueError("duplicate catalog item")
    now, cutoff = _timestamp(context["now"]), _timestamp(context["cutoff"])
    if now < cutoff:
        raise ValueError("ranking clock precedes cutoff")
    _state, watch, _ratings, _engagement_counts, facts, _visible = _cutoff_state(
        catalog, ordered_evidence(inputs["evidence"], kinds), now=now, cutoff=cutoff, config=config, kinds=kinds)
    _allowed, excluded, _seeds, _penalties, _exclusions = _hard_eligibility(catalog, context, config, watch, facts, kinds)
    return [f"{kind}:{id_}" for kind, id_ in sorted(catalog) if (kind, id_) not in excluded]


def _rank(inputs, *, context, config, seed, variant, admission_policy, kinds):
    """Rank cutoff-filtered raw catalog/features/evidence, without I/O or mutation.

    Config is fully resolved by the adapter. Comparison variants isolate the
    audited admission gate, not the full historical ranker. Hydration timing covers bounded in-memory feature
    lookup, not database or endpoint latency.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError("unsupported ranking variant")
    if admission_policy not in ("audited_tag_gated_look_then_cut", "union"):
        raise ValueError("unsupported admission policy")
    _validate_config(config)
    if set(inputs) != {"catalog", "features", "evidence"}:
        raise ValueError("ranking accepts raw catalog/features/evidence only")
    if type(seed) is not int:
        raise ValueError("ranking seed must be an integer")
    limit, offset = context["limit"], context["offset"]
    if type(limit) is not int or type(offset) is not int or limit < 0 or offset < 0:
        raise ValueError("invalid ranking pagination")
    now, cutoff = _timestamp(context["now"]), _timestamp(context["cutoff"])
    if now < cutoff:
        raise ValueError("ranking clock precedes cutoff")
    primary = kinds[0]
    events = ordered_evidence(inputs["evidence"], kinds)
    started = stage = time.perf_counter()
    timings = {}
    rng = random.Random(seed)
    experiment = config["experiment"]
    source_config = dict(config)
    if experiment is not None:
        if not isinstance(experiment, dict) or experiment.get("knob") not in {
            "embedding_weight", "taste_audio_weight", "taste_mix_weight", "contributor_affinity_weight", "diversity", "calibration"
        }:
            raise ValueError("unsupported experiment knob")
        candidate_value = experiment.get("candidate")
        if type(candidate_value) not in (int, float) or not math.isfinite(candidate_value) or candidate_value < 0:
            raise ValueError("invalid experiment candidate")
        source_config[experiment["knob"]] = max(source_config[experiment["knob"]], candidate_value)
    catalog = {_key(row, kinds): row for row in inputs["catalog"]}
    if len(catalog) != len(inputs["catalog"]):
        raise ValueError("duplicate catalog item")
    features = {}
    for row in inputs["features"]:
        key = _key(row, kinds)
        if key in features:
            raise ValueError("features must contain only one current projection per item")
        if key in catalog and row.get("complete", True):
            features[key] = row
    keys = sorted(catalog)
    tags = {key: {int(t): float(v) for t, v in features.get(key, {}).get("tag_seconds", {}).items() if v > 0}
            for key in keys}
    categories = {}
    for key in sorted(features):
        categories.update({int(t): category.lower() for t, category in features[key].get("tag_categories", {}).items()})
    df = defaultdict(int)
    for key in keys:
        if key[0] == primary:
            for tag in tags[key]:
                df[tag] += 1
    corpus = max(sum(key[0] == primary and bool(tags[key]) for key in keys), 1)
    duration = {key: float(catalog[key]["duration_s"]) for key in keys}
    state, watch, ratings, engagement_counts, facts, visible_days = _cutoff_state(
        catalog, events, now=now, cutoff=cutoff, config=config, kinds=kinds)
    allowed, excluded, seeds, penalties, exclusions = _hard_eligibility(catalog, context, config, watch, facts, kinds)
    ordered = sorted((key for key in watch if facts[key]["watch_eligible"]), key=lambda key: (watch[key]["days"], key))
    considered = list(dict.fromkeys(ordered[:config["history_limit"]] + [key for key in sorted(facts) if facts[key]["explicit"]]))
    if not config["image_events_enabled"]:
        considered = [key for key in considered if key[0] == primary]
    liked, disliked, vector_weights = defaultdict(float), defaultdict(float), {}
    for key in considered:
        fact, feature = facts[key], features.get(key, {})
        if not fact["is_like"] and not fact["is_dislike"]:
            continue
        weighted_watch = fact["watch_eligible"] and key in watch
        if weighted_watch:
            w = watch[key]
            decay = 0.5 ** (w["days"] / config["half_life_days"]) if config["half_life_days"] > 0 else 1.0
            repeat = 1.0 + 0.25 * max(0.0, w["visits"] - 1)
            tag_data = feature.get("watched_tag_seconds") or tags[key]
            vector_weight = decay * min(w["watched_s"], 3600.0)
        else:
            decay, repeat, vector_weight = 1.0, 1.0, 600.0
            total = sum(tags[key].values())
            tag_data = {tag: 60.0 * seconds / total for tag, seconds in tags[key].items()} if total > 0 else {}
        vector_weights[key] = (vector_weight * fact["boost"], fact["is_like"])
        target = liked if fact["is_like"] else disliked
        for tag, seconds in tag_data.items():
            target[int(tag)] += seconds * decay * fact["boost"] * (repeat if fact["is_like"] else 1.0)
    weights, profile_meta = weights_from_profiles(liked, disliked, df, corpus, profile_tags=config["profile_tags"],
                                                  dislike_strength=config["dislike_strength"])
    profile_meta.update(watch_evidence_s=sum(w["watched_s"] for w in watch.values()),
                        explicit_only_items=sum(facts[key]["explicit"] and not facts[key]["watch_eligible"] for key in considered))
    intent = context["intent"]
    intent_tags = defaultdict(float)
    for key in seeds:
        for tag, seconds in tags.get(key, {}).items():
            intent_tags[tag] += seconds
    for tag in intent.get("tag_ids", ()):
        intent_tags[int(tag)] += 60.0
    additions, _ = weights_from_profiles(intent_tags, {}, df, corpus, profile_tags=config["profile_tags"],
                                          dislike_strength=config["dislike_strength"])
    for tag, value in additions.items():
        weights[tag] = weights.get(tag, 0.0) + value

    spaces = config["vector_spaces"]
    if set(spaces) != {"visual", "semantic", "voice", "sound"}:
        raise ValueError("vector_spaces must map visual, semantic, voice and sound")
    matrices = {}
    for channel, model in spaces.items():
        rows = [(key, features[key]["vectors"][model]) for key in sorted(features)
                if (key[0] == primary or channel in ("visual", "semantic"))
                and model is not None and model in features[key].get("vectors", {})]
        if rows:
            matrix = np.empty((len(rows), len(rows[0][1])), dtype=np.float16)
            for index, (_key_, vector) in enumerate(rows):
                matrix[index] = _unit(vector)
            matrices[channel] = ([key for key, _ in rows], matrix)
    paired = [key for key in sorted(features) if spaces["visual"] in features[key].get("vectors", {})
              and spaces["semantic"] in features[key].get("vectors", {})]
    if paired:
        first = features[paired[0]]["vectors"]
        width = len(first[spaces["visual"]]) + len(first[spaces["semantic"]])
        matrix = np.empty((len(paired), width), dtype=np.float16)
        for i, key in enumerate(paired):
            vectors = features[key]["vectors"]
            matrix[i] = np.concatenate([_unit(vectors[spaces["visual"]]), _unit(vectors[spaces["semantic"]])]) / np.sqrt(2.0)
        matrices["look"] = (paired, matrix)
    similarities, query_vectors = {}, {}
    for channel, (items, matrix) in matrices.items():
        index = {key: i for i, key in enumerate(items)}
        positive = np.zeros(matrix.shape[1], dtype=np.float32)
        negative = np.zeros(matrix.shape[1], dtype=np.float32)
        n_positive = n_negative = 0
        for key in considered:
            if key not in index or key not in vector_weights:
                continue
            weight, is_like = vector_weights[key]
            if is_like:
                positive += weight * matrix[index[key]].astype(np.float32)
                n_positive += 1
            else:
                negative += weight * matrix[index[key]].astype(np.float32)
                n_negative += 1
        query = None
        norm = float(np.linalg.norm(positive))
        if n_positive and norm > 1e-8:
            query = positive / norm
            if n_negative:
                query = query - 0.5 * negative / max(float(np.linalg.norm(negative)), 1e-8)
                query = query / max(float(np.linalg.norm(query)), 1e-8)
        query = seed_query(query, matrix, index, seeds)
        query_vectors[channel] = None if query is None else query.tolist()
        if query is not None and source_config["embedding_weight"] > 0:
            similarities[channel] = dict(zip(items, map(float, chunked_dot(matrix, query))))
    visual = merge_look_scores(similarities.get("look", {}),
                               [similarities.get(channel, {}) for channel in ("visual", "semantic")])
    voice, sound = similarities.get("voice", {}), similarities.get("sound", {})

    affinity_stats = defaultdict(lambda: [0.0, 0.0])
    for key in sorted(state):
        fact = facts.get(key, {})
        units = min(watch.get(key, {}).get("watched_s", 0.0), 3600.0) / 600.0 if key[0] == primary else (
            1.0 if config["image_events_enabled"] and fact.get("explicit") else 0.0)
        for identity in sorted(set(features.get(key, {}).get("identity_ids", ()))):
            affinity_stats[identity][0] += units
            if fact.get("is_like"):
                affinity_stats[identity][1] += units
    exposure = sum(values[0] for values in affinity_stats.values())
    prior = sum(values[1] for values in affinity_stats.values()) / exposure if exposure else 0.0
    affinity = {identity: (values[1] + 5.0 * prior) / (values[0] + 5.0) for identity, values in affinity_stats.items()}
    affinity_delta = {}
    for key in keys:
        vals = [affinity[i] for i in features.get(key, {}).get("identity_ids", ()) if i in affinity]
        affinity_delta[key] = sum(vals) / len(vals) - prior if vals else None
    timings["profile"] = time.perf_counter() - stage
    stage = time.perf_counter()

    duplicate_groups = {key: row["duplicate_group"] for key, row in catalog.items()
                        if row.get("duplicate_verified") and row.get("duplicate_group") is not None}
    for key in keys:
        penalties[key] *= config["impression_discount"] ** min(len(visible_days[key]), 10)

    def eligible(order, *, already=(), count=None):
        return eligible_ranked_items(order, allowed, excluded, {},
                                     already_selected=tuple(seeds) + tuple(already), limit=count)

    pool_size = config["candidate_pool"]
    budget = pool_size
    vector_budget = max(40, pool_size // 4)
    tag_candidates = {}
    for tag in sorted(tag for tag, weight in weights.items() if weight > 0):
        rows = sorted((key for key in keys if key[0] == primary and tag in tags[key]), key=lambda key: (-tags[key][tag], key))
        for key in eligible(rows, count=max(pool_size, 400)):
            tag_candidates.setdefault(key, {})[tag] = tags[key][tag]
    for key, vector in tag_candidates.items():
        vector.update({tag: tags[key][tag] for tag, weight in weights.items() if weight < 0 and tag in tags[key]})
    def rough(key):
        return sum(weights.get(tag, 0.0) * min(seconds, 600.0) * (
            config["bodyparts_weight"] if categories.get(tag) == "bodyparts" else 1.0)
                   for tag, seconds in tag_candidates[key].items())
    sources = {"tags": sorted(tag_candidates, key=lambda key: (-rough(key), key))}
    for name, scores, enabled in [("visual", visual, source_config["embedding_weight"] > 0),
                                  ("voice", voice, source_config["embedding_weight"] > 0 and source_config["taste_audio_weight"] > 0),
                                  ("sound", sound, source_config["embedding_weight"] > 0 and source_config["taste_mix_weight"] > 0)]:
        rows = sorted((key for key in scores if key[0] == primary), key=lambda key: (-scores[key], key)) if enabled else []
        sources[name] = rows
    pre_budget_sources = {name: eligible(items) for name, items in sources.items()}
    pre_budget_sources["tags"] = eligible([key for key in keys if key[0] == primary and
                                            any(weights.get(tag, 0.0) > 0 for tag in tags[key])])
    all_sources, candidate_universe = admit_sources(sources, {"tags": budget, "visual": vector_budget,
                                          "voice": vector_budget, "sound": vector_budget},
                                  allowed, excluded, {}, already_selected=seeds)
    old_pool = []
    if variant != "current":
        look_scores = similarities.get("look", {})
        look_route = eligible(sorted((key for key in look_scores if key[0] == primary),
                                     key=lambda key: (-look_scores[key], key))) if config["embedding_weight"] > 0 else []
        old_pool = audited_admission({key: rough(key) for key in tag_candidates}, look_route,
                                     weights_available=bool(weights), pool_size=pool_size, limit=limit, offset=offset)
    sources = dict(all_sources)
    pool = candidate_universe
    if admission_policy == "audited_tag_gated_look_then_cut":
        pool = old_pool
        sources = {"tags": [key for key in pool if key in tag_candidates],
                   "visual": [key for key in pool if key not in tag_candidates], "voice": [], "sound": []}
    if variant != "current":
        candidate_universe = list(dict.fromkeys(candidate_universe + old_pool))
    pool_set = set(pool)
    image_scores = {}
    image_rows = eligible([key for key in keys if key[0] != primary])
    if not config["include_images"]:
        image_rows = []
    for key in image_rows:
        if key in visual:
            image_scores[key] = visual[key]
    sources["images"] = eligible(sorted(image_scores, key=lambda key: (-image_scores[key], key)), count=budget)
    source_counts = {name: len(items) for name, items in sources.items()}
    source_counts.update(eligible=len(eligible(keys)), union=len(pool), fallback=0, explore=0, control=0)
    timings["candidates"] = time.perf_counter() - stage
    stage = time.perf_counter()
    # A fixed normalization universe prevents admission changing common-item scores.
    hydrated = {key: (tags[key], duration[key]) for key in candidate_universe}
    timings["hydration"] = time.perf_counter() - stage
    stage = time.perf_counter()
    explanations = {}
    prelim = []
    for key, (vector, seconds) in hydrated.items():
        contributions = []
        rel = relevance(vector, weights, categories, seconds, bodyparts_weight=config["bodyparts_weight"],
                        max_tag_share=config["max_tag_share"], length_floor=config["length_floor_seconds"], contributions=contributions)
        explanations[key] = {"tag_contributions": contributions, "tag_score": rel,
                             "profile": dict(profile_meta), "sources": [name for name, rows in sources.items() if key in rows]}
        prelim.append((key, vector, seconds, rel))
    maximum = max((row[3] for row in prelim), default=0.0) or 1.0
    comps = []
    for key, vector, seconds, rel in prelim:
        explanations[key]["tag_max"] = maximum
        comps.append((key, vector, dominant_category(vector, categories, weights, config["bodyparts_weight"]),
                      max(rel, 0.0) / maximum, visual.get(key), voice.get(key), sound.get(key),
                      penalties[key], affinity_delta[key]))
    image_comps = [(key, image_scores[key], penalties[key], affinity_delta[key]) for key in sources["images"]]
    timings["scoring"] = time.perf_counter() - stage
    stage = time.perf_counter()
    target = category_shares(weights, categories, config["bodyparts_weight"])
    fallback_reasons = []
    if not watch and not ratings and not engagement_counts:
        fallback_reasons.append("no_preference_history")
    if not any(value > 0 for value in weights.values()):
        fallback_reasons.append("no_positive_tag_profile")
    if not any(query is not None for query in query_vectors.values()):
        fallback_reasons.append("no_embedding_profile")
    remainder = [key for key in keys if key[0] == primary and key not in candidate_universe]
    page = rank_page(comps, image_comps, config=config, target_shares=target, seed=seed,
                     allowed=allowed, excluded=excluded, duplicate_groups=duplicate_groups, seeds=seeds,
                     explanations=explanations, admitted=pool_set, fallback=[key for key in keys if key[0] == primary],
                     fallback_reasons=fallback_reasons + ["no_positive_feature_match"],
                     explore=[key for key in remainder if sum(s for t, s in tags[key].items() if t not in weights) >= 30.0],
                     control=remainder, page_size=context.get("page_size", limit or 20), offset=offset, limit=limit, kinds=kinds)
    final = [row["key"] for row in page["all_items"]]
    source_counts["fallback"] = len(final) if page["fallback_active"] else 0
    for row in page["all_items"]:
        for source in ("control", "explore"):
            source_counts[source] += int(source in row["explanation"].get("sources", []))
    final_set = set(final)
    for key in pool:
        if key not in final_set and key not in excluded:
            exclusions.append({"kind": key[0], "id": key[1], "reason": "not_selected"})
    items = []
    for row in page["items"]:
        key, explanation = row["key"], row["explanation"]
        explanation["admission_policy"] = admission_policy
        explanation["revisions"] = {"feature": features.get(key, {}).get("revision"),
                                            "model": features.get(key, {}).get("model_revision")}
        if page["fallback_active"] and not liked and not intent_tags:
            explanation["fallback"] = "no_positive_preference_evidence"
        items.append({"kind": key[0], "id": key[1], "score": float(row["score"]), "explanation": explanation})
    route_ids = {name: rows if admission_policy == "union" or name == "tags" else []
                 for name, rows in pre_budget_sources.items()}
    if admission_policy == "audited_tag_gated_look_then_cut":
        route_ids["visual"] = [key for key in look_route if key not in tag_candidates] if weights and tag_candidates else []
    route_ids["images"] = eligible(sorted(image_scores))
    route_ids["fallback"] = eligible([key for key in keys if key[0] == primary]) if source_counts["fallback"] else []
    remaining = eligible([key for key in keys if key[0] == primary and key not in candidate_universe])
    route_ids["explore"] = [key for key in remaining if sum(seconds for tag, seconds in tags[key].items() if tag not in weights) >= 30.0] if config["explore_slots"] else []
    route_ids["control"] = remaining if config["explore_slots"] and config["control_rate"] else []
    inventory = {"scope": "frozen_context", "complete": True, "supported_routes": list(route_ids),
                 "eligible_ids": [{"kind": key[0], "id": key[1]} for key in eligible(keys)],
                 "routes": {name: {"enabled": bool(rows), "complete": True, "stage": "pre_budget",
                                   "eligible_ids": [{"kind": key[0], "id": key[1]} for key in rows]}
                            for name, rows in route_ids.items()}}
    def snapshot(value):
        if isinstance(value, dict):
            return {key: snapshot(value[key]) for key in sorted(value)}
        if isinstance(value, (tuple, list)):
            return [snapshot(item) for item in value]
        return value
    trace = {"policy": admission_policy,
             "admitted_ids": [{"kind": key[0], "id": key[1]} for key in sorted(pool)],
             "invariant_encoding": "canonical_values_not_digests",
             "invariants": {
                 "profile": {"weights": sorted(weights.items()), "queries": query_vectors,
                             "facts": [(key, dict(facts[key])) for key in sorted(facts)]},
                 "eligibility": {"allowed": {kind: None if ids is None else sorted(ids) for kind, ids in allowed.items()},
                                 "excluded": sorted(excluded), "duplicates": sorted(duplicate_groups.items())},
                 "configuration": snapshot(config),
                  "scoring": {"implementation": "score_components/v1", "tag_max": maximum,
                              "normalization_universe": sorted(candidate_universe)},
                 "selection": {"implementation": "select/incremental-max/v1", "target": dict(target)},
                 "images": "affinity-and-fatigue-before-cut/v1", "fallback": "eligible-weak/v1",
                 "explore": "seeded-independent/v1", "control": "seeded-independent/v1",
             }}
    timings["scoring"] += page["timings"]["scoring"]
    timings["selection"] = time.perf_counter() - stage - page["timings"]["scoring"]
    timings["total"] = time.perf_counter() - started
    return {"items": items, "total": page["total"], "has_more": page["has_more"], "next_offset": page["next_offset"],
            "variant_contract": snapshot(VARIANT_CONTRACT[variant]),
            "source_counts": source_counts, "exclusions": exclusions, "timings": timings,
            "route_inventory": inventory, "admission_trace": trace,
            "fallback": {"active": bool(source_counts["fallback"]), "reasons": fallback_reasons}}

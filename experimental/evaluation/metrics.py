"""Offline metrics. Item identity is always (kind, id), never id alone."""
from collections import Counter
import math

import numpy as np


def key(item):
    return item["kind"], item["id"]


def accuracy(order, relevance, k):
    """AP@k divides by min(k, positives); NDCG uses gain 2**grade - 1.

    Unknown items are not inferred dislikes. These are observed-label metrics,
    not estimates corrected for the logging policy's exposure bias.
    """
    if k <= 0 or len(order) != len(set(order)):
        raise ValueError("invalid k or duplicate recommendation key")
    if any(not math.isfinite(g) or g < 0 for g in relevance.values()):
        raise ValueError("invalid relevance grade")
    positives = sum(g > 0 for g in relevance.values())
    result = {"positives": positives, "returned": min(k, len(order))}
    if not positives:
        return result | {"recall": None, "ndcg": None, "map": None}
    hits = 0
    ap = dcg = 0.0
    for rank, item in enumerate(order[:k], 1):
        grade = relevance.get(item, 0)
        if grade > 0:
            hits += 1
            ap += hits / rank
        dcg += (2 ** grade - 1) / math.log2(rank + 1)
    ideal = sum((2 ** g - 1) / math.log2(i + 2)
                for i, g in enumerate(sorted(relevance.values(), reverse=True)[:k]))
    return result | {"recall": hits / positives, "ndcg": dcg / ideal,
                     "map": ap / min(k, positives)}


def gini(counts):
    values = sorted(counts)
    if any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError("invalid concentration counts")
    total = sum(values)
    if not total:
        return None
    n = len(values)
    return sum((2 * i - n - 1) * v for i, v in enumerate(values, 1)) / (n * total)


def popularity(order, eligible, counts, *, counts_complete=True):
    """Bounds vary membership of items tied at the catalog-decile boundary."""
    eligible = set(eligible)
    if not counts_complete:
        return {"status": "unavailable_unknown_counts", "decile_size": None,
                "boundary_count": None, "boundary_ties": None, "exposure_share": None,
                "exposure_share_min": None, "exposure_share_max": None,
                "training_interaction_share": None}
    if not eligible:
        return {"decile_size": 0, "exposure_share": None,
                "exposure_share_min": None, "exposure_share_max": None,
                "training_interaction_share": None}
    n = math.ceil(len(eligible) / 10)
    boundary = sorted((counts.get(i, 0) for i in eligible), reverse=True)[n - 1]
    above = {i for i in eligible if counts.get(i, 0) > boundary}
    tied = {i for i in eligible if counts.get(i, 0) == boundary}
    slots = n - len(above)
    exposures = Counter(order)
    fixed = sum(exposures[i] for i in above)
    tied_exposures = sorted(exposures[i] for i in tied)
    total = len(order)
    interactions = sum(counts.get(i, 0) for i in eligible)
    return {
        "decile_size": n, "boundary_count": boundary, "boundary_ties": len(tied),
        "exposure_share": (fixed + sum(tied_exposures) * slots / len(tied)) / total if total else None,
        "exposure_share_min": (fixed + sum(tied_exposures[:slots])) / total if total else None,
        "exposure_share_max": (fixed + sum(tied_exposures[-slots:])) / total if total else None,
        "training_interaction_share": (sum(counts.get(i, 0) for i in above) + slots * boundary) / interactions
        if interactions else None,
    }


def list_metrics(order, eligible, counts, features, catalog, space, *, counts_complete=True):
    eligible = set(eligible)
    if not set(order) <= eligible or len(order) != len(set(order)):
        raise ValueError("recommendations violate shared eligibility or uniqueness")
    denominator = sum(counts.get(i, 0) for i in eligible) + len(eligible) if counts_complete else None
    novelty = sum(-math.log2((counts.get(i, 0) + 1) / denominator) for i in order) / len(order) if order and counts_complete else None
    valid_pairs = 0
    distance = 0.0
    duplicate_pairs = known_duplicate_pairs = 0
    vectors = {}
    for item in order:
        vector = features.get(item, {}).get("vectors", {}).get(space)
        if vector is not None:
            array = np.asarray(vector, dtype=np.float64)
            norm = np.linalg.norm(array)
            if array.ndim != 1 or not np.isfinite(array).all() or norm <= 0:
                raise ValueError("invalid diversity vector")
            vectors[item] = array / norm
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if a in vectors and b in vectors:
                if vectors[a].shape != vectors[b].shape:
                    raise ValueError("incompatible diversity representations")
                distance += 1.0 - float(np.clip(vectors[a] @ vectors[b], -1, 1))
                valid_pairs += 1
            if catalog[a]["duplicate_verified"] and catalog[b]["duplicate_verified"]:
                known_duplicate_pairs += 1
                group = catalog[a]["duplicate_group"]
                duplicate_pairs += int(group is not None and group == catalog[b]["duplicate_group"])
    groups = Counter(catalog[i]["duplicate_group"] for i in order
                     if catalog[i]["duplicate_verified"] and catalog[i]["duplicate_group"] is not None)
    total_pairs = len(order) * (len(order) - 1) // 2
    return {
        "novelty_bits": novelty, "novelty_smoothing": "add_one_eligible_training_plays",
        "popularity_status": "measured" if counts_complete else "unavailable_unknown_counts",
        "diversity": distance / valid_pairs if valid_pairs else None,
        "diversity_space": space, "valid_pairs": valid_pairs,
        "missing_pairs": total_pairs - valid_pairs, "total_pairs": total_pairs,
        "duplicate_items": sum(n - 1 for n in groups.values()),
        "duplicate_pairs": duplicate_pairs, "duplicate_known_pairs": known_duplicate_pairs,
        "duplicate_unknown_items": sum(not catalog[i]["duplicate_verified"] for i in order),
        "duplicate_pair_rate": duplicate_pairs / known_duplicate_pairs if known_duplicate_pairs else None,
        "popularity": popularity(order, eligible, counts, counts_complete=counts_complete),
    }


def coverage(rows, weights):
    """A finite sweep measures noncoverage, not permanent unreachability."""
    result = {}
    for kind in ("video", "image"):
        eligible = {i for row in rows for i in row["eligible"] if i[0] == kind}
        exposed = Counter(i for row in rows for i in row["order"] if i[0] == kind)
        cov = len(exposed) / len(eligible) if eligible else None
        result[kind] = {"eligible": len(eligible), "recommended": len(exposed),
                        "coverage": cov, "noncoverage": 1 - cov if cov is not None else None,
                        "gini": gini([exposed[i] for i in sorted(eligible)])}
    result["combined_weights"] = weights
    result["combined_coverage"] = (
        sum(weights[k] * result[k]["coverage"] for k in weights if weights[k])
        if all(not weights[k] or result[k]["coverage"] is not None for k in weights) else None
    )
    return result


def session_mean(rows, field):
    sessions = {}
    for row in rows:
        value = row[field]
        if value is not None:
            sessions.setdefault(row["session_id"], []).append(value)
    means = [sum(values) / len(values) for values in sessions.values()]
    return {"value": sum(means) / len(means) if means else None,
            "measured_sessions": len(means), "measured_contexts": sum(map(len, sessions.values()))}

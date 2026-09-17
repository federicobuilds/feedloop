"""Text search over timestamped feature windows and more-like-this over tag
coverage plus mean vectors, both over slot facts. Encoding is supplied by the
optional TextEncoder; this module only validates and normalizes vectors.
Neither operation mutates preferences, records delivery or invokes Annotator.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import time
from typing import Any, Callable, Dict, Mapping

import numpy as np

from feedloop import catalog as catalog_module
from feedloop.profiles import (D_ABANDON_RATIO, D_FINISHED_RATIO, SECONDARY_EVENT_VEC_WEIGHT, rocchio_over,
                               verdict, watch_rows)
from feedloop.ranking import (admit_sources, cosine, embedding_similarity, face_overlap, mean_seed_vectors, page_items,
                              parse_context, seed_query, select, share_vector, similar_components)
from feedloop.slots import DEFAULT_SPACE_ROLES, MissingKeys
from feedloop.taste import DEFAULT_KINDS, chunked_dot, eligible_ranked_items, item_preferences

D_MIN_SIMILARITY = 0.12
D_SOUND_MIN_SIM = 0.08
D_MEAN_WEIGHT = 0.25
D_POOL = 400
D_PERSONALIZATION = 0.15
TASTE_HALF_LIFE_DAYS = 21.0
CHUNK = 4096
SEARCH_REVISION = "search/v1"
SIMILAR_REVISION = "similar/v1"
D_SEED_TAGS = 20
D_POOL_PER_TAG = 400
D_SIMILAR_POOL = 500
D_BODYPARTS_WEIGHT = 0.3
D_MAX_TAG_SHARE = 0.35
D_EMBED_WEIGHT = 0.5
D_SEM_SHARE = 0.5
D_AUDIO_WEIGHT = 0.15
D_MIX_WEIGHT = 0.10
D_FACE_WEIGHT = 0.15
D_COOLDOWN_DAYS = 21.0
D_RECOVERY_DAYS = 60.0
D_DIVERSITY = 0.2
TOP_CONTRIBS = 6


class Sources:
    """The slot bundle discovery reads: catalog, spaces, signals, optional links/encoder."""

    def __init__(self, *, catalog, spaces, signals, encoder=None, links=None, kinds=DEFAULT_KINDS,
                 roles=DEFAULT_SPACE_ROLES, clock=time.time):
        self.catalog, self.spaces, self.signals = catalog, spaces, signals
        self.encoder, self.links = encoder, links
        self.kinds, self.roles, self.clock = tuple(kinds), dict(roles), clock

    @property
    def primary(self):
        return self.kinds[0]

    def fingerprint(self, keys):
        return catalog_module.fingerprint_snapshot(self.catalog, keys, kinds=self.kinds)

    def rows(self, keys):
        """Catalog rows by key; missing keys are absent, never stubs. Any other
        failure propagates: partial hydration is reported, never worked around."""
        keys = sorted(set(keys))
        while keys:
            try:
                result = self.catalog.fetch(keys)
            except MissingKeys as absent:
                if not absent.keys or not absent.keys <= set(keys):
                    raise RuntimeError("catalog_enumeration_failed") from None
                keys = [key for key in keys if key not in absent.keys]
                continue
            return {(row["kind"], row["id"]): row for row in result["items"]}
        return {}

    def means(self, space, keys=None):
        """{key: unit vector} in one space, restricted to keys when given."""
        loaded = self.spaces.matrix(space)
        if loaded is None:
            return {}
        index, matrix = loaded
        wanted = None if keys is None else set(keys)
        out = {}
        for row, key in enumerate(index):
            if wanted is not None and key not in wanted:
                continue
            vector = np.asarray(matrix[row], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.ndim == 1 and np.isfinite(vector).all() and norm > 0:
                out[key] = vector / norm
        return out

    def trusted_links(self, keys=None):
        if self.links is None:
            return {}
        return {key: set(ids) for key, ids in self.links.links(keys).items()}


# ------------------------------------------------------------------- vectors

def unit_vector(vector, dim=None):
    """Validate shape, finiteness and nonzero norm, then normalize."""
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if vec.ndim != 1 or (dim is not None and vec.shape != (dim,)) or not np.isfinite(vec).all() or norm <= 0:
        raise ValueError("invalid query vector")
    return vec / norm


def encode_query(encoder, space, text, dim=None):
    """None when the encoder does not support the space; ValueError on a malformed vector."""
    if encoder is None:
        return None
    vector = encoder.encode(space, text)
    return None if vector is None else unit_vector(vector, dim)


def window_matrix(spaces, space):
    """Aligned (keys, times, unit fp16 matrix) or None when the space has no windows."""
    loaded = spaces.windows(space)
    if loaded is None:
        return None
    keys, times, matrix = loaded
    keys, times = list(keys), np.asarray(times, dtype=np.float32)
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or len(keys) != matrix.shape[0] or len(times) != matrix.shape[0]:
        raise ValueError("misaligned window rows")
    keep = [i for i in range(len(keys)) if np.isfinite(matrix[i].astype(np.float32)).all() and np.any(matrix[i])]
    matrix = matrix[keep].astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = (matrix / np.clip(norms, 1e-8, None)).astype(np.float16)
    return [keys[i] for i in keep], times[keep], unit


def search_windows(qvec, windows, min_sim, mean_weight, top):
    """Per item: best-window similarity blended with the mean over its windows."""
    if windows is None:
        return [], 0
    keys, ts, m = windows
    if m.shape[0] == 0:
        return [], 0
    sims = chunked_dot(m, qvec, chunk_rows=CHUNK)
    best: Dict[Any, list] = {}
    for i, key in enumerate(keys):
        s = float(sims[i])
        rec = best.setdefault(key, [s, float(ts[i]), 0.0, 0])
        if s > rec[0]:
            rec[0], rec[1] = s, float(ts[i])
        rec[2] += s
        rec[3] += 1
    scored = []
    for key, (mx, bt, total, n) in best.items():
        if mx < min_sim:
            continue
        mean_s = total / max(n, 1)
        scored.append(((1.0 - mean_weight) * mx + mean_weight * mean_s, mx, mean_s, bt, key))
    scored.sort(key=lambda row: (-row[0], -row[1], row[4]))
    return (scored[:top] if top else scored), int(m.shape[0])


def fuse(look, sound, top):
    """Reciprocal-rank fusion: ranks, not raw scores, since the cosine scales differ."""
    K = 60.0
    agg: Dict[Any, list] = {}
    for rank, (score, mx, mn, bt, key) in enumerate(look):
        rec = agg.setdefault(key, [0.0, None, None])
        rec[0] += 1.0 / (K + rank)
        rec[1] = (score, mx, mn, bt)
    for rank, (score, mx, mn, bt, key) in enumerate(sound):
        rec = agg.setdefault(key, [0.0, None, None])
        rec[0] += 1.0 / (K + rank)
        rec[2] = (score, mx, mn, bt)
    out = []
    for key, (rrf, lk, sd) in agg.items():
        primary = lk if lk is not None else sd
        out.append((rrf, primary[1], primary[2], primary[3], key))
    out.sort(key=lambda row: (-row[0], row[4]))
    return out[:top] if top else out


def rerank_query_bands(scored, taste_sims, personalization):
    """Taste can reorder only anchored bands spanning 2% of the top query score."""
    if not scored or not taste_sims or not math.isfinite(personalization) or personalization <= 0:
        return list(scored)
    width = max(row[0] for row in scored) * 0.02
    if width <= 0:
        return list(scored)
    strength = min(personalization, 1.0)
    ordered = sorted(scored, key=lambda row: row[0], reverse=True)
    out, start = [], 0
    while start < len(ordered):
        anchor = ordered[start][0]
        end = start + 1
        while end < len(ordered) and anchor - ordered[end][0] <= width:
            end += 1
        band = ordered[start:end]
        slots = [i for i, row in enumerate(band) if row[4] in taste_sims and math.isfinite(taste_sims[row[4]])]
        ranked = sorted((band[i] for i in slots), key=lambda row: (
            (1.0 - strength) * (row[0] - anchor) / width
            + strength * max(-1.0, min(1.0, taste_sims[row[4]])) / 2.0), reverse=True)
        for i, row in zip(slots, ranked):
            band[i] = row
        out.extend(band)
        start = end
    return out


def taste_prior(sources: Sources, space, *, half_life=TASTE_HALF_LIFE_DAYS):
    """Shared verdicts projected into one embedding space, from Signals facts."""
    try:
        signals = sources.signals.read()
    except Exception:
        return None
    now = sources.clock()
    rows = signals["rows"]
    watch = watch_rows(rows, now=now)
    primary = sources.primary
    ratings = {k: r["rating"] for k, r in rows.items() if r.get("rating") is not None}
    counts = {k: r.get("engagement_count", 0) for k, r in rows.items() if r.get("engagement_count")}
    catalog_rows = sources.rows(set(watch) | set(ratings) | set(counts))
    durations = {k: float(row.get("duration_s") or 0.0) for k, row in catalog_rows.items()}
    primary_watch = {k: w for k, w in watch.items() if k[0] == primary}
    facts = item_preferences(primary_watch, durations, ratings={k: v for k, v in ratings.items() if k[0] == primary},
                             engagement_counts={k: v for k, v in counts.items() if k[0] == primary})
    verdicted = [k for k, f in facts.items() if f["is_like"] or f["is_dislike"]]
    means = sources.means(space, verdicted)
    ids = list(means)
    matrix = np.stack([means[k] for k in ids]) if ids else None
    extras = []
    secondary = {k for k in set(ratings) | set(counts) if k[0] != primary}
    secondary_means = sources.means(space, secondary)
    for key in secondary:
        liked, disliked, boost = verdict(0.0, 0.0, finished_ratio=D_FINISHED_RATIO, abandon_ratio=D_ABANDON_RATIO,
                                         rating=ratings.get(key), engagement_count=counts.get(key))
        if (liked or disliked) and key in secondary_means:
            extras.append((secondary_means[key], SECONDARY_EVENT_VEC_WEIGHT * boost, liked))
    return rocchio_over(matrix, {k: i for i, k in enumerate(ids)}, primary_watch, durations, half_life=half_life,
                        finished_ratio=D_FINISHED_RATIO, abandon_ratio=D_ABANDON_RATIO, history_limit=600,
                        ratings={k: v for k, v in ratings.items() if k[0] == primary}, rating_strength=1.0,
                        engagement_counts={k: v for k, v in counts.items() if k[0] == primary}, extras=extras, facts=facts)


def search(sources: Sources, query, mode="look", *, config=None, context=None, offset=0, limit=20,
           resolve_eligibility=None):
    """Text search over the semantic (look) and sound spaces with taste tie-break."""
    cfg = dict(config or {})
    query = str(query or "").strip()
    min_sim = float(cfg.get("min_similarity", D_MIN_SIMILARITY))
    sound_min_sim = float(cfg.get("sound_min_similarity", D_SOUND_MIN_SIM))
    mean_weight = float(cfg.get("mean_weight", D_MEAN_WEIGHT))
    personalization = float(cfg.get("personalization", D_PERSONALIZATION))
    space = mode or "look"
    if space == "visual":
        space = "look"
    if space not in ("look", "sound", "both"):
        return {"items": [], "total": 0, "has_more": False, "status": "error", "error_code": "invalid_mode"}
    try:
        context = parse_context(context or {}, decode=json.loads, resolve_eligibility=resolve_eligibility, kinds=sources.kinds)
    except ValueError:
        return {"items": [], "total": 0, "has_more": False, "status": "error", "error_code": "invalid_context"}
    low = query.lower()
    for prefix, sp in (("sound:", "sound"), ("both:", "both")):
        if low.startswith(prefix):
            query, space = query[len(prefix):].strip(), sp
            break
    if not query:
        return {"items": [], "total": 0, "has_more": False, "status": "empty"}
    primary = sources.primary
    components, results = {}, {}
    for component in (("look", "sound") if space == "both" else (space,)):
        space_name = sources.roles["semantic" if component == "look" else "sound"]
        floor = min_sim if component == "look" else sound_min_sim
        try:
            windows = window_matrix(sources.spaces, space_name)
            vector = encode_query(sources.encoder, space_name, query, None if windows is None else windows[2].shape[1])
            if vector is None or windows is None:
                components[component] = {"status": "no-feature"}
                continue
            rows, count = search_windows(vector, windows, floor, mean_weight, None)
            results[component] = [row for row in rows if row[4][0] == primary]
            components[component] = {"status": "ok" if count else "no-feature"}
        except Exception:
            components[component] = {"status": "unavailable"}
    if not results:
        code = "encoder_or_features_unavailable" if any(c["status"] == "unavailable" for c in components.values()) else None
        return {"items": [], "total": 0, "has_more": False,
                "status": "unavailable" if code else "no-feature", "components": components,
                **({"error_code": code} if code else {})}
    scored = fuse(results["look"], results["sound"], None) if len(results) == 2 else next(iter(results.values()))
    excluded = {(primary, sid) for sid in context["intent"][f"exclude_{primary}_ids"]}
    allowed = set(eligible_ranked_items((row[4] for row in scored), context["eligible_ids"], excluded, {}))
    scored = [row for row in scored if row[4] in allowed]
    if scored:
        try:
            for end in range(D_POOL, len(scored) + D_POOL, D_POOL):
                candidates = scored[:end]
                inventory = sources.fingerprint([row[4] for row in candidates])
                kept = set(eligible_ranked_items((row[4] for row in candidates),
                                                 {primary: {sid for kind, sid in inventory["present"] if kind == primary}},
                                                 excluded, inventory["groups"]))
                selected = [row for row in candidates if row[4] in kept]
                if len(selected) >= D_POOL or end >= len(scored):
                    scored = selected[:D_POOL]
                    break
        except Exception:
            return {"items": [], "total": 0, "has_more": False, "status": "unavailable",
                    "error_code": "catalog_unavailable", "components": components}
    taste_sims = {}
    if personalization > 0 and scored:
        semantic = sources.roles["semantic"]
        tvec = taste_prior(sources, semantic)
        seeds = [(primary, sid) for sid in context["intent"][f"seed_{primary}_ids"]]
        if seeds:
            means = sources.means(semantic, seeds)
            if means:
                ids = list(means)
                tvec = seed_query(tvec, np.stack([means[k] for k in ids]), {k: i for i, k in enumerate(ids)}, ids)
        if tvec is not None:
            pool = [row[4] for row in scored]
            means = sources.means(semantic, pool)
            taste_sims = {key: float(means[key] @ tvec) for key in pool if key in means}
        tags = context["intent"]["tag_ids"]
        if tags:
            features = sources.catalog.features([row[4] for row in scored])
            for key, feature in features.items():
                values = feature.get("tag_seconds") or {}
                share = sum(values.get(tag, 0) for tag in tags) / max(sum(values.values()), 1e-8)
                taste_sims[key] = (taste_sims.get(key, 0.0) + share) / 2
        scored = rerank_query_bands(scored, taste_sims, personalization)
    try:
        rows = sources.rows([row[4] for row in scored])
    except Exception:
        return {"items": [], "total": 0, "has_more": False, "status": "unavailable",
                "error_code": "catalog_unavailable", "components": components}
    out = []
    for score, mx, mean_s, best_t, key in scored:
        row = rows.get(key)
        if not row or not row.get("files"):
            continue
        out.append({"kind": key[0], "id": key[1], "title": row.get("title"), "media_url": row.get("media_url"),
                    "duration_s": row.get("duration_s"), "score": round(float(score), 6),
                    "search": {"space": space, "query_score": round(float(score), 6), "best_window_similarity": round(mx, 4),
                               "item_mean_similarity": round(mean_s, 4), "best_t": round(float(best_t), 1),
                               "taste_similarity": round(taste_sims[key], 4) if key in taste_sims else None}})
    page, total, has_more = page_items(out, offset=offset, limit=limit or len(out))
    partial = any(value["status"] != "ok" for value in components.values())
    return {"items": page, "total": total, "has_more": has_more,
            "status": "partial" if partial else ("ok" if page else "empty"), "components": components,
            "request_id": context["request_id"], "error_code": "search_component_unavailable" if partial else None}


# -------------------------------------------------------------------- similar

def cooldown(days, watched_s, duration, *, cooldown_days, recovery_days):
    """0 while resting, then a linear fade back to 1.0; finished items rest half."""
    if cooldown_days <= 0:
        return 1.0
    effective = cooldown_days
    if duration > 0 and (watched_s / duration) >= 0.6:
        effective = cooldown_days * 0.5
    if days < effective:
        return 0.0
    if recovery_days <= 0:
        return 1.0
    return min(1.0, (days - effective) / recovery_days)


def _row_duration(row):
    """One duration per item: the catalog's, never a sum over files."""
    return float((row or {}).get("duration_s") or 0.0)


def similar(sources: Sources, *, context=None, seed_ids=(), config=None, offset=0, limit=20, resolve_eligibility=None,
            corpus_features=None):
    """More like this: length-invariant tag shares blended with mean-vector similarity,
    trusted shared-contributor boosts, cooldown, MMR, eligibility and duplicate rechecks.

    corpus_features: {key: feature row} for the primary catalog; when None the
    catalog is enumerated and ``features`` read for every primary item."""
    cfg = dict(config or {})
    primary = sources.primary
    request_context = parse_context(context or {}, seed_ids=seed_ids, decode=json.loads,
                                    resolve_eligibility=resolve_eligibility, kinds=sources.kinds)
    seed_tag_limit = int(cfg.get("seed_tag_limit", D_SEED_TAGS))
    pool_size = int(cfg.get("candidate_pool", D_SIMILAR_POOL))
    bodyparts_weight = float(cfg.get("bodyparts_weight", D_BODYPARTS_WEIGHT))
    max_tag_share = float(cfg.get("max_tag_share", D_MAX_TAG_SHARE))
    embed_weight = float(cfg.get("embed_weight", D_EMBED_WEIGHT))
    sem_share = float(cfg.get("semantic_share", D_SEM_SHARE))
    audio_weight = float(cfg.get("audio_weight", D_AUDIO_WEIGHT))
    mix_weight = float(cfg.get("mix_weight", D_MIX_WEIGHT))
    face_weight = float(cfg.get("contributor_weight", D_FACE_WEIGHT))
    cooldown_days = float(cfg.get("cooldown_days", D_COOLDOWN_DAYS))
    recovery_days = float(cfg.get("recovery_days", D_RECOVERY_DAYS))
    diversity = float(cfg.get("diversity", D_DIVERSITY))
    seed_field = f"seed_{primary}_ids"
    seeds = [(primary, sid) for sid in request_context["intent"][seed_field]]
    intent_tags = request_context["intent"]["tag_ids"]
    if not seeds and not intent_tags:
        return {"items": [], "total": 0, "has_more": False}
    roles = sources.roles
    revisions = {"features": tuple((space, sources.spaces.revision(space)) for space in sorted(sources.spaces.spaces())),
                 "watch": "disabled"}
    if cooldown_days > 0:
        try:
            revisions["watch"] = sources.signals.read()["observed_at"]
        except Exception:
            revisions["watch"] = None
    generating_config = dict(seed_tag_limit=seed_tag_limit, candidate_pool=pool_size, bodyparts_weight=bodyparts_weight,
                             max_tag_share=max_tag_share, embed_weight=embed_weight, semantic_share=sem_share,
                             audio_weight=audio_weight, mix_weight=mix_weight, contributor_weight=face_weight,
                             cooldown_days=cooldown_days, recovery_days=recovery_days, diversity=diversity)
    provenance = json.loads(json.dumps({
        "ranking_revision": SIMILAR_REVISION, "config": generating_config,
        "config_hash": catalog_module.digest(generating_config), "revisions": {**revisions, "features": list(revisions["features"])},
        "captured_at": sources.clock(), "seed": 0, "experiment": None,
        "intent": request_context["intent"], "eligible_ids": request_context["eligible_ids"],
        "filter_identity": request_context["filter_identity"]}, sort_keys=True, allow_nan=False, default=str))
    # complete corpus facts: coverage and categories for every primary item
    if corpus_features is None:
        corpus_rows = catalog_module.read_catalog(sources.catalog, kinds=(primary,))
        corpus_features = sources.catalog.features([(r["kind"], r["id"]) for r in corpus_rows])
    coverage = {key: {int(t): float(v) for t, v in (f.get("tag_seconds") or {}).items() if v and v > 0}
                for key, f in corpus_features.items() if key[0] == primary}
    cats = {}
    for f in corpus_features.values():
        cats.update({int(t): str(c).lower() for t, c in (f.get("tag_categories") or {}).items()})
    df = defaultdict(int)
    for tags in coverage.values():
        for tag in tags:
            df[tag] += 1
    corpus = max(sum(bool(tags) for tags in coverage.values()), 1)  # items with positive coverage, as the source counts
    idf_of = lambda t: math.log(1.0 + corpus / max(df.get(t, 1), 1))
    # 1. seed content vector: durations merge, shares average
    seed_rows = sources.rows(seeds)
    seed_duration = sum(_row_duration(seed_rows.get(k)) for k in seeds)
    merged = defaultdict(float)
    for key in seeds:
        for tag, secs in coverage.get(key, {}).items():
            merged[tag] += secs
    for tag in intent_tags:
        merged[int(tag)] += 60.0
    idf = {t: idf_of(t) for t in merged}
    seed_vec = share_vector(merged, seed_duration, idf, cats, bodyparts_weight=bodyparts_weight, max_tag_share=max_tag_share)
    look_share = max(0.0, 1.0 - audio_weight - mix_weight)
    active_models = {roles["visual"]: (1 - sem_share) * look_share, roles["semantic"]: sem_share * look_share,
                     roles["voice"]: audio_weight, roles["sound"]: mix_weight}
    embeds = {}
    if embed_weight > 0:
        for space in sources.spaces.spaces():
            if active_models.get(space, 0) > 0:
                for key, vector in sources.means(space, seeds).items():
                    embeds.setdefault(key, {})[space] = vector
    seed_embed = {m: v for m, v in mean_seed_vectors(seeds, embeds).items() if active_models.get(m, 0) > 0}
    source_availability = {"tags": bool(seed_vec), "look": any(m in seed_embed for m in (roles["visual"], roles["semantic"])),
                           "voice": roles["voice"] in seed_embed, "sound": roles["sound"] in seed_embed}
    idents = sources.trusted_links() if face_weight > 0 else {}
    seed_idents = set().union(*(idents.get(k, set()) for k in seeds)) if seeds else set()
    # 2. recruit: strongest seed tags select ids; complete vectors score
    top_tags = [t for t, _ in sorted(seed_vec.items(), key=lambda kv: (-kv[1], kv[0]))][:seed_tag_limit]
    excluded = set(seeds) | {(primary, sid) for sid in request_context["intent"][f"exclude_{primary}_ids"]}
    allowed = request_context["eligible_ids"][primary]
    allowed_set = None if allowed is None else set(allowed)
    cand_vecs = defaultdict(dict)
    if allowed_set != set():
        for tag in top_tags:
            rows = sorted(((coverage[k][tag], k) for k in coverage if tag in coverage[k] and k not in excluded
                           and (allowed_set is None or k[1] in allowed_set)), key=lambda r: (-r[0], r[1]))[:D_POOL_PER_TAG]
            for secs, key in rows:
                cand_vecs[key][tag] = secs
    rough = sorted(cand_vecs.items(), key=lambda kv: (-sum(min(s, 600.0) * idf.get(t, 1.0) *
                   (bodyparts_weight if cats.get(t) == "bodyparts" else 1.0) for t, s in kv[1].items()), kv[0]))[:pool_size]
    source_orders = {"tags": [key for key, _ in rough]}
    vector_budget = max(40, pool_size // 4)
    if seed_embed and allowed_set != set():
        for space in sorted(seed_embed):
            loaded = sources.spaces.matrix(space)
            if loaded is None:
                continue
            index, matrix = loaded
            query = np.asarray(seed_embed[space], dtype=np.float32)
            ranked = []
            for start in range(0, len(index), 2048):
                batch_keys = list(index[start:start + 2048])
                block = np.asarray(matrix[start:start + 2048], dtype=np.float32)
                keep, vectors = [], []
                for i, key in enumerate(batch_keys):
                    if key[0] != primary or key in excluded or (allowed_set is not None and key[1] not in allowed_set):
                        continue
                    vector = block[i]
                    norm = float(np.linalg.norm(vector))
                    if vector.shape != query.shape or not np.isfinite(vector).all() or norm <= 0:
                        continue
                    keep.append(key)
                    vectors.append((vector / norm).astype(np.float16))
                if not keep:
                    continue
                scores = chunked_dot(np.stack(vectors), query)
                ranked = sorted(ranked + [(float(s), key) for key, s in zip(keep, scores) if s > 0],
                                key=lambda item: (-item[0], item[1]))[:vector_budget]
            source_orders[space] = [key for _s, key in ranked]
    if seed_idents:
        source_orders["identity"] = sorted((k for k in idents if k[0] == primary and seed_idents & idents[k]),
                                           key=lambda k: (-face_overlap(seed_idents, idents[k]), k))
    sources_admitted, pool = admit_sources(source_orders, {name: pool_size if name == "tags" else vector_budget for name in source_orders},
                                           {primary: allowed_set}, excluded, {})
    pool_keys = list(pool)
    pool_rows = sources.rows(pool_keys)
    durations = {k: _row_duration(pool_rows.get(k)) for k in pool_keys}
    for tag in {t for k in pool_keys for t in coverage.get(k, {})} - set(idf):
        idf[tag] = idf_of(tag)
    if seed_embed:
        for space in seed_embed:
            for key, vector in sources.means(space, pool_keys).items():
                embeds.setdefault(key, {})[space] = vector
    # 3. score
    watch = {}
    if cooldown_days > 0:
        try:
            watch = {k: w for k, w in watch_rows(sources.signals.read(pool_keys)["rows"], now=sources.clock()).items()}
        except Exception:
            watch = {}
    comps, details = [], {}
    for key in pool_keys:
        if not pool_rows.get(key):
            continue
        cvec = share_vector(coverage.get(key, {}), durations.get(key, 0.0), idf, cats,
                            bodyparts_weight=bodyparts_weight, max_tag_share=max_tag_share)
        contributions = []
        tag_sim = cosine(seed_vec, cvec, contributions=contributions)
        emb_sim = None
        if embed_weight > 0 and seed_embed:
            emb_sim = embedding_similarity({{roles["visual"]: "visual", roles["semantic"]: "semvisual", roles["voice"]: "audioembed",
                                             roles["sound"]: "audiomix"}[m]: v for m, v in seed_embed.items()},
                                           {{roles["visual"]: "visual", roles["semantic"]: "semvisual", roles["voice"]: "audioembed",
                                             roles["sound"]: "audiomix"}.get(m, m): v for m, v in (embeds.get(key) or {}).items()},
                                           sem_share, audio_weight, mix_weight)
        face_sim = face_overlap(seed_idents, idents.get(key, set())) if face_weight > 0 else None
        cd = 1.0
        w = watch.get(key)
        if w:
            cd = cooldown(w["days"], w["watched_s"], durations.get(key, 0.0), cooldown_days=cooldown_days, recovery_days=recovery_days)
        comps.append((key, cvec, tag_sim, emb_sim, face_sim, cd))
        details[key] = {"source_availability": source_availability,
                        "sources": [name for name, ids in sources_admitted.items() if key in ids],
                        "tag_contributions": contributions, "contribution_basis": "tag_similarity",
                        "tag_similarity": round(tag_sim, 4), "embed_similarity": None if emb_sim is None else round(emb_sim, 4),
                        "cooldown_multiplier": round(cd, 3), "contributor_overlap": None if face_sim is None else round(face_sim, 3),
                        "duration_s": round(durations.get(key, 0.0), 1)}
    scored = similar_components(comps, embed_weight=embed_weight, face_weight=face_weight, details=details)
    pin = sources.fingerprint(set(pool_keys) | set(seeds))
    eligible = set(eligible_ranked_items([row[1] for row in scored], {primary: allowed_set},
                                         excluded | (set(pool_keys) - pin["present"]), pin["groups"], already_selected=seeds))
    scored = [row for row in scored if row[1] in eligible]
    # 4. selection
    want = max(pool_size, len(scored))
    chosen = select(scored, want=want, diversity=diversity, calibration=0, target_shares={}, details=details)
    score_by_key = {sid: rel for rel, sid, _v, _cat in scored}
    if not chosen:
        resting = {k for k, w in watch.items() if cooldown(w["days"], w["watched_s"], durations.get(k, 0.0),
                                                            cooldown_days=cooldown_days, recovery_days=recovery_days) <= 0}
        universe = [(primary, sid) for sid in allowed] if allowed is not None else \
            [(r["kind"], r["id"]) for r in catalog_module.read_catalog(sources.catalog, kinds=(primary,))]
        fallback = sorted(set(universe) - excluded - resting)[:want]
        pool_rows.update(sources.rows(fallback))
        chosen = [k for k in fallback if pool_rows.get(k)]
        for key in chosen:
            score_by_key[key] = 0.0
            details[key] = {"sources": ["fallback"], "fallback": "no_positive_feature_match", "strength": "weak",
                            "source_availability": source_availability,
                            "fallback_reasons": ["no_" + s + "_seed" for s, ok in source_availability.items() if not ok],
                            "score": 0.0, "tag_contributions": [], "selection": {"method": "eligible_fallback"}}
    pin = sources.fingerprint(set(chosen) | set(seeds))
    chosen = eligible_ranked_items(chosen, {primary: allowed_set}, excluded | (set(chosen) - pin["present"]),
                                   pin["groups"], already_selected=seeds)
    provenance["revisions"]["catalog_fingerprints"] = pin["revision"]
    out = []
    for source_rank, key in enumerate(chosen):
        row = pool_rows.get(key)
        if not row:
            continue
        top = sorted(details[key].get("tag_contributions", []), key=lambda r: abs(r["contribution"]), reverse=True)[:TOP_CONTRIBS]
        out.append({"kind": key[0], "id": key[1], "title": row.get("title"), "media_url": row.get("media_url"),
                    "duration_s": row.get("duration_s"), "score": round(score_by_key.get(key, 0.0), 6),
                    "similar": {"source_rank": source_rank, **details.get(key, {}), "seed_ids": [k[1] for k in seeds],
                                "contributors": [{**r, "category": cats.get(r["tag_id"], "other"),
                                                  "duration_s": round(coverage.get(key, {}).get(r["tag_id"], 0.0), 1)} for r in top]}})
    if not out:
        return {"items": [], "total": 0, "has_more": False}
    stable = (sources.fingerprint(pin["keys"]) == pin and all(v is not None for _s, v in revisions["features"])
              and revisions["features"] == tuple((space, sources.spaces.revision(space)) for space in sorted(sources.spaces.spaces()))
              and revisions["watch"] is not None
              and (cooldown_days <= 0 or revisions["watch"] == sources.signals.read()["observed_at"]))
    provenance["revision_status"] = "stable" if stable else "unavailable_or_changed"
    provenance["ranking_generation_id"] = hashlib.sha256(json.dumps(
        [provenance, [(item["kind"], item["id"], item["score"], item["similar"]) for item in out]],
        sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()).hexdigest()
    for item in out:
        item["provenance"] = json.loads(json.dumps(provenance))
    page, total, has_more = page_items(out, offset=offset, limit=limit or len(out))
    return {"items": page, "total": total, "has_more": has_more, "status": "ok" if page else "empty",
            "provenance": provenance}


__all__ = ["Sources", "unit_vector", "encode_query", "window_matrix", "search_windows", "fuse", "rerank_query_bands",
           "taste_prior", "search", "cooldown", "similar", "D_MIN_SIMILARITY", "D_SOUND_MIN_SIM", "D_MEAN_WEIGHT",
           "D_POOL", "D_PERSONALIZATION", "CHUNK"]

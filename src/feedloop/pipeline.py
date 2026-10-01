"""The Engine's native feed preparation (v0.6.0): the rank_page inputs of one generation,
built from the slots in a fixed order.

1. taste profile from Signals, Catalog features and secondary-item events, plus the
   request's temporary tag and seed intent;
2. candidates: items watched within cooldown/6 are hard-excluded, each positive profile
   tag contributes its top items by tag seconds, negative seconds join before the cut;
3. every embedding source (look, voice, sound) owns its own budget, independent of tags;
4. the tag source is bounded on a capped rough score, the sources are admitted together
   and the union is hydrated with full tag coverage;
5. raw scoring components (relevance, similarities, cooldown x view fatigue, contributor
   affinity deviation); every knob-dependent combination happens in rank_page;
6. the secondary lane, explore and control picks, lazy fallback and best windows.

Returned in the prepare_feed shape, so the native path and a host's prepare_feed meet the
same rank_page. Pure over its arguments: no store is written.
"""
from __future__ import annotations

import hashlib

import numpy as np

from feedloop import profiles, ranking, taste

POOL_RANK_CAP_S = 600.0  # a tag's seconds counted by the rough candidate score
FATIGUE_CAP = 10  # view days counted by the secondary lane's fatigue
EXPLORE_MIN_S = 30.0  # seconds outside the profile an explore pick needs
COOLDOWN_HARD_SHARE = 6.0  # items watched within cooldown_days / this are never candidates
NEAREST_LIKE_MIN = 0.8  # look cosine a pick needs to its closest liked item to name it
NEAREST_LIKE_CAP = 200  # most recently watched likes a pick is compared against
LEGACY_LOOK = "means"  # the single look space hosts named before space_roles


def tag_limit(config):
    return config.get("tag_candidate_limit") or max(config["candidate_pool"], 400)


def source_budget(config):
    return config.get("source_budget") or max(40, config["candidate_pool"] // 4)


def _md5_order(ids, seed):
    return sorted(ids, key=lambda i: (hashlib.md5((str(i) + str(seed)).encode()).hexdigest(), i))


def sims_over(m, index, qvec, ids):
    if m is None or qvec is None:
        return {}
    return {i: float(m[index[i]].astype(np.float32) @ qvec) for i in ids if i in index}


NO_VIEW = (None, None, None)


def _unit_view(ids, rows):
    # 2026-09-29 (E-7): the slot allows non-unit finite rows, so similarity would follow row
    # length; rows are scaled to unit once here and zero rows leave the view entirely
    # 2026-09-29 (R2): float32 norms overflow on large finite rows and underflow on tiny ones,
    # so the norm and the division run in float64 and only the unit rows are float32
    rows = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    keep = norms > 0
    if not keep.any():
        return NO_VIEW
    ids = [sid for sid, kept in zip(ids, keep) if kept]
    return ids, (rows[keep] / norms[keep, None]).astype(np.float32), {sid: row for row, sid in enumerate(ids)}


class KindMatrices:
    """Per-kind (ids, unit-row matrix, {id: row}) views of the FeatureSpaces matrices, kept
    while the slot hands back the same array at the same revision."""

    def __init__(self, spaces):
        self.spaces, self._memo = spaces, {}

    def _loaded(self, space):
        return self.spaces.matrix(space) if space in set(self.spaces.spaces()) else None

    def _cached(self, key, arrays, spaces, build):
        # 2026-09-29 (E-8): identity alone misses an in-place rewrite, so the revision is part
        # of the key and nothing is kept when a space reports no revision
        revision = tuple(self.spaces.revision(space) for space in spaces)
        entry = self._memo.get(key)
        if entry is not None and entry[1] == revision and all(a is b for a, b in zip(entry[0], arrays)):
            return entry[2]
        value = build()
        if None in revision:
            self._memo.pop(key, None)
        else:
            self._memo[key] = (arrays, revision, value)
        return value

    def get(self, space, kind):
        loaded = self._loaded(space)
        if loaded is None:
            return NO_VIEW
        keys, m = loaded

        def build():
            positions = [i for i, key in enumerate(keys) if key[0] == kind]
            if not positions:
                return NO_VIEW
            first, last = positions[0], positions[-1]
            sub = m[first:last + 1] if last - first + 1 == len(positions) else m[positions]
            return _unit_view([int(keys[i][1]) for i in positions], sub)
        return self._cached(("space", space, kind), (m,), (space,), build)

    def paired(self, first, second, kind):
        """The look view: each item's unit first-space and second-space rows side by side over
        sqrt(2), the width the paired frame windows are scored in."""
        view = self._paired(first, second, kind)
        # 2026-09-29 (K3): a host with one space named "means" lost its look view when reads went
        # through roles; that space stands in whenever the paired view is unavailable
        if view[0] is None and LEGACY_LOOK not in (first, second) and self._loaded(LEGACY_LOOK) is not None:
            return self.get(LEGACY_LOOK, kind)
        return view

    def _paired(self, first, second, kind):
        a, b = self._loaded(first), self._loaded(second)
        if a is None or b is None:
            return NO_VIEW

        def build():
            ids_a, m_a, _index_a = self.get(first, kind)
            ids_b, m_b, index_b = self.get(second, kind)
            if ids_a is None or ids_b is None:
                return NO_VIEW
            rows_a = [row for row, sid in enumerate(ids_a) if sid in index_b]
            if not rows_a:
                return NO_VIEW
            ids = [ids_a[row] for row in rows_a]
            m = np.concatenate([m_a[rows_a], m_b[[index_b[sid] for sid in ids]]], axis=1) / np.float32(np.sqrt(2.0))
            return ids, m, {sid: row for row, sid in enumerate(ids)}
        return self._cached(("paired", first, second, kind), (a[1], b[1]), (first, second), build)


def prepare(*, context, config, seed, kinds, catalog, signals, rows, features, links, views, now, matrices,
            windows_read, windows_revision, windows_cache, primary, secondary=None):
    from feedloop.engine import best_windows, choose_moment, fallback_ids, random_control  # engine imports this module
    c = config
    intent, eligible = context["intent"], context["eligible_ids"]
    seed_ids = list(intent[f"seed_{primary}_ids"])
    exclude_primary = list(intent.get(f"exclude_{primary}_ids", ()))
    exclude_secondary = list(intent.get(f"exclude_{secondary}_ids", ())) if secondary else []
    allowed = eligible.get(primary)
    half_life, min_watch, history_limit = c["half_life_days"], c["min_watch_seconds"], c["history_limit"]
    profile_tags, pool_size = c["profile_tags"], c["candidate_pool"]
    finished_ratio, abandon_ratio = c["finished_ratio"], c["abandon_ratio"]
    dislike_min_watch, short_watch_ratio = c["dislike_min_watch_seconds"], c["short_watch_ratio"]
    dislike_strength, rating_strength = c["dislike_strength"], c["rating_strength"]
    category_weights, max_tag_share, length_floor = c["category_weights"], c["max_tag_share"], c["length_floor_seconds"]
    cooldown_days, recovery_days = c["cooldown_days"], c["recovery_days"]
    impression_discount, image_events_enabled = c["impression_discount"], bool(c["image_events_enabled"])
    include_secondary = bool(c["include_images"]) and secondary is not None and secondary in kinds
    knobs = {k: float(c[k]) for k in ("embedding_weight", "contributor_affinity_weight", "taste_audio_weight", "taste_mix_weight")}
    experiment = c.get("experiment")
    active = experiment["knob"] if experiment else None
    # 2026-09-29 (E-4): hosts name their spaces through space_roles, so every read goes through
    # the role mapping; the paired look view is built from the visual and semantic roles
    roles = c["vector_spaces"]
    visual_space, semantic_space = roles["visual"], roles["semantic"]
    voice_space, sound_space = roles["voice"], roles["sound"]

    # catalog facts: tag seconds and categories of every primary item, durations of every item
    tags, item_categories = {}, {}
    for key, feature in features.items():
        if key[0] == primary:
            tags[key[1]] = {int(t): float(s) for t, s in (feature.get("tag_seconds") or {}).items() if s and s > 0}
            item_categories[key[1]] = {int(t): str(cat) for t, cat in (feature.get("tag_categories") or {}).items()}
    duration_of = {key: float(row.get("duration_s") or 0.0) for key, row in rows.items()}
    tag_category = {}

    def learn_categories(sid, tag_ids):
        # a tag's category is known once an item carrying it was read, as the host's reads learn it
        for t in tag_ids:
            cat = (item_categories.get(sid, {}).get(t) or "").lower()
            if cat:
                tag_category.setdefault(t, cat)

    corpus = max(sum(1 for v in tags.values() if v), 1)

    def document_stats(tag_ids):
        wanted, df = set(tag_ids), {}
        for vector in tags.values():
            for t in wanted.intersection(vector):
                df[t] = df.get(t, 0) + 1
        return df, corpus

    def weights_for(liked, disliked):
        df, n = document_stats(set(liked) | set(disliked)) if liked else ({}, 1)
        return ranking.weights_from_profiles(liked, disliked, df, n, profile_tags=profile_tags, dislike_strength=dislike_strength)

    # 1. taste profile
    signal_rows = signals["rows"]
    watch = {key[1]: w for key, w in profiles.watch_rows({k: r for k, r in signal_rows.items() if k[0] == primary}, now=now).items()}
    ratings = {k[1]: r["rating"] for k, r in signal_rows.items() if k[0] == primary and r.get("rating") is not None}
    engaged = {k[1]: int(r["engagement_count"]) for k, r in signal_rows.items() if k[0] == primary and int(r.get("engagement_count") or 0) > 0}
    durations = {sid: duration_of[(primary, sid)] for sid in dict.fromkeys([*watch, *ratings, *engaged]) if (primary, sid) in duration_of}
    facts = taste.item_preferences(watch, durations, ratings=ratings, engagement_counts=engaged, min_watch=min_watch,
                                   short_watch_ratio=short_watch_ratio, finished_ratio=finished_ratio, abandon_ratio=abandon_ratio,
                                   dislike_min_watch=dislike_min_watch, rating_strength=rating_strength)
    events = {}
    if image_events_enabled and secondary is not None:
        events = {k: {"rating": r.get("rating"), "engagement_count": int(r.get("engagement_count") or 0)}
                  for k, r in sorted(signal_rows.items()) if k[0] == secondary
                  and (r.get("rating") is not None or int(r.get("engagement_count") or 0) > 0)}
    means_s = matrices.paired(visual_space, semantic_space, secondary) if events else NO_VIEW
    tag_extras, vec_extras = [], []
    if events:
        coverage = {k: features.get(k, {}).get("tag_seconds") or {} for k in events}
        tag_extras, vec_extras, _n = profiles.secondary_event_extras(
            events, coverage, None if means_s[1] is None else (means_s[1], {(secondary, i): r for i, r in means_s[2].items()}),
            rating_strength=rating_strength)
    features_by_id = {sid: features[(primary, sid)] for sid in tags}
    liked, disliked, profile_meta = profiles.build_profiles(
        watch, durations, features_by_id, half_life=half_life, min_watch=min_watch, finished_ratio=finished_ratio,
        ratings=ratings, rating_strength=rating_strength, abandon_ratio=abandon_ratio, history_limit=history_limit,
        dislike_min_watch=dislike_min_watch, short_watch_ratio=short_watch_ratio, engagement_counts=engaged,
        extra_tag_events=tag_extras, facts=facts)
    weights, weight_meta = weights_for(liked, disliked)
    profile_meta.update(weight_meta)
    liked_primary = {sid for sid, fact in facts.items() if fact["is_like"]}
    if seed_ids or intent["tag_ids"]:
        request_liked = {}
        for sid in seed_ids:
            for t, s in tags.get(sid, {}).items():
                request_liked[t] = request_liked.get(t, 0.0) + s
        for t in intent["tag_ids"]:
            request_liked[t] = request_liked.get(t, 0.0) + 60.0
        added, _ = weights_for(request_liked, {})
        weights = dict(weights)
        for t, value in added.items():
            weights[t] = weights.get(t, 0.0) + value
        profile_meta["temporary_intent"] = {f"seed_{primary}_ids": seed_ids, "tag_ids": list(intent["tag_ids"])}

    # 2. candidates: cooldown/6 floor, positive-tag tops, negative seconds before the cut
    hard_exclude = {sid for sid, w in watch.items() if w["days"] < cooldown_days / COOLDOWN_HARD_SHARE}
    hard_exclude.update(exclude_primary)
    # 2026-09-29 (polish C-a): an explicitly disliked item was served again when it had no watch
    # row to rest it; like the secondary lane, it never returns
    hard_exclude.update(sid for sid, fact in facts.items() if fact["is_dislike"] and fact["explicit"])
    hard_exclude.update(seed_ids)
    allowed_set = None if allowed is None else set(allowed)
    # 2026-10-01 (v0.8.4): a just-seen item kept returning, since a qualified view only adds
    # fatigue; one viewed within recent_view_hours, liked or not, stays out of fresh pages. When
    # that would leave fewer open items than the page, the least recently viewed ones stay in.
    recent_s = float(c.get("recent_view_hours") or 0.0) * 3600.0
    if recent_s > 0 and views.get("status") == "ok":
        open_ids = {key[1] for key in catalog["present"] if key[0] == primary and key[1] not in hard_exclude
                    and (allowed_set is None or key[1] in allowed_set)}
        last_view = {}
        for event in views.get("events", ()):
            if event["type"] == "visible" and event["kind"] == primary and event["id"] in open_ids:
                at = ranking._timestamp(event["occurred_at"])
                if now - at < recent_s:
                    last_view[event["id"]] = max(at, last_view.get(event["id"], at))
        shortfall = int(context.get("page_size") or 20) - len(open_ids - set(last_view))
        recent = sorted(last_view, key=lambda sid: (last_view[sid], str(sid)))
        hard_exclude.update(recent[max(shortfall, 0):])
    positive_tags = [t for t, w in weights.items() if w > 0]
    negative_tags = [t for t, w in weights.items() if w < 0]
    vectors = {}
    if positive_tags and allowed != ():
        by_tag = {t: [] for t in dict.fromkeys(positive_tags)}
        for sid, vector in tags.items():
            if sid in hard_exclude or (allowed_set is not None and sid not in allowed_set):
                continue
            for t in by_tag.keys() & vector.keys():
                by_tag[t].append(sid)
        limit = tag_limit(c)
        for t, sids in by_tag.items():
            for sid in sorted(sids, key=lambda sid: (-tags[sid][t], sid))[:limit]:
                vectors.setdefault(sid, {})[t] = tags[sid][t]
                learn_categories(sid, (t,))
    for sid, vector in vectors.items():
        extra = [t for t in negative_tags if t in tags[sid]]
        vector.update((t, tags[sid][t]) for t in extra)
        learn_categories(sid, extra)

    # 3. every embedding source owns its budget, independent of tags
    budget = source_budget(c)
    source_knobs = dict(knobs)
    if experiment:
        source_knobs[active] = max(source_knobs[active], float(experiment["candidate"]))
    history = dict(half_life=half_life, finished_ratio=finished_ratio, abandon_ratio=abandon_ratio, history_limit=history_limit,
                   ratings=ratings, rating_strength=rating_strength, dislike_min_watch=dislike_min_watch,
                   engagement_counts=engaged, facts=facts)

    def taste_query(m, index, extras=None):
        query = profiles.rocchio_over(m, index, watch, durations, extras=extras, **history)
        return ranking.seed_query(query, m, index, seed_ids)

    def look(kind, model):
        _ids, m, index = matrices.get(model, kind)
        return m, index

    def look_scores(qvec, channel_queries, *, kind, item_ids=None):
        ids, m, index = matrices.paired(visual_space, semantic_space, kind)
        if item_ids is not None and kind == primary:
            paired = sims_over(m, index, qvec, item_ids) if qvec is not None else {}
        else:
            paired = dict(zip(ids, map(float, taste.chunked_dot(m, qvec)))) if m is not None and qvec is not None else {}
        channels = []
        for model, query in channel_queries.items():
            ids_c, m_c, index_c = matrices.get(model, kind)
            if m_c is None:
                continue
            channels.append(sims_over(m_c, index_c, query, item_ids) if item_ids is not None
                            else dict(zip(ids_c, map(float, taste.chunked_dot(m_c, query)))))
        return ranking.merge_look_scores(paired, channels)

    def embedding_source(ids, m, query):
        if allowed is not None and not allowed:
            return []
        return ranking.embedding_candidates(ids, m, query, exclude=hard_exclude, eligible_ids=allowed, limit=budget, kind=primary)

    profile_vec, look_queries, profile_audio, profile_mix = None, {}, None, None
    source_orders = {}
    means_ids, means_m, means_index = matrices.paired(visual_space, semantic_space, primary)
    if source_knobs["embedding_weight"] > 0:
        profile_vec = taste_query(means_m, means_index, vec_extras)
        for model in dict.fromkeys((visual_space, semantic_space)):
            m, index = look(primary, model)
            extras = []
            if events:
                _ids, m_s, index_s = matrices.get(model, secondary)
                if m_s is not None:
                    extras = profiles.secondary_event_extras(events, {}, (m_s, {(secondary, i): r for i, r in index_s.items()}),
                                                             rating_strength=rating_strength)[1]
            if m is None and not extras:
                continue
            query = taste_query(m, index, extras)
            if query is not None:
                look_queries[model] = query
        if allowed is not None and not allowed:
            source_orders["visual"] = []
        elif look_queries:
            scores = look_scores(profile_vec, look_queries, kind=primary)
            source_orders["visual"] = [sid for sid in sorted(scores, key=lambda sid: (-scores[sid], sid))
                                       if sid not in hard_exclude and (allowed_set is None or sid in allowed_set)][:budget]
        else:
            source_orders["visual"] = embedding_source(means_ids, means_m, profile_vec)
        audio_ids, audio_m, audio_index = matrices.get(voice_space, primary)
        mix_ids, mix_m, mix_index = matrices.get(sound_space, primary)
        profile_audio = taste_query(audio_m, audio_index)
        profile_mix = taste_query(mix_m, mix_index)
        for source, ids, m, query, weight in (("voice", audio_ids, audio_m, profile_audio, source_knobs["taste_audio_weight"]),
                                              ("sound", mix_ids, mix_m, profile_mix, source_knobs["taste_mix_weight"])):
            if query is not None and weight > 0:
                source_orders[source] = embedding_source(ids, m, query)

    # 4. bound the tag source on the capped rough score, admit, hydrate the union
    def pool_key(vec):
        total = 0.0
        for t, s in vec.items():
            w = weights.get(t, 0.0)
            if w:
                total += w * min(s, POOL_RANK_CAP_S) * category_weights.get(tag_category.get(t), 1.0)
        return total
    rough = sorted(vectors.items(), key=lambda kv: (-pool_key(kv[1]), kv[0]))[:pool_size]
    source_orders = {"tags": [sid for sid, _ in rough], **source_orders}
    sources, pool_keys = ranking.admit_sources(
        {name: [(primary, sid) for sid in ids] for name, ids in source_orders.items()},
        {name: len(ids) for name, ids in source_orders.items()},
        {primary: None if allowed is None else frozenset(allowed)}, {(primary, sid) for sid in hard_exclude}, {})
    pool_ids = [sid for _kind, sid in pool_keys]
    for sid in pool_ids:
        vectors[sid] = dict(tags.get(sid, {}))
        learn_categories(sid, vectors[sid])
    present = {sid for sid in pool_ids if (primary, sid) in duration_of}
    view_counts = views.get("counts", {}) if views.get("status") == "ok" else {}
    penalties = {sid: int(n) for (kind, sid), n in view_counts.items() if kind == primary}
    # 2026-09-29 (skip decay): each delivery since the last watch that never got a view drifts the
    # item down by the same discount; explicitly liked items keep their place
    if impression_discount < 0.999 and views.get("status") == "ok":
        for (kind, sid), delivered in views.get("deliveries", {}).items():
            fact = facts.get(sid) or {}
            if kind != primary or (fact.get("explicit") and fact.get("is_like")):
                continue
            since_watch = (watch.get(sid) or {}).get("days", np.inf) * 86400.0
            skips = sum(1 for at in delivered if now - at < since_watch)
            if skips:
                penalties[sid] = penalties.get(sid, 0) + skips
    # 2026-09-30: view days and skips share one cap, so the two penalties never stack past it
    fatigue = ({sid: impression_discount ** min(n, FATIGUE_CAP) for sid, n in penalties.items()}
               if impression_discount < 0.999 else {})

    # contributor affinity over trusted links, shrunk and capped in rank_page
    aff_map, aff_prior, item_links = {}, 0.0, {}
    if knobs["contributor_affinity_weight"] > 0 or active == "contributor_affinity_weight":
        liked_s = {k[1] for k, e in events.items() if taste.verdict(
            0.0, 0.0, finished_ratio=finished_ratio, abandon_ratio=abandon_ratio, rating=e["rating"],
            rating_strength=rating_strength, engagement_count=e["engagement_count"])[0]}
        seen_s = {k[1] for k, e in events.items() if e["engagement_count"] > 0 or (rating_strength > 0 and e["rating"] is not None)}
        item_links = dict(links.links()) if links is not None else {}
        aff_map, aff_prior = profiles.contributor_affinity(
            item_links, {(primary, sid) for sid in liked_primary}, [(primary, sid) for sid in set(watch)],
            {(secondary, i) for i in liked_s}, [(secondary, i) for i in seen_s], watch={(primary, sid): w for sid, w in watch.items()})
    emb_sims = look_scores(profile_vec, look_queries, kind=primary, item_ids=pool_ids) if (profile_vec is not None or look_queries) and pool_ids else {}
    audio_sims = sims_over(audio_m, audio_index, profile_audio, pool_ids) if profile_audio is not None and pool_ids else {}
    mix_sims = sims_over(mix_m, mix_index, profile_mix, pool_ids) if profile_mix is not None and pool_ids else {}

    # 5. raw scoring components: every knob-dependent combination happens in rank_page
    prelim, contributions = [], {}
    for sid in pool_ids:
        if sid not in present:
            continue
        contributions[sid] = []
        duration = duration_of[(primary, sid)]
        rel = ranking.relevance(vectors[sid], weights, tag_category, duration, category_weights=category_weights,
                                max_tag_share=max_tag_share, length_floor=length_floor, contributions=contributions[sid])
        prelim.append((sid, vectors[sid], duration, rel))
    max_rel = max((p[3] for p in prelim), default=0.0) or 1.0
    comps, details = [], {}
    for sid, vec, duration, rel in prelim:
        w = watch.get(sid)
        cooldown = 1.0
        if w:
            judgeable = profiles.watch_counts(w, duration, min_watch, short_watch_ratio)
            is_dislike = False
            if judgeable:
                _l, is_dislike, _b = taste.verdict(w["watched_s"], duration, finished_ratio=finished_ratio, abandon_ratio=abandon_ratio,
                                                   dislike_min_watch=dislike_min_watch, rating=ratings.get(sid),
                                                   rating_strength=rating_strength, engagement_count=engaged.get(sid))
            cooldown = taste.cooldown_multiplier(w["days"], cooldown_days=cooldown_days, recovery_days=recovery_days,
                                                 judgeable=judgeable, is_dislike=is_dislike)
        pen = fatigue.get(sid, 1.0)
        aff_vals = [aff_map[i]["affinity"] for i in (item_links.get((primary, sid)) or []) if i in aff_map]
        aff_dev = (sum(aff_vals) / len(aff_vals) - aff_prior) if aff_vals else None
        cat = ranking.dominant_category(vec, tag_category, weights, category_weights)
        comps.append(((primary, sid), vec, cat, max(rel, 0.0) / max_rel,
                      emb_sims.get(sid), audio_sims.get(sid), mix_sims.get(sid), cooldown * pen, aff_dev))
        details[(primary, sid)] = {
            "sources": [name for name, ids in sources.items() if (primary, sid) in ids],
            "tag_score": rel, "tag_max": max_rel, "tag_contributions": contributions[sid], "relevance": round(rel, 6),
            "visual_similarity": None if emb_sims.get(sid) is None else round(emb_sims[sid], 4),
            "cooldown_multiplier": round(cooldown, 3), "impression_multiplier": round(pen, 3),
            "duration_s": round(duration, 1),
            "days_since_watch": None if not w or w["days"] > 1e5 else round(w["days"], 1), "dominant_category": cat}

    # the MMR redundancy view: the look view, else the visual space alone
    sim_ids, sim_m, sim_index = (means_ids, means_m, means_index) if means_m is not None else matrices.get(visual_space, primary)

    # 2026-09-29: explanation only, for the Home "Because you watched" shelf: the one liked item
    # a look-sourced pick sits closest to; no score, order or knob reads it
    likes = [sid for sid in sorted(liked_primary, key=lambda sid: ((watch.get(sid) or {}).get("days", np.inf), sid))
             if sim_m is not None and sid in sim_index][:NEAREST_LIKE_CAP]
    looked = [key for key, d in details.items() if likes and "visual" in d["sources"] and key[1] in sim_index]
    if looked:
        cos = sim_m[[sim_index[sid] for _kind, sid in looked]] @ sim_m[[sim_index[sid] for sid in likes]].T
        cos[np.array([sid for _kind, sid in looked])[:, None] == np.array(likes)[None, :]] = -np.inf
        best = cos.argmax(axis=1)
        for row, key in enumerate(looked):
            if cos[row, best[row]] > NEAREST_LIKE_MIN:
                details[key]["nearest_like"] = {"kind": primary, "id": likes[best[row]], "cosine": round(float(cos[row, best[row]]), 4)}
    want = max(pool_size, len(comps))
    target_shares = ranking.category_shares(weights, tag_category, category_weights)

    # 6. secondary lane: request exclusions, disliked items never return, fatigue by view days;
    # items absent from the pinned catalog are dropped after the cut
    secondary_comps = []
    allowed_s = eligible.get(secondary) if include_secondary else ()
    allowed_s = None if allowed_s is None else frozenset(allowed_s)
    disliked_s = {k[1] for k, e in events.items() if taste.verdict(
        0.0, 0.0, finished_ratio=finished_ratio, abandon_ratio=abandon_ratio, rating=e["rating"],
        rating_strength=rating_strength, engagement_count=e["engagement_count"])[1]}
    if include_secondary and (profile_vec is not None or look_queries):
        sims = look_scores(profile_vec, look_queries, kind=secondary)
        for iid in sorted(sims, key=lambda iid: (-sims[iid], iid)):
            if iid in exclude_secondary or (allowed_s is not None and iid not in allowed_s) or iid in disliked_s:
                continue
            vals = [aff_map[i]["affinity"] for i in item_links.get((secondary, iid), ()) if i in aff_map]
            delta = sum(vals) / len(vals) - aff_prior if vals else None
            secondary_comps.append(((secondary, iid), sims[iid], impression_discount ** min(int(view_counts.get((secondary, iid), 0)), FATIGUE_CAP), delta))
            if len(secondary_comps) >= pool_size:
                break
        secondary_comps = [row for row in secondary_comps if row[0] in catalog["present"]]

    resting = {sid for sid, w in watch.items() if taste.cooldown_multiplier(
        w["days"], cooldown_days=cooldown_days, recovery_days=recovery_days,
        judgeable=facts.get(sid, {}).get("watch_eligible", False), is_dislike=facts.get(sid, {}).get("is_dislike", False)) <= 0}
    excluded = [(primary, sid) for sid in sorted(hard_exclude | resting)] + [(secondary, iid) for iid in exclude_secondary]
    explore_ids, control_ids = [], []
    if c["explore_slots"]:
        explore_exclude = hard_exclude | resting | set(pool_ids)
        profile_set = set(weights)

        def open_to(sid):
            return sid not in explore_exclude and (allowed_set is None or sid in allowed_set)
        if want > 0 and allowed != ():
            novel = [sid for sid, v in tags.items() if open_to(sid) and sum(s for t, s in v.items() if t not in profile_set) >= EXPLORE_MIN_S]
            explore_ids = _md5_order(novel, seed)[:want]
        if c["control_rate"]:
            control = random_control([sid for sid, v in tags.items() if v], explore_exclude, eligible_ids=allowed, seed=seed)
            control_ids = [control] if control is not None else []
    fallback_reasons = []
    if not watch and not ratings and not engaged:
        fallback_reasons.append("no_preference_history")
    if not any(value > 0 for value in weights.values()):
        fallback_reasons.append("no_positive_tag_profile")
    if profile_vec is None and not look_queries and profile_audio is None and profile_mix is None:
        fallback_reasons.append("no_embedding_profile")

    def fallback():
        # the whole-catalog scan runs only when selection left nothing
        ids = fallback_ids(hard_exclude | resting, want, eligible_ids=allowed, seed=seed,
                           enumerate_ids=lambda: [key[1] for key in catalog["present"] if key[0] == primary])
        keys = [(primary, sid) for sid in ids if (primary, sid) in duration_of]
        if include_secondary:
            # 2026-09-29 (F-1): with no profile the secondary lane is empty, so a cold start
            # served no images; the fallback offers every requested kind for rank_page to interleave
            iids = fallback_ids(set(exclude_secondary) | disliked_s, want, eligible_ids=allowed_s, seed=seed,
                                enumerate_ids=lambda: [key[1] for key in catalog["present"] if key[0] == secondary])
            keys += [(secondary, iid) for iid in iids if (secondary, iid) in catalog["present"]]
        return keys

    def similarity(keys):
        # cosines of unit look rows, rescaled so the pool's mean pairwise cosine is 0 and identity
        # is 1: embedding spaces are anisotropic (unrelated items often sit near 0.6), and raw
        # cosine made every candidate look redundant with every pick. A key without a row is
        # similar to nothing.
        rows = np.array([sim_index.get(sid, -1) if kind == primary else -1 for kind, sid in keys], dtype=np.int64)
        have = rows >= 0
        cos = sim_m[np.maximum(rows, 0)].astype(np.float64) @ sim_m[np.maximum(rows, 0)].astype(np.float64).T
        pairs = have.sum() * (have.sum() - 1)
        base = min((cos[np.ix_(have, have)].sum() - have.sum()) / pairs, 0.999) if pairs else 0.0
        return np.clip((cos - base) / (1.0 - base), 0.0, 1.0) * np.outer(have, have)

    def windows(keys):
        ids = [key[1] for key in keys if key[0] == primary]
        times = best_windows(profile_vec, ids, read=windows_read, revision=windows_revision, cache=windows_cache) if profile_vec is not None else {}
        return {(primary, sid): choose_moment(moments, seed=seed, sid=sid) for sid, moments in times.items()}

    return {"comps": comps, "image_comps": secondary_comps, "target_shares": target_shares, "explanations": details,
            "admitted": None, "excluded": excluded, "seeds": [(primary, sid) for sid in seed_ids],
            "explore": [(primary, sid) for sid in explore_ids if (primary, sid) in duration_of],
            "control": [(primary, sid) for sid in control_ids if (primary, sid) in duration_of],
            "fallback": fallback, "fallback_reasons": fallback_reasons + ["no_positive_feature_match"],
            "profile": {**profile_meta, "weights": dict(weights)},
            "source_counts": {name: len(ids) for name, ids in sources.items()}, "windows": windows,
            "similarity": similarity if sim_m is not None else None}

"""feedloop: ranking, an append-only evidence ledger and taste math over any item catalog."""
from feedloop.taste import (
    DEFAULT_KINDS, verdict, item_preferences, chunked_dot, eligible_ranked_items, blend_emb,
    team_draft, shannon_entropy, cooldown_multiplier, place_explore, welch_interval,
    tagging_pending, embed_pending,
)
from feedloop.ranking import (
    SUPPORTED_VARIANTS, AUDITED_ADMISSION, VARIANT_CONTRACT, REQUIRED_CONFIG, TRANSIENT_CONFIG,
    parse_context, context_key, admit_sources, embedding_candidates, merge_look_scores,
    share_vector, face_overlap, embedding_similarity, mean_seed_vectors, similar_components,
    seed_query, cosine, weights_from_profiles, relevance, category_shares, dominant_category,
    score_components, affinity_scale, select, ordered_evidence, interleave_kinds, rank,
    audited_admission, fingerprint_groups, rank_page, page_items, compare_admission,
    shared_hard_eligibility,
)
from feedloop.ledger import (
    SCHEMA_VERSION, ATTRIBUTION_WINDOW_S, ATTRIBUTION_POLICY_REVISION, ATTRIBUTION_MIN_ADVANCE_S,
    ContractError, authorize_mutation, initialize_event_store, put_eligibility_snapshot,
    read_eligibility_snapshot, record_session_mapping, watch_capture_cutover,
    imported_watch_captures, import_watch_capture, record_served, record_event,
    begin_sync_capture, finish_sync_capture, attribute_outcomes, advance_attribution,
    read_evidence, read_view_counts, read_capture_readiness, summarize_trials,
    perform_feedback, reconcile_feedback,
)

from feedloop.slots import (
    ItemKey, DEFAULT_SPACE_ROLES, TransportError, AuthorityError, ResponseError, MissingKeys,
    Catalog, Signals, FeatureSpaces, TextEncoder, IdentityLinks, Annotator, item_key, wire_key, parse_wire_key,
)
from feedloop.profiles import (
    TTLCache, feature_cached, like_bar, watch_counts, watch_rows, coverage_for, build_profiles, rocchio_over,
    secondary_event_extras, contributor_affinity, affinity_multiplier, trial_reward,
)
from feedloop.catalog import CatalogMissing, read_catalog, ranking_identity, fingerprint_snapshot
from feedloop.tuning import Tuner, TUNER_REGISTRY, KNOB_DEFAULTS, decide, session_means, entropy
from feedloop.serving import (
    feed_request, feed_intent, feed_eligibility, build_feed, serve_feed, continue_cursor, view_request,
    build_fatigue, build_scorecard,
)
from feedloop.discovery import Sources, search, similar, rerank_query_bands, fuse, search_windows
from feedloop.engine import Engine, initialize_stores, DEFAULT_CONFIG, DEFAULT_ATTRIBUTION

__all__ = [name for name in dir() if not name.startswith("_") and name not in ("taste", "ranking", "ledger", "slots", "profiles", "catalog", "tuning", "serving", "discovery", "engine")]

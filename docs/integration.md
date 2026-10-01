# Integrating feedloop

feedloop ships the recommendation brain: ranking, a profile from watch
history, an append-only evidence ledger, attribution, a self-tuner, a serving
contract and a web client. It does not ship your content, your users'
behavior or your item vectors. To run it you supply three inputs. This
document describes each one as the code accepts it. Field names and types are
exact; when the code leaves something open, this says what the code does
rather than inventing behavior.

The three inputs:

1. Capture: the events that say what a user was shown, what they looked at,
   and what they watched, rated or reacted to.
2. Features: embeddings per item, with revisions.
3. Catalog: the items themselves, their tags and their files.

The engine reads all three through six slots (`feedloop.slots`): `Catalog`,
`Signals`, `FeatureSpaces`, `TextEncoder`, `IdentityLinks` and `Annotator`.
It never opens a host database, socket or file by itself.

## Bring your own capture

Capture is the ledger. The ledger stores one row per event and never infers a
watch from a total. An event is `record_event`:

```python
from feedloop import ledger

event_id = ledger.record_event(ledger_path, event={
    "event_type": "viewed",
    "source": "my-player",
    "source_event_id": "view-0001",
    "session_id": "s1",
    "kind": "video",
    "item_id": 42,
    "occurred_at": 1759000000.0,
    "parent_id": "served-event-id",
    "payload": {"visible_fraction": 0.9, "dwell_ms": 1500, "foreground": True,
                "display_rank": 3, "placement": "feed"},
})
```

### Event fields

`record_event` requires `event_type`, `source`, `source_event_id`,
`session_id`, `kind`, `item_id`, `occurred_at` and `payload`. It accepts the
optional `event_id`, `parent_id`, `request_id` and `corrects_id`.

| Field | Type | Notes |
| --- | --- | --- |
| `event_type` | `str` | `"viewed"` or `"outcome"` only. |
| `source` | `str` | The capturing adapter. `"engine"`, `"feedback"` and `"sync_capture"` are reserved. |
| `source_event_id` | `str` | Your stable id for this event. `(source, source_event_id)` is unique; a replay is rejected as a duplicate, never double-counted. |
| `session_id` | `str` | Your session identity. |
| `kind` | `str` | One of the configured kinds, primary first. Default kinds are `("video", "image")`. |
| `item_id` | `int` | Positive, never a boolean. `(kind, id)` qualifies one item. |
| `occurred_at` | `float` | Unix seconds. Must be at or after the ledger cutover and at or before `received_at`. |
| `payload` | `Mapping` | See below. |
| `parent_id` | `str` | The `served` event this view is of (for `viewed`), or the observed event (for other outcomes). |
| `request_id` | `str` | Set from the parent when omitted. |
| `corrects_id` | `str` | Only for a `correction` outcome. |

`served` rows are not written through `record_event`. The engine writes them
from a feed page (`ledger.record_served`). You do not synthesize them.

### Viewed payload

`viewed` requires `parent_id` naming a `served` event in the same session and
the same item. The payload keys are `visible_fraction`, `dwell_ms`,
`foreground`, `display_rank` and `placement`. The ledger enforces the
qualified-view policy `foreground-60pct-1200ms-v1`: `visible_fraction` is a
number in `[0.6, 1]`, `dwell_ms` is a number `>= 1200`, `foreground` is
`True`, `display_rank` is an integer and `placement` is a string. These are
browser assertions; the ledger validates them and does not independently
verify them.

### Outcome payload

`outcome` requires a `signal` and a `provenance`, plus exactly the fields of
that signal.

| `signal` | Required payload fields |
| --- | --- |
| `watch` | `watched_s_delta` (number `> 0`), `started_at` (number `<= occurred_at`) |
| `rating` | `rating_before`, `rating_after` (each `None` or `0..100`, and they must differ) |
| `engagement` | `engagement_delta` (must be the integer `1`) |
| `correction` | none, and `corrects_id` must be set |

`provenance` is `"confirmed_delta_v1"` or `"unknown"`. A `watch` outcome must
use the primary kind. `"unknown"` provenance is retained but earns no attribution.
These are trusted engine APIs, not browser endpoints: only a source adapter
that can prove a real delta may assert `"confirmed_delta_v1"`. A browser
observation never proves outcome provenance.

### Watch batches

For watch time, prefer a batch over one `record_event` per progress point. A
batch carries a chain of player steps and the ledger credits only proven
monotone deltas. A batch is imported with
`ledger.import_watch_capture(db_path, batch=...)`. The batch requires
`capture_id`, `source_id`, `received_at`, `source_revision`, `status`, `reason`
and `events_json`. `status` is `"committed"` (with `reason` falsy) or
`"quarantined"`; `events_json` is a list of steps. A step has `id`,
`stream_session_id`, `type`, `item_id`, `occurred_at`, `position`, `duration`,
`session_id`, `viewed_event_id`, `previous_event_id`, `playback_rate` and
`canonical_session_id`. `type` is one of `view_start`, `view_progress`,
`view_seek`, `view_pause`, `view_complete`. Each step must reference a
committed `viewed` event (`viewed_event_id`) that already exists and matches
the step's item and session. The first observed progress point is only an
anchor; the ledger never guesses its start or awards time before it. A batch
is idempotent by `(source_id, capture_id)` and content hash.

### Feedback operations

A rating, an engagement count or their undo is an operation, not a raw event.
`ledger.perform_feedback(db_path, operation=..., read_current=..., apply_change=...)`
requires `operation_id`, `kind`, `item_id`, `action` and `session_id`, plus
optional `request_id` and `viewed_event_id`. `action` is `"rating"` (with
`rating100`, which may be `None`), `"engagement"` (increments once) or
`"undo"` (with `undo_of`). The engine calls your `read_current(key)` and
`apply_change(key, change)` callbacks to read and write authoritative state;
neither is called under a SQLite transaction. `read_current` returns
`{"status": "ok", "rating100": None|int, "engagement_count": int}` and
`apply_change` returns the same authoritative fields. An uncertain increment
is never retried and stays owned until resolved.

### How events reach the ledger

Three entry points exist:

- Library calls: `ledger.record_event`, `ledger.import_watch_capture`,
  `ledger.perform_feedback`.
- The engine facade: `Engine.record([...])` accepts `{"type": "viewed",
  "event": {...}}`, `{"type": "watch_capture", "batch": {...}}` and
  `{"type": "feedback", "operation": {...}}`.
- The bundled server: `POST /api/view`, `POST /api/watch` and
  `POST /api/feedback` validate and forward to the same functions. The server
  is for a local deployment; a host that runs its own HTTP surface calls the
  library or the engine directly.

Feedloop owns attribution. An outcome is credited to the `served` page that
preceded it only when the chain is complete (served -> viewed -> outcome, same
session, same item), the session mapping is current, and the outcome falls
inside the attribution window (production `ATTRIBUTION_WINDOW_S = 3600.0`,
policy `production-explicit-w3600-v1`). A correction replaces the original
outcome in attribution. When any link is missing the outcome stays in the
ledger but earns nothing; `read_evidence` reports the reason.

### One synthetic capture example

A user is served item 42, sees it for 1.5 s at 90 percent, then watches 40
seconds and rates it 90. In order:

```python
# 1. served: written by the engine when the page is delivered (not shown here).
served_event_id = "..."

# 2. viewed
viewed_id = ledger.record_event(ledger_path, event={
    "event_type": "viewed", "source": "my-player", "source_event_id": "v1",
    "session_id": "s1", "kind": "video", "item_id": 42,
    "occurred_at": 1759000001.0, "parent_id": served_event_id,
    "payload": {"visible_fraction": 0.9, "dwell_ms": 1500, "foreground": True,
                "display_rank": 3, "placement": "feed"},
})

# 3. watch outcome
ledger.record_event(ledger_path, event={
    "event_type": "outcome", "source": "my-player", "source_event_id": "o1",
    "session_id": "s1", "kind": "video", "item_id": 42,
    "occurred_at": 1759000041.0, "parent_id": viewed_id,
    "payload": {"signal": "watch", "provenance": "confirmed_delta_v1",
                "watched_s_delta": 40.0, "started_at": 1759000001.0},
})

# 4. explicit rating through the feedback callback pair
ledger.perform_feedback(ledger_path, operation={
    "operation_id": "fb1", "kind": "video", "item_id": 42,
    "action": "rating", "session_id": "s1", "rating100": 90,
}, read_current=source.read_current, apply_change=source.apply_change)
```

Ratings trump implicit signals in both directions: a rating `>= 80` forces a
like, `<= 40` forces a dislike, and `41..79` forces neutral regardless of
watch time. One `verdict()` implementation in `feedloop.taste` is used
everywhere. An engagement count above zero also forces a like and multiplies
the boost.

## Bring your own embeddings

Embeddings live in named feature spaces. The engine reads them through the
`FeatureSpaces` slot and never decodes media itself. A space is either
means-only (one vector per item) or timed (mean rows plus per-window rows with
real timestamps).

### The FeatureSpaces contract

```python
class FeatureSpaces(Protocol):
    def spaces(self) -> Sequence[str]: ...
    def matrix(self, space) -> tuple[Sequence[ItemKey], np.ndarray] | None: ...
    def revision(self, space) -> Any | None: ...
    def windows(self, space) -> tuple[Sequence[ItemKey], np.ndarray, np.ndarray] | None: ...
```

- `spaces()` lists the space names you provide.
- `matrix(space)` returns `(id_index, matrix)`. `id_index` is the item keys
  (`(kind, id)` tuples) in row order; `matrix` is a 2-D array with one row per
  key. The matrix must be finite and its width is the space dimension.
- `windows(space)` returns `(item_keys, times_s, matrix)` with one row per time
  window and the same key repeated per window, distinguished by `times_s`, or
  `None` for a means-only space. Search needs windows with real timestamps; a
  means-only space is no-feature for Search while Similar and the Feed still
  use its means.
- `revision(space)` returns the committed revision of the space, or `None`
  when it is unavailable. Any opaque comparable value works.

### Dimensions and revisions

feedloop does not fix a dimension. Each space carries whatever width its
matrix has, and that width must be consistent within the space. Different
spaces may have different dimensions; the engine maps roles onto your space
names with `space_roles`, default
`{"visual": "visual", "semantic": "semvisual", "voice": "audioembed",
"sound": "audiomix"}`. You pick the space names.

A revision is how the engine knows the vectors moved. Before ranking, the
engine reads `spaces.revision(space)` for every space and digests them into a
feature revision. If any space returns `None`, the feature revision is `None`
and dependent cache publication is disabled. A revision change drops memoized
windows and invalidates a frozen page, so a new delivery re-ranks. The revision
must change whenever any vector in the space changes.

### Storage expectations

The bundled filesystem source stores one space as
`<state>/spaces/<space>.npz`. A custom slot need not use NumPy files; it must
return the shapes above. The `.npz` layout, if you copy it, is:

| Array | Meaning |
| --- | --- |
| `keys` | `<kind>:<id>` strings, one per matrix row |
| `matrix` | `float32`, shape `(n_items, dim)` |
| `window_keys` | optional, `<kind>:<id>` per window row |
| `window_times` | optional, `float32` timestamp per window row |
| `window_matrix` | optional, `float32`, shape `(n_windows, dim)` |
| `meta` | one JSON string |

`meta` names `provenance` (a string) and `window_scope` (`"none"` or
`"timed"`), and the writer records `dim`, `items` and a `revision` that is the
digest of the stored arrays and the metadata. Timed windows require real
per-window timestamps; a means-only space must not invent window rows.

### How the engine reads them

During ranking the engine loads the requested space matrices, computes score
components from the query profile, and uses windows only for Search. Row order
is not trusted: the engine aligns rows to keys from `id_index`. A space whose
matrix has the wrong row count, is non-finite, or whose keys do not match the
catalog present set is treated as unavailable. For Similar and the Feed, a
means-only space is enough. For natural-language Search you also supply a
`TextEncoder` whose `encode(space, text)` returns a vector compatible with the
same space.

## Bring your own catalog

The catalog is the set of items the feed can rank. The engine reads it through
the `Catalog` slot. Every read is complete or it raises; a partial page is a
failure, never a shorter catalog.

### Catalog contract

```python
class Catalog(Protocol):
    def enumerate(self, kinds, page, page_size, *, timeout_s=30) -> Mapping[str, Any]: ...
    def fetch(self, keys, *, timeout_s=30) -> Mapping[str, Any]: ...
    def features(self, keys) -> Mapping[ItemKey, Mapping[str, Any]]: ...
    def tag_names(self) -> Mapping[int, str]: ...
```

`enumerate` and `fetch` return `{"items": [...], "total": int,
"change_token": str}`. `enumerate` is one page of one kind, 1-based, with an
exact `total`. `fetch` is a complete keyed read, not paginated, any size. A
missing requested key is reported with `MissingKeys`; a transport failure is
retried, authority and malformed-response failures are not. A fileless item
must appear with `files=[]`.

### What the engine needs per item

Each item row has:

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `int` | Positive. |
| `kind` | `str` | One of the configured kinds. |
| `title` | `str` | Display title. |
| `duration_s` | `float` | Seconds. `0` or less means unknown. |
| `media_url` | `str` | Where the client plays it. |
| `tags` | `list[str]` | Display strings. |
| `contributor_ids` | `list[str]` | Trusted contributor ids; these, not display names, drive affinity. |
| `updated` | opaque token | May move on any metadata touch. Not part of ranking identity. |
| `files` | `list` | `[]` for a fileless item; else `{"fingerprints": [{"type": str, "value": str}, ...]}` per file. |

`features(keys)` returns, per key, `{"tag_seconds": {tag_id: seconds},
"watched_tag_seconds": {...} | None, "tag_categories": {tag_id: category}}`
with positive-integer tag ids. `tag_names()` maps known tag ids to display
names; unknown ids stay unnamed. A key absent from `features` has no features.

Categories are host-defined strings. The `category_weights` config maps a
category name to a multiplier on that category's tag contribution (relevance,
candidate pre-ranking, category shares and the dominant category); an unlisted
category weighs `1.0`, and the default `{}` is neutral. For example
`config={"category_weights": {"featured": 0.5}}` halves the pull of tags in the
`featured` category. Multipliers are finite and non-negative.

### Signals contract

```python
class Signals(Protocol):
    def read(self, keys=None) -> Mapping[str, Any]: ...
```

`read(keys)` returns `{"observed_at": float, "rows": {key: row}}` with the
current ratings, engagement counts and watch history; `None` reads every row.
A row has `rating` (0-100 or `None`), `engagement_count` (int >= 0), `watch`
only when the item was actually watched, and `updated_at`: the epoch seconds
of that item's last rating, engagement or watch change, never later than
`observed_at`.

The tuner judges a past trial from these current rows, so it must know which
rows changed after the trial's evidence cutoff. With `updated_at` on every
row, only the items changed since the cutoff lose their facts. If any returned
row lacks `updated_at`, the engine falls back to the global `observed_at` as
the cutoff: while the host keeps writing, `observed_at` stays newer than every
cutoff and tuner evidence is unavailable, so the tuner cannot promote.

`feedloop.slots.check_signals(signals, keys=None)` reads once and returns a
list of contract problems (missing `updated_at`, `updated_at` after
`observed_at`, non-finite numbers); an empty list means the read conforms.

### Ranking identity

The engine derives ranking identity from item existence plus file
fingerprints only, through `catalog.fingerprint_snapshot`. It performs two
consecutive validated reads and accepts them only when the ranking identity
agrees; a set that keeps changing is an error. The `updated` token and
`change_token` are deliberately excluded, so a rating or a view does not
change the ranking identity. Duplicate groups are derived from the file
fingerprints, so the same file listed twice is served once.

## Reference implementation

`feedloop.sources.filesystem` is the reference implementation of all three
links above: it serves `Catalog`, `Signals` and `FeatureSpaces` over a media
folder, sidecars and SQLite, and the bundled server records capture events
through the ledger. A larger deployment typically has the same shape: a
browser or player plugin captures watch time and forwards JSON events to a
recording endpoint that writes them through the ledger; an offline pipeline
writes one embedding row per item and model (model name, timestamp, vector)
to a database that a `FeatureSpaces` adapter reads back with a revision per
model; and a `Catalog` adapter over the host's media API supplies items,
durations, tags and file fingerprints. The capture layer is then the only
writer of the ledger, and the engine can run as a separate service.

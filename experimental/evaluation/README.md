# Experimental: frozen offline evaluation

**Status: no validated measurement yet.** Nothing here has been run against a
real snapshot in this repository; the harness is the instrument, not a result.

This directory ports the offline evaluator that sits beside the engine. It is
experimental and outside the installed package: import it from a checkout as
`experimental.evaluation`.

What it does:

- Freezes a JSON bundle with `manifest`, `catalog`, `features`, `events` and
  `contexts` sections, each hashed as canonical JSON bytes.
- Pins the production ranking sources by path and SHA-256
  (`src/feedloop/ranking.py`, `taste.py`, `profiles.py`) and executes them
  through a tripwired loader that rejects application imports and I/O.
- Measures ranked lists with observed-label accuracy (recall, NDCG, MAP),
  coverage, novelty, diversity, duplicate exposure and a popularity-decile
  exposure share, and reports admission parity from owner-produced traces.
- Exports prospective snapshots from staged current facts and an optional
  ledger read through `src/feedloop/ledger.py`'s `read_evidence`.

What it refuses to do: substitute scores for unmeasured outcomes. A missing
adapter, an incomplete play-count table, a probe without labels, or an
unpinned source stays `unmeasured`, `unavailable_unknown_counts`,
`unsupported` or a failed validation. Nothing here is a claim about the
engine's quality; it is the instrument for making such a claim later.

CLI: `python -m experimental.evaluation.evaluate --help` (export, validate,
run, judge-export, judge-import). Requires NumPy.

Adaptations from the private evaluator (2026-09-17): module names and paths
above, `threading` added to the pure-import allowlist because `profiles.py`
uses a lock, item kinds `video`/`image`, the engagement signal named
`engagement`/`engagement_delta`, and capture metadata keys
`source_equal_passes`, `video_embeddings`, `image_embeddings`. The offline
ledger reader also allows `feedloop.taste` because the public ledger imports
`DEFAULT_KINDS` from it. No behaviour was changed; no evaluation run is
claimed.

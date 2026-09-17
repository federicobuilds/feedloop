# feedloop

A learning recommendation feed over any item catalog: ranking with explanations, an
append-only evidence ledger, explicit attribution, taste math over tags and feature
spaces, and an autonomous tuner with a reversible ledger. The library defines six
slots (catalog, signals, feature spaces, text encoder, identity links, annotator) that
a host fills; `feedloop.sources.filesystem` fills them over a plain media folder so the
package runs on its own: folder in, learning feed out. Version 0.x: the slot contracts
are proven against the filesystem source only and may still change.

## Quick start

```sh
pip install feedloop            # numpy only
feedloop demo ./my-media        # creates ./my-media/.feedloop, serves on http://127.0.0.1:8765/
```

`demo` scans the folder (videos: mp4, webm, mkv, mov, m4v; images: jpg, jpeg,
png, gif, webp), assigns each file a stable id, reads an optional
`<file>.json` sidecar (`title`, `tags`, `contributors`, `duration_s`,
`tag_seconds`), and opens the client. It prints an address that carries the
shared key for this run in the fragment; mutations (deliveries, views, watch
batches, feedback, tuner controls) need that key and an exact loopback origin.
Reads (observation, search, similar, scorecard) do not.

The demo uses a five second attribution window and a fifteen second tick so a
watch shows up on the Engine page within a minute. `feedloop serve` reuses an
existing state directory with the production policy (a one hour window, six
hour trial ripening) and never creates stores unless you pass `--init`.

### What is measured and what is not

The base install decodes no media. A duration is known only when a sidecar
states it or the browser player reports it; otherwise cards say "duration not
measured". The demo's Search runs over a space named `sidecar_text` whose
vectors hash the sidecar tag words; it is labelled in its metadata as not a
measurement of the media. Text search needs window rows with real timestamps,
so it works only for videos whose sidecar supplies timed `segments`
(`[{"start_s": 30, "tags": ["harbor"]}]`, the author's own positions); without
segments the space has mean vectors only and Search honestly reports
no-feature. `feedloop demo --fixture-sidecars` writes labelled synthetic
sidecars with such segments for files that have none. Learned spaces come from the optional
extractors:

```sh
pip install 'feedloop[extract]'    # torch, open_clip_torch, pillow, transformers, soundfile
feedloop extract ./my-media --kind visual --model ViT-B-32 --vocabulary tags.txt
feedloop extract ./my-media --kind audio --model laion/clap-htsat-unfused
```

The visual extractor embeds image files (video frames only through a caller
supplied frame sampler) and writes zero-shot tags from your vocabulary to
`<file>.generated.json`, never into your own sidecar; an existing generated
file is kept unless you pass `--overwrite`, and even then your own fields and
non-generated tags inside it survive. Zero-shot tags are weaker
than a trained tagger. The audio extractor decodes with `soundfile` (audio
files, not video containers) and doubles as the text encoder for sound search.
Neither extractor has been run against real models in this repository yet;
their contracts are tested with controlled doubles.

### Client

`web/` is a no-build ES module client: Feed (scroll-snap column), Home
(shelves with Load more), Search, Similar, and the Engine page with the real
scorecard, the current experiment and the reversible tuner ledger. Every card
explains itself in words; an absent signal reads "not measured", never zero.

## How the engine learns

**The six slots and the Engine.** A host fills protocols from `feedloop.slots`:
`Catalog` (items, files, fingerprints, tag features), `Signals` (ratings,
engagement counts, watch history as of a known time), `FeatureSpaces` (mean
vectors per item, optionally timed window rows), `TextEncoder` (text to a
vector in a named space), `IdentityLinks` (trusted shared-contributor links,
optional) and `Annotator` (tag writes, optional and never called on a read
path). Ratings and engagement change only through two callbacks the Engine is
constructed with. The filesystem source fills the first three and supplies the
callbacks:

```python
import time

from feedloop import Engine, initialize_stores
from feedloop.sources.filesystem import FilesystemSource, TextHashEncoder

source = FilesystemSource("./my-media", "./my-media/.feedloop")
initialize_stores(ledger_path="./my-media/.feedloop/ledger.sqlite",
                  tuner_path="./my-media/.feedloop/tuner.sqlite", cutover_ts=time.time())
engine = Engine(catalog=source, signals=source, spaces=source, encoder=TextHashEncoder(["sidecar_text"]),
                ledger_path="./my-media/.feedloop/ledger.sqlite", tuner_path="./my-media/.feedloop/tuner.sqlite",
                read_current=source.read_current, apply_change=source.apply_change,
                space_roles={"visual": "visual", "semantic": "sidecar_text", "voice": "audioembed", "sound": "audiomix"})
page = engine.feed({"limit": 24, "images": True, "surface": "feed", "session_id": "s1",
                    "request_id": "r1", "client_request_id": "c1"})
```

**Delivery, qualified view, outcome.** A delivery (impression) is one served
page, journaled with its ranking generation; it proves nothing about attention.
A qualified view is a card that stayed at least 60 percent visible for 1,200 ms
in a foreground tab, posted by the client and journaled against the delivery.
An outcome is evidence that follows a qualified view: watched seconds from a
committed player batch (validated for continuity and playback speed before the
source stores anything), an explicit rating, or an engagement count. Ratings
trump implicit signals in both directions; the browser never asserts an outcome
by itself.

**Attribution window and tuner ripening.** Outcomes are credited to the
delivery that preceded them only after the attribution window closes
(production: 3,600 s; `demo`: 5 s, labelled `demo-explicit-w5-v1`). Trials then
ripen for a further six hours before the tuner may count them. Both gates are
visible on the Engine page; the demo shortens only the first.

**Explicit ratings and reversible feedback.** Like, dislike, clear and count
engagement are ledgered operations: the engine journals, then asks the
authority to apply, then confirms with the authority's values. Undo is a new
operation that restores the exact original value; a lost reply leaves the
operation unresolved, blocks further actions on that item, and offers "Check
status", which reconciles against the ledger and keeps the receipt so Undo still
works.

**Self-grading promotion with undo.** The tuner runs one experiment at a time
(base versus candidate value of one knob). Once each arm has enough ripened
trials across enough sessions, and the difference is significant and does not
collapse category diversity, it promotes the winner by itself and rotates to the
next knob; otherwise it stalls or waits. Every move is a ledger row with its
evidence and can be reverted from the Engine page; reset restores the standard
values. Knob values are not enjoyment probabilities.

**Cursor-stable pagination.** A page carries a cursor bound to its ranking
generation. When the generation is gone (a restart, changed features, a
changed catalog) the server refuses the cursor explicitly; the client restarts
once from the top with fresh delivery identities, keeps the cards it has, drops
duplicates, and re-arms that single restart only when new content arrives.
Repeated refusals become a Retry, never a loop.

**Bring your own models.** Feature spaces are files; any extractor that writes
mean vectors (and, with real timestamps, window rows) into a named space plugs
in, and `space_roles` maps the visual, semantic, voice and sound roles onto
your names. Text search needs an encoder for the same space. Zero-shot tags are
weaker than a trained tagger; unavailable features are reported as unavailable,
not zero.

**v0.x limitations.** The slot contracts are proven against the filesystem
source only and may change before 1.0. Text search requires window rows with
real timestamps (sidecar segments or a timed extractor); a means-only space is
no-feature for Search while Similar and the Feed still use its means. Video
frame sampling and container audio decoding are not shipped.

## Evaluation

`experimental/evaluation/` ports an offline evaluator (frozen snapshots, pinned
ranking sources, observed-label metrics). It is outside the public API and the
test gates, and it carries the label **no validated measurement yet**: nothing
in this repository measures the engine's quality, causal lift or ranking
superiority. See `experimental/evaluation/README.md`.

## License

`LICENSE` (GNU Affero General Public License v3.0 or later, AGPL-3.0-or-later)
governs use; `COMMERCIAL-LICENSE.md` describes the commercial option for
organizations that cannot comply with the AGPL; `CLA.md` is the contributor
license agreement.

## Repository layout

```
src/feedloop/            the package (numpy only)
  sources/filesystem.py  Catalog, Signals and FeatureSpaces over a folder, sidecars and SQLite
  extractors/            optional visual (open CLIP) and audio (CLAP) extractors
  server/app.py          standard-library HTTP server
  cli.py                 feedloop demo | serve | extract
web/                     the client, packaged with the wheel
scripts/verify_demo.py   portable end-to-end demonstration runner (real server, labelled fixtures)
scripts/check_client.py  Playwright browser contract runner
experimental/evaluation  frozen offline evaluator (not installed; see its README)
tests/                   pure unit tests
```


# feedloop

feedloop is a self-contained recommendation brain. It ranks items, learns a
taste profile from what a user was shown, looked at, watched, rated and
skipped, journals every event to an append-only evidence ledger, attributes
outcomes to deliveries, tunes its own knobs one at a time, and serves a
standard-library web client. Everything it learns can be reverted.

It does not ship your content, your users' behavior or your item vectors.
feedloop supplies the engine; you supply capture, embeddings and a catalog.
[docs/integration.md](docs/integration.md) describes those three inputs as the
code accepts them.

Point it at media and it runs on its own: `feedloop demo` scans a folder of
videos and images, gives each file a stable id, reads an optional sidecar, and
serves a local web client with a feed, search, similar items and an Engine page
that shows the evidence behind each decision. State lives in an append-only
ledger next to your files.

## What you must bring yourself

To run feedloop against a real library, three inputs are yours to provide.
Each is a slot the engine reads; the bundled filesystem source fills all three
for a plain folder of files, and any host can fill them instead.

- Capture. The events that say what was shown, looked at, watched, rated and
  reacted to. feedloop validates and stores them; it cannot invent user
  activity. See [capture](docs/integration.md#bring-your-own-capture).
- Embeddings. Item vectors in named feature spaces, with a revision per space.
  The base install decodes no media, so learned vectors come from your own
  encoder, not from feedloop. See
  [embeddings](docs/integration.md#bring-your-own-embeddings).
- Catalog. The items, their durations, tags and file fingerprints. See
  [catalog](docs/integration.md#bring-your-own-catalog).

The bundled filesystem source derives all three from a folder and optional
sidecars, which is enough for a demo and for media you can describe with
sidecars. A host with its own database fills the six `feedloop.slots`
protocols instead.

## Setup

The package is not on PyPI. Install it from GitHub. The base install depends on
numpy only. `numpy` is required; tests also need `pytest`.

```sh
git clone https://github.com/federicobuilds/feedloop
cd feedloop
python3.11 -m venv .venv && . .venv/bin/activate
# Windows PowerShell instead of the line above: py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e .
```

Run the test suite:

```sh
pip install pytest
.venv/bin/python -m pytest tests -q
```

These are pure unit tests over the engine, the ledger, the serving contract and
the client. They need no backend and no media.

To serve the UI, point `demo` at a folder that holds some mp4, webm, mkv, mov,
m4v, jpg, jpeg, png, gif or webp files. Replace `./my-media` with your folder.
`--fixture-sidecars` writes labelled synthetic sidecars, with timed segments
for videos, for files that have none; it never touches an existing sidecar.

```sh
feedloop demo ./my-media --fixture-sidecars
```

For a UI smoke test with no real media, create a folder of small valid images
and run the same command; images are first-class items and need no duration.

```sh
python3 - <<'PY'
import base64, pathlib
png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
d = pathlib.Path("smoke"); d.mkdir(exist_ok=True)
for i in range(6):
    (d / f"still-{i}.png").write_bytes(png)
PY
feedloop demo ./smoke --fixture-sidecars
```

`demo` prints two addresses. Open the second one,
`http://127.0.0.1:8765/#key=...`. The fragment carries the key that mutations
(deliveries, views, watch batches, ratings, tuner controls) need. Reads work
without it.

## The interface

`demo` scans the folder, gives each file a stable id, reads an optional
`<file>.json` sidecar, and serves on http://127.0.0.1:8765/. The interface is
plain HTML, one stylesheet and a few ES modules under `web/`, served by
feedloop's own server. It has no build step, loads nothing from the network,
and works offline. It calls only the server's public routes (`/api/feed`,
`/api/view`, `/api/watch`, `/api/feedback`, `/api/search`, `/api/similar`,
`/api/scorecard`, `/api/config`, the tuner controls) and the media under
`/media/`.

- Feed: one full-height item at a time, with a black stage for the player and
  a side panel. A video opens at its matching moment and plays automatically, muted until you turn sound on with M or the sound button; that choice is remembered. The next video is preloaded and videos loop. The panel shows the
  title, an evidence strip (one segment per signal, sized by contribution;
  an unmeasured signal is hatched, never zero), a reason line in words, and
  one metadata line. Like and Dislike are toggles; pressing an active one
  clears the rating, with Undo next to the confirmation. A More disclosure
  holds Clear rating and Count engagement, each explained in one line. Next
  moves on; More like this opens Similar. Move with the arrow keys or J and
  K, rate with L and D. Space or a tap pauses, and a video that scrolls out of
  view pauses.
- Home: horizontal shelves named for why their picks were chosen: Continue watching, Because you watched <title>,
  Because you like <tag>, Something new, New to you, and category shelves
  otherwise. Each shelf shows item counts, Load more, and a skeleton while
  loading. Hovering a card plays a
  muted preview after a short delay. Clicking it turns the card into the
  player. Only that player counts as watching; the preview does not. A
  catalog item may carry `preview_url` (a short silent clip the hover
  preview plays instead) and `open_url` (the host's own page, which a card
  click opens instead of the inline player). A catalog item may carry
  `animated_image: true` when its media is an animated image (GIF or similar):
  cards, Feed and the Similar seed then render an `<img>` that plays natively,
  never a player, and it records no watch time. An
  empty library explains how to add media and sidecars.
- Search: describe a look in words. A Look / Sound / Both control stays in
  sync with a typed `sound:` or `both:` prefix in both directions. The mode
  of the last search is kept in the URL. With fixture sidecars search works
  on videos; an item without segments reports no-feature.
- Similar: more like one video, from tag shares blended with mean-vector
  similarity. It is not a tab: it opens from More like this on a video
  card, with a Back control and the seed's thumbnail, title and Change
  (enter another id). Without a seed it returns to Feed.
- Engine: the dashboard. A summary row shows the taste profile the ranker
  reported with the latest Feed page, attributed verdicts and liked
  outcomes, committed watch outcomes, and whether automatic tuning is on.
  Below it are the evidence gates in plain sentences, the running experiment
  with a like-rate bar for each arm, the knobs, and the reversible tuner
  ledger. Refresh shows when the data last updated and auto-refreshes while
  the tab is visible; Reset knobs asks for confirmation first. The demo uses
  a five second attribution window and a fifteen second tick, so a watch
  shows up here within a minute.

The interface is a quiet, dark console with neutral surfaces so the media
carries the colour. On narrow screens the navigation becomes a bottom tab
bar. Motion is short and switches off under `prefers-reduced-motion`, and
that includes the hover previews.

Stop the server with Ctrl-C. State lives in `<folder>/.feedloop`
(`source.sqlite`, `ledger.sqlite`, `tuner.sqlite`, `spaces/`) unless you pass
`--state DIR`. To serve that state again with the production policy (one hour
window, six hour trial ripening), run `feedloop serve ./my-media`. `serve`
never creates stores unless you pass `--init`.

## Your own media and sidecars

Put a `<file>.json` next to a media file. Every field is optional. Tags can be
plain strings or objects with `name`, `category` and `seconds`. `duration_s` is
the only way the base install knows a duration before the browser player
reports one. `segments` give Search its timestamps and are your own positions,
not measurements.

```json
{
	"title": "Harbor at dawn",
	"tags": ["harbor", {"name": "fog", "category": "weather", "seconds": 40}],
	"contributors": ["alice"],
	"duration_s": 95.0,
	"tag_seconds": {"harbor": 60},
	"segments": [{"start_s": 0, "tags": ["harbor"]}, {"start_s": 30, "tags": ["fog"]}]
}
```

## Models

The base install decodes no media. Learned feature spaces come from the
optional extractors. Install them with `pip install -e '.[extract]'` (torch,
open_clip_torch, pillow, transformers, soundfile, imageio-ffmpeg).
imageio-ffmpeg ships an ffmpeg binary, about 80 MB, so no system ffmpeg is
needed.

| Extractor | `--model` | Weights | Notes |
| --- | --- | --- | --- |
| Visual ([open_clip](https://github.com/mlfoundations/open_clip)) | `ViT-B-32` (same weights as `hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) | 605 MB | Recommended. CPU and MPS are fine. |
| Visual, larger ([open_clip](https://github.com/mlfoundations/open_clip)) | `hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | 1.7 GB | Slower. A GPU helps. |
| Audio ([CLAP](https://huggingface.co/laion/clap-htsat-unfused)) | `laion/clap-htsat-unfused` | 615 MB | Reads the audio track of each video. |

```sh
feedloop extract ./my-media --kind visual --model ViT-B-32 --vocabulary tags.txt
feedloop extract ./my-media --kind audio --model laion/clap-htsat-unfused
```

The first run downloads the weights into the Hugging Face cache,
`~/.cache/huggingface/hub`. A bare open_clip architecture name loads pretrained
weights: `--pretrained TAG` picks the open_clip weights tag, the default for
`ViT-B-32` is `laion2b_s34b_b79k`, and any other bare name gets the first tag
open_clip lists for it (`open_clip.list_pretrained_tags_by_model`). A bare name
open_clip lists no weights for stops with an error that names `--pretrained`.
An `hf-hub:` name carries its own weights and ignores the default. Both
commands take `--space NAME` (default `visual` or `audioembed`), `--device
cpu|cuda|mps` and `--overwrite`. Without `--overwrite` an existing space is
kept. `tags.txt` holds one zero-shot tag per line:

```
harbor
fog
forest
night
crowd
```

The visual extractor embeds image files and writes zero-shot tags from the
vocabulary to `<file>.generated.json`. It never writes into your sidecar, and
with `--overwrite` it replaces only its own fields in the generated file. A
video is embedded as the mean of 8 frames that ffmpeg grabs at evenly spaced
timestamps. The audio extractor decodes each video's audio track with ffmpeg
and doubles as the text encoder for sound search. A file ffmpeg cannot read,
such as an empty file or a video without an audio track, is skipped and named
in the report. Verified with real weights on an M1 Max (`--device mps`).

## How the engine learns

A host fills six protocols from `feedloop.slots`: `Catalog` (items, files,
fingerprints, tag features), `Signals` (ratings, engagement counts, watch
history as of a known time), `FeatureSpaces` (mean vectors per item, optionally
timed window rows), `TextEncoder` (text to a vector in a named space),
`IdentityLinks` (trusted shared-contributor links, optional) and `Annotator`
(tag writes, optional and never called on a read path). Ratings and engagement
change only through two callbacks the Engine is constructed with. The
filesystem source fills the first three and supplies the callbacks:

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

A delivery is one served page, journaled with its ranking generation; it proves
nothing about attention. A qualified view is a card that stayed at least 60
percent visible for 1,200 ms in a foreground tab. An outcome follows a
qualified view: watched seconds from a committed player batch, an explicit
rating, or an engagement count. Ratings trump implicit signals in both
directions.

Outcomes are credited to the delivery that preceded them only after the
attribution window closes (production 3,600 s; `demo` 5 s, labelled
`demo-explicit-w5-v1`). Trials then ripen for six more hours before the tuner
may count them. The demo shortens only the first gate.

Like, dislike, clear and count engagement are ledgered operations: the engine
journals, asks the authority to apply, then confirms with the authority's
values. Undo restores the exact original value. A lost reply blocks further
actions on that item and offers Check status, which reconciles against the
ledger and keeps the receipt so Undo still works.

The tuner runs one experiment at a time, base against candidate for one knob.
Once each arm has enough ripened trials across enough sessions, and the
difference is significant and does not collapse category diversity, it promotes
the winner by itself and rotates to the next knob. Every move is a ledger row
that the Engine page can revert; reset restores the standard values.

A host with its own profile and candidate builder passes `prepare_feed` to the
Engine. It receives the frozen request context, the resolved configuration, the
generation seed, the requested kinds and the pinned catalog, and returns the
inputs of `ranking.rank_page`: scored components, image components, target
category shares, explanations, admitted, excluded and seed keys, exploration,
control and fallback candidates, profile metadata and source counts. The Engine
selects with the public `rank_page`, rejects unknown or ineligible keys, fences
every committed space and keeps provenance, cursors and delivery. That path is
revision `feedloop_feed_prepared/v1` and claims no historical-ranker parity.

A page carries a cursor bound to its ranking generation. When the generation is
gone (a restart, changed features, a changed catalog) the server refuses the
cursor, and the client restarts once from the top, keeps its cards and drops
duplicates. Repeated refusals become a Retry, never a loop.

## Integration

If you supply your own capture, embeddings and catalog instead of a media
folder, read [docs/integration.md](docs/integration.md). It gives the event
contract, the feature-space contract and the catalog contract with exact field
names and types, the attribution rules, and a synthetic capture example.

## Limits (v0.x)

- 0.8.4: Feed, Search and Similar items (and the Similar seed) carry the item's Signals rating as `rating100`; an item with a qualified view in the last `recent_view_hours` (default 24, 0 disables; needs `impression_discount` below 1) stays out of fresh Feed pages unless the library is too small to fill the page.
- 0.8.3: animated_image catalog flag renders GIF-style media as an `<img>` on cards, Feed and the Similar seed, with no player or watch capture.
- 0.8.2: tuner evidence is scoped to the active experiment's cohort; a trial-level problem excludes that trial only, unlinked records are diagnostics, and the scorecard shows recorded, evaluable and excluded counts per arm.
- 0.8.1: optional preview_url and open_url on catalog items for grid-card previews and links.
- 0.8.0: per-category weighting moved to category_weights={name: multiplier}; the default is neutral.

- The slot contracts are proven against the filesystem source only and may
  change before 1.0.
- Search needs window rows with real timestamps (sidecar `segments` or a timed
  extractor). A means-only space is no-feature for Search while Similar and the
  Feed still use its means.
- The base install decodes no media; durations come from sidecars or the
  browser player.
- The extractors are optional extras (`pip install feedloop[extract]`).
  Frame sampling and container audio decoding need ffmpeg on the path.
  Without real model weights the visual and audio spaces fall back to
  sidecar-text means; verifying the full model weights is the reader's
  step.
- Zero-shot tags are weaker than a trained tagger. Knob values are not
  enjoyment probabilities.

## Evaluation

`experimental/evaluation/` ports an offline evaluator (frozen snapshots, pinned
ranking sources, observed-label metrics) outside the public API and the test
gates. It carries the label **no validated measurement yet**: nothing in this
repository measures the engine's quality, causal lift or ranking superiority.

## License

`LICENSE` is the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). `COMMERCIAL-LICENSE.md` describes the commercial option
for organizations that cannot comply with the AGPL, and `CLA.md` is the
contributor license agreement.

## Repository layout

```
src/feedloop/            the package (numpy only)
  sources/filesystem.py  Catalog, Signals and FeatureSpaces over a folder, sidecars and SQLite
  extractors/            optional visual (open CLIP) and audio (CLAP) extractors
  server/app.py          standard-library HTTP server
  cli.py                 feedloop demo | serve | extract
docs/integration.md      capture, embeddings and catalog contracts for a custom host
web/                     the client, packaged with the wheel
scripts/verify_demo.py   portable end-to-end demonstration runner (real server, labelled fixtures)
scripts/check_client.py  Playwright browser contract runner
experimental/evaluation  frozen offline evaluator (not installed; see its README)
tests/                   pure unit tests
```

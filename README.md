# feedloop

feedloop is a recommendation feed that learns from what you watch, rate and skip, and explains every ranking in words. Point it at a folder of videos and images and it serves a local web client with a feed, search, similar items and an Engine page that shows the evidence behind each decision. Everything it learns lives in an append-only ledger next to your files, and every automatic change can be undone.

## Setup

The package is not on PyPI. Install it from GitHub. The base install depends on numpy only. Replace `./my-media` with a folder that holds some mp4, webm, mkv, mov, m4v, jpg, jpeg, png, gif or webp files. `--fixture-sidecars` writes labelled synthetic sidecars, with timed segments for videos, for files that have none. It never touches an existing sidecar.

```sh
git clone https://github.com/federicobuilds/feedloop
cd feedloop
python3.11 -m venv .venv && . .venv/bin/activate
# Windows PowerShell instead of the line above: py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e .
feedloop demo ./my-media --fixture-sidecars
```

`demo` prints two addresses. Open the second one, `http://127.0.0.1:8765/#key=...`. The fragment carries the key that mutations (deliveries, views, watch batches, ratings, tuner controls) need. Reads work without it.

## What the demo shows

`demo` scans the folder, gives each file a stable id, reads an optional `<file>.json` sidecar, and serves on http://127.0.0.1:8765/.

- Feed: a scroll-snap column of cards. Each card explains its rank in words; an absent signal reads "not measured", never zero.
- Home: shelves with Load more.
- Search: text search over timed window rows. With fixture sidecars it works on videos; an item without segments reports no-feature.
- Similar: more like one video, from tag shares blended with mean-vector similarity.
- Engine: the scorecard, the running experiment and the reversible tuner ledger. The demo uses a five second attribution window and a fifteen second tick, so a watch shows up here within a minute.

Stop the server with Ctrl-C. State lives in `<folder>/.feedloop` (`source.sqlite`, `ledger.sqlite`, `tuner.sqlite`, `spaces/`) unless you pass `--state DIR`. To serve that state again with the production policy (one hour window, six hour trial ripening), run `feedloop serve ./my-media`. `serve` never creates stores unless you pass `--init`.

## Your own media and sidecars

Put a `<file>.json` next to a media file. Every field is optional. Tags can be plain strings or objects with `name`, `category` and `seconds`. `duration_s` is the only way the base install knows a duration before the browser player reports one. `segments` give Search its timestamps and are your own positions, not measurements.

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

The base install decodes no media. Learned feature spaces come from the optional extractors. Install them with `pip install -e '.[extract]'` (torch, open_clip_torch, pillow, transformers, soundfile).

| Extractor | `--model` | Weights | Notes |
| --- | --- | --- | --- |
| Visual ([open_clip](https://github.com/mlfoundations/open_clip)) | `hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 605 MB | Recommended. CPU and MPS are fine. |
| Visual, larger ([open_clip](https://github.com/mlfoundations/open_clip)) | `hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | 1.7 GB | Slower. A GPU helps. |
| Audio ([CLAP](https://huggingface.co/laion/clap-htsat-unfused)) | `laion/clap-htsat-unfused` | 615 MB | Audio files only. |

```sh
feedloop extract ./my-media --kind visual --model hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K --vocabulary tags.txt
feedloop extract ./my-media --kind audio --model laion/clap-htsat-unfused
```

The first run downloads the weights into the Hugging Face cache, `~/.cache/huggingface/hub`. The CLI has no `--pretrained` flag: a bare open_clip architecture name such as `ViT-B-32` builds the model with random weights, so name the model in `hf-hub:` form as in the table. Both commands take `--space NAME` (default `visual` or `audioembed`), `--device cpu|cuda|mps` and `--overwrite`. Without `--overwrite` an existing space is kept. `tags.txt` holds one zero-shot tag per line:

```
harbor
fog
forest
night
crowd
```

The visual extractor embeds image files and writes zero-shot tags from the vocabulary to `<file>.generated.json`. It never writes into your sidecar, and with `--overwrite` it replaces only its own fields in the generated file. The audio extractor decodes with soundfile and doubles as the text encoder for sound search. Not yet verified against real weights; video frames and container audio are not extracted.

## Bring your own vectors

Any extractor that writes mean vectors, and with real timestamps window rows, into a named space under `<state>/spaces/` plugs in. `space_roles` maps the visual, semantic, voice and sound roles onto your space names, and text search needs a `TextEncoder` for the same space.

## How the engine learns

A host fills six protocols from `feedloop.slots`: `Catalog` (items, files, fingerprints, tag features), `Signals` (ratings, engagement counts, watch history as of a known time), `FeatureSpaces` (mean vectors per item, optionally timed window rows), `TextEncoder` (text to a vector in a named space), `IdentityLinks` (trusted shared-contributor links, optional) and `Annotator` (tag writes, optional and never called on a read path). Ratings and engagement change only through two callbacks the Engine is constructed with. The filesystem source fills the first three and supplies the callbacks:

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

A delivery is one served page, journaled with its ranking generation; it proves nothing about attention. A qualified view is a card that stayed at least 60 percent visible for 1,200 ms in a foreground tab. An outcome follows a qualified view: watched seconds from a committed player batch, an explicit rating, or an engagement count. Ratings trump implicit signals in both directions.

Outcomes are credited to the delivery that preceded them only after the attribution window closes (production 3,600 s; `demo` 5 s, labelled `demo-explicit-w5-v1`). Trials then ripen for six more hours before the tuner may count them. The demo shortens only the first gate.

Like, dislike, clear and count engagement are ledgered operations: the engine journals, asks the authority to apply, then confirms with the authority's values. Undo restores the exact original value. A lost reply blocks further actions on that item and offers Check status, which reconciles against the ledger and keeps the receipt so Undo still works.

The tuner runs one experiment at a time, base against candidate for one knob. Once each arm has enough ripened trials across enough sessions, and the difference is significant and does not collapse category diversity, it promotes the winner by itself and rotates to the next knob. Every move is a ledger row that the Engine page can revert; reset restores the standard values.

A page carries a cursor bound to its ranking generation. When the generation is gone (a restart, changed features, a changed catalog) the server refuses the cursor, and the client restarts once from the top, keeps its cards and drops duplicates. Repeated refusals become a Retry, never a loop.

## Limits (v0.x)

- The slot contracts are proven against the filesystem source only and may change before 1.0.
- Search needs window rows with real timestamps (sidecar `segments` or a timed extractor). A means-only space is no-feature for Search while Similar and the Feed still use its means.
- The base install decodes no media; durations come from sidecars or the browser player.
- Video frame sampling and container audio decoding are not shipped; the extractors have not run against real weights in this repository.
- Zero-shot tags are weaker than a trained tagger. Knob values are not enjoyment probabilities.

## Evaluation

`experimental/evaluation/` ports an offline evaluator (frozen snapshots, pinned ranking sources, observed-label metrics) outside the public API and the test gates. It carries the label **no validated measurement yet**: nothing in this repository measures the engine's quality, causal lift or ranking superiority.

## License

`LICENSE` is the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). `COMMERCIAL-LICENSE.md` describes the commercial option for organizations that cannot comply with the AGPL, and `CLA.md` is the contributor license agreement.

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

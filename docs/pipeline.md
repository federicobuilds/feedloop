# Run the FeedLoop pipeline

FeedLoop contains a recommendation engine and an ingestion pipeline. The
pipeline runs the stages you choose, saves each model result, and publishes
configured embedding spaces for Similar and recommendations.

## Run locally

```sh
python -m pip install '.[extract]'
feedloop pipeline ./media --stages stages.json
feedloop serve ./media --init
```

Save a stage manifest as `stages.json`.

```json
[
  {
    "name": "visual",
    "adapter": "feedloop.folder_pipeline:embed",
    "recipe": {"kind": "visual", "model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
    "space": "visual"
  },
  {
    "name": "sound",
    "adapter": "feedloop.folder_pipeline:embed",
    "recipe": {"kind": "audio", "model": "laion/clap-htsat-unfused"},
    "kinds": ["video"],
    "space": "audioembed"
  }
]
```

The built-in adapters default to CPU. Set `"device": "cuda"` in a recipe
for NVIDIA CUDA or AMD ROCm with compatible PyTorch. When you rerun a folder,
the pipeline reuses successes with the same file fingerprint and recipe and
retries failures. Give incompatible model revisions distinct feature-space
names and select them with `serve --space-role`.

## Run on Runpod

Build a worker image using the same checkout as the controller. The provided
Dockerfile uses a Runpod PyTorch base with SSH support. Include any custom
adapters and their dependencies in the image. If your image is private,
configure a Runpod registry credential and pass its ID with `--registry`.

```sh
docker build --platform linux/amd64 -f docker/pipeline.Dockerfile -t my-registry/feedloop-worker:my-version .
docker push my-registry/feedloop-worker:my-version
```

Set `RUNPOD_API_KEY` in the controller environment and provide a dedicated
Ed25519 key pair. Set `"device": "cuda"` in your GPU stage recipes. Select an
available Runpod GPU catalog ID and a compute rate you approve.

```sh
feedloop pipeline ./media --stages stages.json --provider runpod --allow-remote \
  --image my-registry/feedloop-worker:my-version --gpu 'NVIDIA A40' \
  --ssh-key ~/.ssh/feedloop_ed25519 --max-hourly-rate 0.59 --min-cuda 12.8
```

FeedLoop rents one Pod only when a stage requires work. It checks pricing,
pins the worker's SSH host key through the authenticated Runpod API, and
streams the files required by pending stages. It uses ephemeral storage on
the Pod and saves results on the controller. The account API key stays on
the controller. Runpod mode supports macOS and Linux controllers.

FeedLoop confirms teardown before publishing from the run. If an interruption
leaves an allocation unresolved, use the same Runpod account and state
directory to reconcile it before retrying.

```sh
feedloop pipeline-cleanup --state ./media/.feedloop
```

## Supply a model

Install a trusted Python adapter in the model environment. The manifest
names a `module:callable` taking `(path, recipe)` and returning a JSON object.
An `embedding` is a nonempty list of numbers. Results such as labels and
detections can be read through `feedloop.multistage.ResultStore`. Embeddings
from stages with a `space` are published to the engine. A host can supply a
publisher for additional result types.

Use `feedloop.multistage` for item-level dispatch and result storage,
`feedloop.workflow` for the shared run lifecycle, and `feedloop.runpod` for
provider operations with an injected transport or allocation journal.

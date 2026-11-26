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

Use the public [FeedLoop image on Docker Hub](https://hub.docker.com/r/federicobuilds/feedloop).
It includes the FeedLoop engine, pipeline CLI, worker, and extraction
dependencies for NVIDIA CUDA 12.8. The image is built for Linux amd64.
Runpod pulls it without Docker Hub credentials. Adapters download their model weights on first use when needed.

Install the matching controller in a Python 3.11+ environment on macOS or Linux.

```sh
python -m pip install 'feedloop @ git+https://github.com/federicobuilds/feedloop.git@v0.10.0'
ssh-keygen -t ed25519 -f ~/.ssh/feedloop_ed25519 -N ""
```

Use an existing dedicated Ed25519 key pair if you have one. Set
`RUNPOD_API_KEY` in the controller's environment using your own
[Runpod API key](https://console.runpod.io/user/settings). Save the stage
manifest from the local example and add `"device": "cuda"` inside each GPU
stage's `recipe`. Select a GPU available in your account and set your maximum
hourly compute rate. The A40 and rate below are examples.

```sh
feedloop pipeline ./media --stages stages.json --provider runpod --allow-remote \
  --image federicobuilds/feedloop:v0.10.0 --gpu 'NVIDIA A40' \
  --ssh-key ~/.ssh/feedloop_ed25519 --max-hourly-rate 0.59 --min-cuda 12.8
```

FeedLoop rents one Pod only when a stage requires work. It checks pricing,
pins the worker's SSH host key through the authenticated Runpod API, and
streams the files required by pending stages. It uses ephemeral storage on
the Pod and saves results on the controller. The account API key stays on
the controller. Runpod mode supports macOS and Linux controllers.

Verified on Runpod with the published `v0.10.0` image pinned by digest. One
NVIDIA A40 ran four stages across two synthetic fixtures using PyTorch 2.9.1
and CUDA 12.8. Both items were published to the controller's visual embedding
space. An unchanged rerun reused all four results, and the Runpod API
confirmed the pod was deleted.

FeedLoop confirms teardown before publishing from the run. If an interruption
leaves an allocation unresolved, use the same Runpod account and state
directory to reconcile it before retrying.

```sh
feedloop pipeline-cleanup --state ./media/.feedloop
```

### Build a custom worker image

For custom adapters, use [the worker Dockerfile](../docker/pipeline.Dockerfile)
from the same FeedLoop release as your controller. Add the adapter and its
dependencies to the image so the worker can import it.

```sh
docker build --platform linux/amd64 -f docker/pipeline.Dockerfile -t my-registry/feedloop-worker:my-version .
docker push my-registry/feedloop-worker:my-version
```

Pass the custom image with `--image`. For a private image, configure a Runpod
registry credential and pass its ID with `--registry`.

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

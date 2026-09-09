# Containerization plan for the SVM-XAI demo service

This plan describes how to containerize the FastAPI service that currently runs
as `svm-xai.service` (`uvicorn main_api:app` on port 8000) and how to publish it
through a Cloudflare tunnel, with notes for each candidate host.

## Goals

- Replace the fragile, end-of-life runtime (Ubuntu 20.04 + Python 3.8.10 +
  TensorFlow 2.13) with a reproducible image on a modern, supported base, so a
  host reinstall or dead drive no longer means rebuilding the environment by
  hand.
- Add real restart and health-check behavior on top of the old
  `Restart=always`.
- Version both code and data so neither a dead drive nor a wiped TrueNAS loses
  the demo.
- Keep the public entry point (`paul.whitten.dev` via Cloudflare tunnel) working
  regardless of which host runs the container.

## The resilience gap to keep in mind

Containerizing improves reproducibility and recovery speed, but it does not by
itself protect against the failure that took the site down: a disk failure.
Real resilience needs all three of the following.

- The image pushed to a registry (GitHub Container Registry, Docker Hub, or a
  registry hosted on TrueNAS).
- The runtime data (`models/`, `kb/`, `out_neighbors/`, `static/`) backed up on
  redundant storage. TrueNAS ZFS with snapshots is ideal for this.
- A documented one-command redeploy so any host can bring the service back.

## Host options and the key trade-off

| Host | Architecture | RAM | Notes |
|---|---|---|---|
| TrueNAS (i9-9900K) | x86_64 | 64 GB | Plenty of RAM and redundant disk; best home for both the service and its backups. |
| This machine | x86_64 | — | Same architecture as TrueNAS; images are interchangeable. |
| NVIDIA AGX Thor | aarch64 (arm64) | — | Different architecture. TensorFlow 2.13 wheels are x86-only, so a TF-based image will not run here without heavy rework. |

The decisive point: **the current image imports TensorFlow only as a side effect
of `import tensorflow` in `case_based_mnist.py`, but the served endpoints use
pickled scikit-learn models, not Keras.** Removing that import makes the image
pure Python + numpy/scipy/scikit-learn/scikit-image, which:

- drops the image by roughly 1 GB and runtime RAM by roughly 0.5 GB, and
- becomes trivially multi-architecture, so the same build target works on the
  x86 hosts and on the arm64 AGX Thor.

Recommendation: do the TensorFlow removal first (see step 2). It is the single
change that keeps all three host options open.

## Runtime modernization: base image and Python version

The current stack (Ubuntu 20.04 + Python 3.8.10) is end of life. Rather than
freezing an EOL runtime, containerize directly onto a modern, supported base.
This is safe here because removing TensorFlow (step 2) eliminates the only
dependency that pinned the old Python.

Target: `python:3.12-slim-bookworm` (Debian 12 "bookworm", Python 3.12).

- `case-explainer` supports Python 3.8 through 3.12, so 3.12 is in range.
- The serving dependencies (numpy, scipy, scikit-learn, scikit-image, fastapi,
  uvicorn, pillow, imageio) all ship 3.12 wheels.
- TensorFlow 2.13 is the only blocker: it caps at Python 3.11 and ships x86-only
  wheels. Dropping it (step 2) is therefore a prerequisite for both the newer
  Python and the arm64 (AGX Thor) build.

If you prefer maximum wheel maturity over newest features,
`python:3.11-slim-bookworm` is a conservative fallback; the rest of the plan is
unchanged.

Regenerate `requirements-serve.txt` against the chosen Python rather than reusing
the 3.8 pins. Let `pip` resolve current versions of the serving-only packages,
and pin `case-explainer==0.2.0` (or the latest release).

## Sizing summary

Grounded in what the app loads at startup.

- Steady-state RAM: approximately 2 to 2.5 GB (dominated by
  `out_neighbors/train_data.json`, a 349 MB JSON parsed into Python lists, plus
  the TensorFlow import if not removed).
- Serving image size: approximately 2.5 to 3.0 GB with TensorFlow, or
  approximately 1.3 to 1.8 GB without it.
- Suggested container memory limit: 4 GB (comfortable on the 64 GB TrueNAS).

## Step 1: Define the runtime file set

Include only what the service reads at runtime. Exclude all training and build
artifacts.

Include:

- Application code: `main_api.py`, `calculate_sample.py`, `case_based_mnist.py`,
  `transform_parallel.py`, `load_mnist_data.py`, `load_emnist_resnet_data.py`,
  and any other modules imported at startup.
- Data: `models/mnist/nn/002/`, `kb/mnist/nn/002/`, `out_neighbors/`, `static/`,
  `trans_explainability.json`.

Exclude via `.dockerignore`:

- `images/` (3.6 GB), `transforms/` (735 MB), `cbe_out_mnist/` (369 MB),
  `old_pipelines/`, `log/`, `.venv/`, `.git/`, build scripts, and every
  `pipeline_config_*.json` not needed to serve.

## Step 2: Remove the TensorFlow dependency from the serving path (recommended)

- Confirm no served endpoint calls a Keras model. The prediction path uses the
  pickled sklearn MLP in `out_neighbors/nn.model` and the pickled transform
  models in `models/mnist/nn/002/`.
- Remove or guard `import tensorflow as tf` and `from tensorflow import keras`
  in `case_based_mnist.py`.
- Drop `tensorflow`, `tensorflow-estimator`, `tensorboard*`, `keras`, `h5py`,
  and related packages from a serving-only `requirements-serve.txt`.
- Validate the service still answers `POST /submit_pbe` and the case-based
  endpoints before building the image.

## Step 3: Convert the large data file (recommended)

`out_neighbors/train_data.json` (349 MB) is parsed into nested Python lists,
which balloon to roughly 1.5 GB in RAM. Convert it once to a `float32` NumPy
array and load it memory-mapped.

- Offline: `np.save('out_neighbors/train_data.npy', np.asarray(train_data, dtype=np.float32))`.
- In `case_based_mnist.py`: load with `np.load(path, mmap_mode='r')` instead of
  `json.load`.
- Effect: startup RAM for this dataset drops from roughly 1.5 GB to roughly
  150 MB, and startup is faster.

## Step 4: Author the container build

Use a multi-stage build so the final image carries only site-packages plus the
runtime data.

```dockerfile
# Build stage: install dependencies into a virtual environment
FROM python:3.12-slim-bookworm AS build
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Runtime stage: copy venv and application, nothing else
FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=build /opt/venv /opt/venv
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Notes:

- The base is `python:3.12-slim-bookworm` (Debian 12). See the runtime
  modernization section; this replaces the end-of-life Ubuntu 20.04 / Python 3.8
  stack and is unblocked by the TensorFlow removal in step 2.
- If the AGX Thor is a target, build multi-arch with
  `docker buildx build --platform linux/amd64,linux/arm64`. This is only
  practical after step 2 removes TensorFlow.

## Step 5: Cloudflare tunnel in containers

The old machine used a token-based (remotely managed) tunnel: the ingress rules
live in the Cloudflare Zero Trust dashboard, not in a local file. That model
containerizes cleanly by running `cloudflared` as its own container alongside
the app.

- Run a second container from `cloudflare/cloudflared` with
  `tunnel run --token <TOKEN>`.
- In the Cloudflare Zero Trust dashboard, set the public hostname
  (`paul.whitten.dev`) to route to `http://app:8000`, where `app` is the app
  container name on the shared Docker network. No local ingress file is needed.

Security: the token embedded in the old `cloudflared.service` unit must be
treated as compromised, since the drive has been mounted elsewhere. Create a new
tunnel (or rotate the token) in the Cloudflare dashboard and use the new token
in the container.

### Compose sketch

```yaml
services:
  app:
    build: .
    image: svm-xai:latest
    restart: unless-stopped
    mem_limit: 4g
    expose:
      - "8000"
    volumes:
      # Runtime assets from the ZFS dataset (Gitea-versioned), not baked into the image
      - /mnt/pool/apps/svm-xai/models:/app/models:ro
      - /mnt/pool/apps/svm-xai/kb:/app/kb:ro
      - /mnt/pool/apps/svm-xai/out_neighbors:/app/out_neighbors:ro

  cloudflared:
    image: cloudflare/cloudflared:latest
    restart: unless-stopped
    command: tunnel --no-autoupdate run --token ${CF_TUNNEL_TOKEN}
    depends_on:
      - app
```

Keep `CF_TUNNEL_TOKEN` in a `.env` file or a secret, never committed to git.

## Step 6: Hosting on TrueNAS specifically

TrueNAS SCALE runs containers, so both the app and the tunnel can live there.

- Create a dataset (for example `pool/apps/svm-xai`) to hold the runtime data
  and the `.env`. ZFS snapshots then cover your backups.
- Deploy the two-container stack. Depending on the TrueNAS version, use either
  the built-in Docker Compose support (recent SCALE releases) or a Custom App
  definition. Put both containers on the same bridge network so `cloudflared`
  can reach the app as `http://app:8000`.
- Because the tunnel is outbound-only, you do not need to open any inbound port
  or router firewall rule on TrueNAS; `cloudflared` dials out to Cloudflare.
- Use TrueNAS to hold the registry copy of the image and the dataset snapshots,
  making it the durable home even if the demo later runs elsewhere.

## Version control and data storage

Use a three-layer split so code stays resilient while bulky binaries stay local
and versioned.

- Public source repo (GitHub or GitLab): code, configs, pipeline scripts, and
  this plan. `.gitignore` and `.dockerignore` exclude all binaries. This is the
  resilience anchor: it contains everything needed to regenerate the models, so
  it survives even if TrueNAS is wiped. Optionally mirror the small runtime set
  (`models/mnist/nn/002`, `kb/mnist/nn/002`, `out_neighbors/nn.model`, roughly
  80 MB) here via Git LFS so a cold start never depends on TrueNAS.
- Gitea assets repo (on TrueNAS): LFS-tracked `models/`, `kb/`, `out_neighbors/`,
  and pipeline or training outputs worth versioning. Enable the Gitea LFS server
  (`[server] LFS_START_SERVER = true` in `app.ini`). No bandwidth quotas apply.
- ZFS dataset (on TrueNAS): the durable runtime copy, bind-mounted into the
  container. Snapshot it and replicate the snapshots to another pool, host, or
  offsite target (or back up the Gitea LFS store with restic/borg) so a wipe
  cannot destroy the only copy of the trained binaries.

Do not use two remotes on one repo to achieve the public/private split: the same
commits would reference the LFS files, so both remotes would see the same file
set. Use two separate repos linked at deploy time by a fetch step or a git
submodule.

Because the container runs on TrueNAS, prefer bind-mounting the ZFS dataset over
cloning LFS at runtime. Gitea LFS then provides versioning, ZFS provides
durability, and the bind mount hands the data to the container with no clone
step, which also keeps the image small (code only).

### Repositories

The concrete repositories for the split, both MIT licensed:

- Source (public, GitHub): `git@github.com:paulwhitten/explainable-ai-demos.git`.
  Holds code, configs, pipeline scripts, and this plan. No binaries.
- Assets (Gitea on TrueNAS, LFS): `http://truenas.local:30008/paulwhitten/explainable-ai-demo-assets.git`.
  Holds LFS-tracked `models/`, `kb/`, `out_neighbors/`, and pipeline or training
  outputs, populated by the pipeline.
- Local working area: `working/` under the old repo, containing local clones of
  both repositories during migration.

## Step 7: Backup and recovery

Recovery rests on the three-layer split in the version control and data storage
section.

- Push the built image to a registry (GitHub Container Registry or a registry on
  TrueNAS) after each change.
- Snapshot the ZFS runtime dataset and replicate the snapshots offsite; the
  Gitea assets repo gives per-change history of the binaries.
- Keep the public source repo current so, in the worst case (TrueNAS gone), you
  can regenerate the binaries from the pipeline and redeploy with one
  `docker compose up -d`.
- Optionally keep the roughly 80 MB runtime-set LFS mirror in the public repo so
  the demo can boot immediately without regeneration.

## Open decisions

- Final host: TrueNAS, this machine, or AGX Thor. If the AGX Thor is in scope,
  step 2 (remove TensorFlow) becomes required, not optional.
- Runtime data delivery: resolved toward mounting the assets from the ZFS
  dataset and keeping the image code-only (see version control and data
  storage).
- Python version: resolved toward migrating to `python:3.12-slim-bookworm`
  (`python:3.11-slim-bookworm` as a conservative fallback), contingent on the
  TensorFlow removal in step 2.
- Which host serves as the public remote (new GitHub repo versus the existing
  GitLab remote), and whether to mirror the small runtime set into the public
  repo via Git LFS.

## Suggested order of work

1. Stand up the two repos: public source (excludes binaries) and the Gitea
   assets repo with LFS; create the ZFS dataset for runtime assets (version
   control and data storage).
2. Create `.dockerignore` and a fresh `requirements-serve.txt` targeting Python
   3.12 (steps 1 and 2 and the runtime modernization section).
3. Remove the TensorFlow import and verify the endpoints still work (step 2).
4. Convert `train_data.json` to `.npy` and switch the loader (step 3).
5. Build the multi-stage image on `python:3.12-slim-bookworm`; confirm RAM and
   image size (step 4).
6. Create a new Cloudflare tunnel and wire up the compose stack with the ZFS
   bind mounts (steps 5 and 6).
7. Deploy on the chosen host, validate the public hostname, and set up snapshot
   replication (steps 6 and 7).

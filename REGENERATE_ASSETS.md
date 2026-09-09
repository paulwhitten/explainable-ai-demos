# Regenerate assets runbook

This runbook regenerates the demo's model and data assets from scratch under the
modern Python 3.12 stack, then publishes them to the LFS assets repository. Run
it whenever the stack changes (as now, moving off Python 3.8 / scikit-learn
1.3.2), since scikit-learn does not guarantee that pickles load correctly across
versions.

## What gets regenerated

Two independent tracks feed the demo:

- Property-based (transform) models and knowledge base, used by `/submit_pbe`
  and `/submit`:
  - `models/mnist/nn/002/` — one classifier per transform.
  - `kb/mnist/nn/002/` — `kb.json`, `cm.json`, and `*-stat.json`.
- Case-based data, used by `/submit_cbe` and `/submit_similarity_compare`:
  - `out_neighbors/nn.model` — the MLP.
  - `out_neighbors/train_labels.json` and `out_neighbors/train_data.npy` — the
    case base.

The minimal runtime set the container needs is those five items. Everything else
the pipeline emits (`transforms/`, `images/`, `neighbors.json`,
`case-based_results.json`, logs) is intermediate and does not belong in the
assets repo.

## Prerequisites

- Python 3.12 with the serving dependencies. The regeneration steps need
  numpy, scipy, scikit-learn, scikit-image, pillow, and imageio, all of which
  are already in `requirements-serve.txt`. TensorFlow is not needed for the `nn`
  path.
- Node.js, for `process_kb.js`.
- The MNIST IDX files: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`,
  `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`.

```bash
cd explainable-ai-demos
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-serve.txt
export MNIST_DIR="$HOME/mnist"   # folder holding the four IDX files
```

## Track A: property-based models and knowledge base

`pipeline.py` reads a config that points at the MNIST IDX files and writes the
transforms, models, and knowledge base. Update the four file paths in the config
to your `MNIST_DIR` first (the checked-in config uses an older absolute path).

```bash
# point the config at your MNIST data
python - <<'PY'
import json, os
p = "pipeline_config_mnist_nn_002.json"
c = json.load(open(p))
d = os.environ["MNIST_DIR"]
c["training_images_file"] = os.path.join(d, "train-images-idx3-ubyte")
c["training_labels_file"] = os.path.join(d, "train-labels-idx1-ubyte")
c["test_images_file"]     = os.path.join(d, "t10k-images-idx3-ubyte")
c["test_labels_file"]     = os.path.join(d, "t10k-labels-idx1-ubyte")
json.dump(c, open(p, "w"), indent=4)
print("updated", p)
PY

# run the full transform -> train -> build_kb -> process_kb pipeline
python pipeline.py -c pipeline_config_mnist_nn_002.json
```

This produces `models/mnist/nn/002/` and `kb/mnist/nn/002/`. The pipeline skips
stages whose output directory already exists, so remove a stage's folder to force
a rebuild.

## Track B: case-based data

`case_based_mnist.py` trains the case-based MLP and writes the case base,
including `nn.model`, into the output folder.

```bash
mkdir -p out_neighbors
python case_based_mnist.py -m "$MNIST_DIR" -n out_neighbors -o out_neighbors -t nn
```

This writes `out_neighbors/train_labels.json`, `out_neighbors/train_data.json`,
and `out_neighbors/train_data.npy` (the compact float32 array the serving path
memory-maps), plus intermediate `neighbors.json` and `case-based_results.json`.

## Verify before publishing

Confirm the serving service loads the fresh assets and returns sensible
explanations.

```bash
python - <<'PY'
import json, numpy as np
from case_explainer_service import CaseExplainerService
svc = CaseExplainerService("out_neighbors")
row0 = np.asarray(json.load(open("out_neighbors/train_data.json"))[0], dtype=np.float32)
res = svc.explain_both(row0 * 255.0)
f, a = res["feature"], res["activation"]
print("prediction:", res["prediction"])
print("feature   :", round(f["correspondence"], 3), f["correspondence_interpretation"],
      "self-match dist", round(f["neighbors"][0]["distance"], 4))
print("activation:", round(a["correspondence"], 3), a["correspondence_interpretation"])
PY
```

Expected: the feature-space nearest neighbor of a training row is itself
(distance about 0), and both correspondences are high for a confidently
classified digit.

## Publish to the assets repository

Copy only the minimal runtime set into the assets repo, then commit through Git
LFS. `train_data.npy` (about 188 MB) replaces `train_data.json` (about 365 MB);
the serving loader prefers the `.npy`, so the JSON does not need to be published.

```bash
ASSETS=../explainable-ai-demo-assets
mkdir -p "$ASSETS/models/mnist/nn/002" "$ASSETS/kb/mnist/nn/002" "$ASSETS/out_neighbors"

cp models/mnist/nn/002/*        "$ASSETS/models/mnist/nn/002/"
cp kb/mnist/nn/002/*            "$ASSETS/kb/mnist/nn/002/"
cp out_neighbors/nn.model       "$ASSETS/out_neighbors/"
cp out_neighbors/train_labels.json "$ASSETS/out_neighbors/"
cp out_neighbors/train_data.npy    "$ASSETS/out_neighbors/"

cd "$ASSETS"
git lfs track                       # confirm patterns cover *.model, *.npy, kb/**, out_neighbors/**
git add -A
git lfs ls-files                    # confirm the large files are LFS pointers, not raw blobs
git commit -m "Regenerate MNIST nn assets under Python 3.12 stack"
git push
```

## Deploy the refreshed assets

The container mounts these from the ZFS dataset (see the containerization plan).
Sync the published assets onto the dataset and restart:

```bash
# on TrueNAS: pull the assets repo (or rsync the five items) into the dataset
git -C /mnt/pool/apps/svm-xai/assets pull   # dataset checkout of the assets repo
docker compose restart app
```

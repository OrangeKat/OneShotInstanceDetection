# Hugging Face publishing

Two artifacts, published separately:

| Artifact | Hub type | Contents |
|---|---|---|
| Weights | model repo | `config.json`, `model.pt`, model card |
| Demo (optional) | Space | Gradio or Streamlit app that installs `osid` from GitHub |

Only the trained decoder and heads are stored. The backbone is referenced by
name and fetched from timm, so uploads stay small and DINOv3's own licence keeps
applying to those weights.

## 1. Model repository

```python
from osid.model.model import BoundaryModel
from osid.model.hub import push_to_hub

model = BoundaryModel()  # or a model restored from a checkpoint
push_to_hub(model, "your-name/osid-boundary", private=True)
```

The call writes `config.json` and `model.pt` into a staging directory, creates
the repository if needed, and uploads both. Load it back with:

```python
from osid.model.hub import from_pretrained

model = from_pretrained("your-name/osid-boundary")
```

`huggingface_hub` is not a project dependency; use
`uv run --with huggingface_hub ...` or install it in the environment that does
the upload.

## 2. Model card

Copy the template below to `README.md` in the model repository. The YAML header
is what the Hub parses; the prose is what a reader sees.

```markdown
---
license: other
license_name: dinov3
library_name: osid
tags:
  - instance-detection
  - keypoint-matching
  - boundary-detection
  - dinov3
datasets:
  - flickrlogos32
  - rpc
  - hpatches
metrics:
  - average_precision
  - hit-rate
---

# osid-boundary

Boundary-anchored keypoint matching for one-shot instance detection: given a
reference crop, predict keypoints on both images and the correspondences
between them, with keypoints on the object outline rather than anywhere the
detector finds contrast.

## Architecture

- Frozen DINOv3 ViT-S/16 backbone, multi-layer features.
- Fusion decoder to stride 4.
- Boundary head (1 channel) and descriptor head (128-D, L2-normalised).
- Keypoints: local maxima of the boundary map, top-k.
- Matching: mutual nearest neighbour with a ratio test, homography by RANSAC.

## Training

Not trained yet. The scaffold runs end to end with untrained heads; Phase 0 of
the design decides whether the boundary premise holds before training.

## Licence

The backbone weights are DINOv3 and remain under Meta's DINOv3 licence. The
head weights are released under the repository's MIT licence.
```

## 3. Space (optional)

A Space only needs to install the package and call the model:

```text
app.py
requirements.txt   # osid @ git+https://github.com/OrangeKat/OneShotInstanceDetection
README.md          # YAML: sdk: gradio
```

Because the repository uses the `src/` layout, installing from git works without
extra configuration:

```bash
pip install "osid @ git+https://github.com/OrangeKat/OneShotInstanceDetection"
```

## Before the first upload

- Confirm the DINOv3 licence terms for redistributing derived weights.
- Keep `hf_hub` auth as a token in the environment, never in the repository.
- Do not upload datasets; the four used here have their own terms.

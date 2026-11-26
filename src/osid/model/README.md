# osid/model

The boundary-anchored matcher: given a reference crop and a query image, put
keypoints on the object outline, describe them, and match them.

Status: scaffold. The pipeline runs end to end, but the decoder and heads are
untrained, so the output is structurally correct and semantically meaningless
until Phase 0 in [`docs/model-design.md`](../../../docs/model-design.md) says
the premise is worth training.

## Reading order

| File | What it does | Where to change things |
|---|---|---|
| `config.py` | every knob, in one dataclass | backbone name, input size, tapped layers, widths, top-k, thresholds |
| `backbone.py` | frozen DINOv3 ViT-S/16, multi-layer features, resize + normalise, feature→pixel mapping | swap the backbone, change how layers are tapped or resized |
| `heads.py` | fusion decoder to stride 4, boundary head, descriptor head | decoder depth, channel widths, head design |
| `matcher.py` | local-maxima keypoint selection, mutual-NN matching with a ratio test | how keypoints are picked, how matching is scored |
| `model.py` | the pipeline: `match(reference, query) -> MatchResult` | wiring, homography estimation |
| `hub.py` | save/load and upload to the Hugging Face Hub | staging, repo naming |
| `__main__.py` | run one pair and write an overlay | the quickest way to see the model work |

## Run it

```bash
uv run python -m osid.model --index 0
uv run python -m osid.model --instance carlsberg --query 145074777.jpg --top-k 256
```

It prints keypoint and match counts and writes `outputs/model-demo.png` with the
reference and query side by side.

## Contract

`BoundaryModel.match` returns the same `MatchResult` the eight baselines return,
so once the weights are trained it can be registered in
`osid.benchmark.registry.MODELS` and benchmarked on both protocols with no new
plumbing.

## Deliberate omissions

- **No training loop yet.** The design says measure first; Phase 0 is in
  `scripts/analysis/`.
- **No multi-instance output.** One homography, like the baselines, so the
  comparison stays like-for-like.
- **Descriptors come from the same decoder as the boundary map.** That is a
  choice worth revisiting: a separate descriptor branch may match better, at the
  cost of another head to train.

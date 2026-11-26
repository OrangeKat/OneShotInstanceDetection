# osid/model

A per-image detector and descriptor: one image in, keypoints and descriptors
out. Matching is not part of the model — that keeps detection measurable,
cacheable, and independent of whichever matcher you put on top.

Status: scaffold. The pipeline runs end to end, but the decoder and heads are
untrained, so the output is structurally correct and semantically meaningless
until Phase 0 in [`docs/model-design.md`](../../../docs/model-design.md) says the
premise is worth training.

## Reading order

| File | What it does | Where to change things |
|---|---|---|
| `config.py` | every knob, in one dataclass | backbone name, input size, tapped layers, widths, top-k, thresholds |
| `backbone.py` | frozen DINOv3 ViT-S/16, multi-layer features, resize + normalise, feature→pixel mapping | swap the backbone, change which blocks are tapped |
| `heads.py` | fusion decoder to stride 4, boundary head, descriptor head | decoder depth, widths, head design |
| `detector.py` | the model: `detect(image) -> Detections` | how keypoints are selected, what is returned |
| `matcher.py` | matching utilities: mutual NN, ratio test | matching rules, when you want them shared |
| `adapter.py` | benchmark adapter: detect both images, then match | the matcher used for benchmarking, homography settings |
| `__main__.py` | run one image and write an overlay | the quickest way to see the model work |

## Run it

```bash
uv run python -m osid.model --index 0                 # query image from the manifest
uv run python -m osid.model --index 0 --reference     # reference crop instead
uv run python -m osid.model --image photo.jpg --top-k 256
```

It prints keypoint and descriptor statistics and writes `outputs/model-demo.png`
with points sized and coloured by boundary score.

## Why per-image

- **The claim is a detector property.** "Points sit on the outline" is measurable
  on one image against a pseudo-mask. Inside a pair model it would be entangled
  with matching, and failures could not be attributed.
- **Training without pairs.** Boundary supervision comes from pseudo-masks;
  descriptor supervision from two augmented views of the same image. Every image
  becomes training signal, which matters with ~2.2k Flickr crops.
- **Caching.** One reference against 69 queries costs 70 forwards, not 138, and
  the features are reusable across every matcher variant and ablation.
- **Matching is an ablation.** Mutual NN + ratio today; optimal transport or a
  learned matcher later, with the detector untouched.

The cost: ambiguity that only cross-image context can resolve — repeated logos,
sibling products — is out of reach until a learned matcher is added.

## Benchmark compatibility

`BenchmarkAdapter.match(reference, query)` implements the same `MatchResult`
contract as the eight baselines, so the model can be registered in
`osid.benchmark.registry.MODELS` once trained. It also exposes
`match_detections(detections0, detections1)`, which skips the backbone when the
same image is matched many times.

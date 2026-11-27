# osid/model

A per-image detector and descriptor: one image in, keypoints and descriptors
out. Matching is not part of the model — that keeps detection measurable,
cacheable, and independent of whichever matcher you put on top.

Status: scaffold. The pipeline runs end to end, but the decoder and heads are
untrained, so the output is structurally correct and semantically meaningless
until Phase 0 in [`docs/model-design.md`](../../../docs/model-design.md) says the
premise is worth training.

## Architecture

One image in, keypoints and descriptors out. Matching is the dashed part: it
belongs to `adapter.py`, not to the model.

```mermaid
flowchart TD
    A["image<br/>448 x 448 x 3 RGB"] --> B

    subgraph backbone["backbone.py — frozen DINOv3 ViT-S/16, 21.6M params"]
        B["patch embed<br/>16 x 16"] --> C["12 transformer blocks"]
        C --> D["tapped at blocks<br/>3, 6, 9, 11"]
        D --> E["4 x feature maps<br/>1 x 384 x 28 x 28"]
    end

    E --> F

    subgraph head["heads.py — trainable, 493k params"]
        F["FuseDecoder<br/>1x1 project each layer, sum<br/>2x upsample twice"] --> G["fused features<br/>1 x 128 x 112 x 112"]
        G --> H["BoundaryHead<br/>1x1 conv"]
        G --> I["DescriptorHead<br/>1x1 conv + L2 norm"]
        H --> J["boundary logits<br/>1 x 1 x 112 x 112"]
        I --> K["descriptor map<br/>1 x 128 x 112 x 112"]
    end

    J --> L

    subgraph pick["matcher.py — extraction, no learning"]
        L["3x3 max-pool peak test<br/>+ threshold + top-k"] --> M["keypoints<br/>N x 2, original pixels"]
        K --> N["gather at keypoints"]
        M --> N
        N --> O["Detections<br/>keypoints, scores, descriptors"]
    end

    O -. "adapter.py: mutual NN + ratio + RANSAC" .-> P["MatchResult<br/>matches, homography"]
    P -.-> Q["benchmark registry<br/>same contract as the 8 baselines"]
```

The solid path is the model. The dotted path is what the benchmark needs and
what you replace when you want a learned matcher instead of a geometric one.

### Tensor shapes

```text
input                     H x W x 3            resized to 448 on the long side
patch embed               h/16 x w/16          stride 16, so 28 x 28 at 448
tapped layers (4)         28 x 28 x 384        blocks 3, 6, 9, 11
fused (sum)               28 x 28 x 128        1x1 projections, summed
decoder output            112 x 112 x 128      two 2x upsampling stages, stride 4
boundary logits           112 x 112 x 1        sigmoid -> probability
descriptor map            112 x 112 x 128      L2-normalised per cell
keypoints                 N x 2                N <= top_k (512), original pixels
descriptors               N x 128              gathered from the map
```

### Where each part lives

| Stage | File | Function or class |
|---|---|---|
| resize, normalise, patch embed, blocks | `backbone.py` | `Dinov3Backbone.forward`, `_prepare` |
| layer fusion and upsampling | `heads.py` | `FuseDecoder.forward` |
| boundary logits | `heads.py` | `BoundaryHead.forward` |
| descriptors | `heads.py` | `DescriptorHead.forward` |
| peak test, threshold, top-k | `matcher.py` | `extract_keypoints` |
| detection set | `detector.py` | `DetectorDescriptor.detect` → `Detections` |
| matching for the benchmark | `adapter.py` | `BenchmarkAdapter.match`, `match_detections` |
| matching rules | `matcher.py` | `match_descriptors` |
| knobs | `config.py` | `ModelConfig` |

## Reading order

| File | What it does | Where to change things |
|---|---|---|
| `config.py` | every knob, in one dataclass | backbone name, input size, tapped layers, widths, top-k, thresholds |
| `backbone.py` | frozen DINOv3 ViT-S/16, multi-layer features, resize + normalise, feature→pixel mapping | swap the backbone, change which blocks are tapped |
| `heads.py` | fusion decoder to stride 4, boundary head, descriptor head | decoder depth, widths, head design |
| `detector.py` | the model: `detect(image) -> Detections` | how keypoints are selected, what is returned |
| `losses.py` | boundary heatmap target + focal loss, descriptor InfoNCE | loss design, temperatures, symmetry handling |
| `matcher.py` | matching utilities: mutual NN, ratio test | matching rules, when you want them shared |
| `adapter.py` | benchmark adapter: detect both images, then match | the matcher used for benchmarking, homography settings |
| `synth.py` | synthetic shapes on flat backgrounds, with masks and an exact homography | shape complexity, contrast, blur, noise, placement |
| `__main__.py` | run one image and write an overlay | the quickest way to see the model work |

## Run it

```bash
uv run python -m osid.model --index 0                 # query image from the manifest
uv run python -m osid.model --index 0 --reference     # reference crop instead
uv run python -m osid.model --image photo.jpg --top-k 256
```

It prints keypoint and descriptor statistics and writes `outputs/model-demo.png`
with points sized and coloured by boundary score.

## Synthetic data

`synth.py` produces pairs of a filled shape on a flat background, with each
knob set independently per view: shape family, size, rotation, perspective,
contrast against the background, Gaussian blur, and additive noise. Backgrounds
can differ between views, so the only shared signal is the shape.

Five families, so the model does not learn one shape prior: straight polygons
(3-10 sides), rounded polygons (Chaikin-smoothed corners), smooth blobs (summed
radial harmonics), stars (smooth lobes), and ellipses. Sizes run from 8% to 50%
of the canvas in radius, which is roughly 0.6% to 47% of the area.

```bash
uv run python -m osid.model.synth --samples 6              # contact sheet
uv run python -m osid.model.synth --samples 20 --check     # label sanity
uv run python -m osid.model.synth --grid                   # contrast × blur grid
uv run python -m osid.model.synth --samples 4 --seed 11    # reproducible sheet
```

Two views of the same generator. Random pairs show what a training batch looks
like — different backgrounds, shapes, scales and angles per view:

![Synthetic pairs: the same polygon on two different backgrounds with different contrast, blur and noise](../../../docs/devnotes/images/week4/synthetic_examples.png)

The grid holds everything fixed except contrast and blur, which is how the
knobs are checked rather than assumed:

![Synthetic grid: one polygon at four contrast levels and three blur levels](../../../docs/devnotes/images/week4/synthetic_grid.png)

`--check` verifies the label two ways. Projecting view A's polygon vertices
through the homography and comparing them with view B's gives a reprojection
error of **1e-13 px**, so the correspondence is exact. Warping the mask and
comparing silhouettes gives IoU **0.980 mean / 0.919 min**; the spread is edge
rasterisation, which bites hardest on the smallest shapes, not a label error.
Polygons are kept inside the canvas, since clipping one view but not the other
would invalidate the part of the correspondence that falls outside.

What this generator is for: isolating capability. If the model cannot find a
high-contrast polygon and match it under blur and noise, it will not do better
on a logo photographed in the wild. What it cannot provide is realism —
appearance change, texture, occlusion by other objects, real backgrounds. That
is the next generator, not this one.

## Losses

Two, and only two for the first attempt.

**Boundary heatmap loss.** The silhouette is rasterised into a Gaussian heatmap
at the decoder resolution (112×112, σ = 0.9 cells) and trained with the
penalty-reduced focal loss:

```text
L_heat = -1/N Σ (1-p)^α log p        on cells at the boundary, α = 2
         -1/N Σ (1-y)^β p^α log(1-p) elsewhere, β = 4
```

Full-resolution bands were ruled out: a 1 px edge at 448 px is a quarter of a
cell at stride 4, so the target would be almost empty.

**Descriptor loss.** Symmetric InfoNCE at τ = 0.07 over descriptors sampled at
corresponding boundary points, with an optional mask that excludes negatives
closer than a threshold in object space. Without that mask, a symmetric shape
(ellipse, regular star) penalises the model for a correct but ambiguous match.

```bash
uv run python -m osid.model.losses     # numerical checks
```

The checks confirm the three things that silently ruin training: the heatmap
peaks land on the true boundary (89.8 px against a 90 px circle), the focal loss
prefers an aligned prediction to an inverted one, and InfoNCE separates matched
pairs (0.003) from shuffled ones (14.7).

Deferred on purpose: coverage/spread regularisation, subpixel offset regression,
curvature weighting. Each is a refinement of a signal that has to exist first.

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

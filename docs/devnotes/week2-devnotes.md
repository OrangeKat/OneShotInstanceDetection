# Week 2 Devnotes: HPatches Validation and Failure Analysis

> **Status.** Week 2 integrated native HPatches metrics into the same benchmark
> and added a failure-analysis pass over the FlickrLogos-32 results.
>
> **Trace vs development.** §1–§2 and §4 are the initial implementation trace
> and are kept unchanged. §3.1–§3.2 are the original expected-vs-actual
> comparison; [§3.3](#33-reproducing-the-published-hpatches-values-2026-11-09)
> records the later reproduction work. §5 lists references.
>
> **Project arc.** Benchmark current solutions → diagnose failure modes → build
> the final pipeline / full model → publish a full report.
>
> **References.** Paper, dataset, and software links are inline or in
> [§5 References](#5-references).

## 1. What Changed

- Added a `--protocol hpatches` mode to `main.py` / `src/pipeline.py`.
- Kept one model registry (`MODELS`): adding a new matcher automatically makes
  it available for both the box benchmark and HPatches validation.
- Added `src/validation/` for HPatches sequence loading, MMA/HEA/corner-error
  metrics, and the Markdown report.
- Added `scripts/prepare_hpatches.py` for the sequence download and box-AP
  manifest.
- Added SuperPoint + LightGlue and ALIKED + LightGlue, using vendored
  [LightGlue](https://github.com/cvg/LightGlue) components and the same
  `FeatureMatcher` interface.
- Added [LoFTR](https://arxiv.org/abs/2104.00680) via Kornia,
  [RoMa v1](https://arxiv.org/abs/2305.15404) with vendored inference code, and
  [FlexGlue](https://arxiv.org/abs/2610.32976) with vendored inference code.
  The LoFTR and FlexGlue wrappers cap the long image side at 640 px and map
  keypoints back to the original resolution; unbounded HPatches inputs (up to
  6.3 MP) would otherwise exhaust memory.
- The box benchmark now streams samples per model instead of preloading every
  image, keeping memory bounded on the full five-dataset suite.
- Added failure analysis over the FlickrLogos-32 results: per-class, logo-size,
  multi-instance, confidence, and latency breakdowns.

There is no separate validation executable. The command is the benchmark:

```bash
make data-hpatches
make hpatches
```

## 2. HPatches Validation

**Initial implementation trace (2026-11-09).** This is the first complete
eight-model run. It uses the harness as originally written: native image
resolution, raw-pixel thresholds, and a fixed 3 px RANSAC threshold. The numbers
are preserved as-is; the protocol audit and re-measurement are in §3.3.

**Protocol.** 116 sequences, 580 reference/target pairs, 2,048 keypoints
(or correspondences for dense matchers), RANSAC threshold 3 px. Classical
matchers use a simple ratio test at 0.75; the learned matchers use their own
score filters. LoFTR and FlexGlue see images with the long side capped at
640 px.

- **MMA@k**: per-pair fraction of matches within `k` pixels after projecting
  reference keypoints through the ground-truth homography; averaged over pairs
  with at least one match.
- **HEA@k**: fraction of all pairs whose mean four-corner error is within `k`
  pixels; missing homographies count as failures.
- **Corner error**: mean Euclidean distance between the four reference corners
  projected by the ground-truth and estimated homographies.
- **AUC@k**: normalized area under the cumulative corner-error curve up to `k`
  pixels. Missing homographies count as infinite error.

| Model | MMA@1 | MMA@3 | MMA@5 | MMA@10 | HEA@1 | HEA@3 | HEA@5 | HEA@10 | AUC@1 | AUC@3 | AUC@5 | AUC@10 | Median CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIFT | 0.505 | 0.734 | 0.777 | 0.798 | 0.441 | 0.705 | 0.809 | 0.891 | 0.236 | 0.477 | 0.593 | 0.726 | 1.344 px |
| ORB | 0.385 | 0.720 | 0.789 | 0.814 | 0.245 | 0.529 | 0.667 | 0.772 | 0.123 | 0.318 | 0.436 | 0.584 | 2.515 px |
| AKAZE | 0.435 | 0.752 | 0.824 | 0.858 | 0.309 | 0.586 | 0.700 | 0.828 | 0.134 | 0.362 | 0.479 | 0.627 | 2.025 px |
| SuperPoint + LightGlue | 0.342 | 0.807 | 0.924 | 0.975 | 0.352 | 0.748 | 0.867 | 0.941 | 0.170 | 0.455 | 0.600 | 0.751 | 1.447 px |
| ALIKED + LightGlue | 0.502 | 0.864 | 0.942 | 0.977 | 0.440 | 0.778 | 0.888 | 0.933 | 0.223 | 0.505 | 0.643 | 0.778 | 1.181 px |
| LoFTR | 0.516 | 0.847 | 0.914 | 0.948 | 0.479 | 0.762 | 0.853 | 0.914 | 0.269 | 0.528 | 0.642 | 0.767 | 1.080 px |
| RoMa | 0.703 | 0.937 | 0.976 | 0.990 | 0.248 | 0.548 | 0.684 | 0.828 | 0.110 | 0.324 | 0.445 | 0.606 | 2.445 px |
| FlexGlue | 0.480 | 0.881 | 0.963 | 0.994 | 0.419 | 0.791 | 0.891 | 0.947 | 0.203 | 0.502 | 0.641 | 0.784 | 1.246 px |

Match filtering: the classical matchers use the 0.75 ratio test; LightGlue uses
`filter_threshold=0.1`, FlexGlue uses its own `filter_threshold=0.1`, LoFTR keeps
the highest-confidence correspondences, and RoMa thresholds certainty at 0.05.
`--ratio` only affects the classical rows.

[Efe et al. 2021](https://arxiv.org/abs/2108.08179) report SIFT MMA@3 / HEA@3 of
0.87 / 0.76, ORB 0.68 / 0.40, and AKAZE 0.80 / 0.65. Their protocol uses tuned
ratio tests, mutual nearest neighbors, 108 sequences, and
[OpenCV 4.5.2](https://docs.opencv.org/4.5.2/), so exact agreement is not
expected. The learned rows are measured under the same MMA/HEA protocol;
published SuperPoint/ALIKED HPatches numbers use homography AUC, so compare
within the same metric.

The measured AUC values are lower than the published learned-model AUC numbers
because the protocols differ in input resize, keypoint budget, RANSAC settings,
and corner-error definition. Use the measured AUC for relative comparison
within this harness, not as a reproduction of the published values.

Per split, viewpoint is harder than illumination for homography accuracy. The
gap is largest for RoMa (0.797 → 0.309) and smallest for FlexGlue (0.912 →
0.675):

| Model | Illumination HEA@3 | Viewpoint HEA@3 |
|---|---:|---:|
| SIFT | 0.849 | 0.566 |
| ORB | 0.670 | 0.393 |
| AKAZE | 0.719 | 0.458 |
| SuperPoint + LightGlue | 0.895 | 0.607 |
| ALIKED + LightGlue | 0.905 | 0.654 |
| LoFTR | 0.937 | 0.593 |
| RoMa | 0.797 | 0.309 |
| FlexGlue | 0.912 | 0.675 |

## 3. Expected HPatches Results for the Planned Suite

This table lists published HPatches reference values for the **released base
models**, with no HPatches fine-tuning. It is a sanity-check target, not one
leaderboard: protocols differ in keypoint budget, resize, RANSAC vs DLT, and
error thresholds.

| Model | Base weights / tuning | HPatches metric | Published value | Source |
|---|---|---|---|---|
| SIFT | OpenCV default, no training | MMA@3 / HEA@3 | 0.87 / 0.76 | [Efe et al. 2021](https://arxiv.org/abs/2108.08179) |
| ORB | OpenCV default, no training | MMA@3 / HEA@3 | 0.68 / 0.40 | [Efe et al. 2021](https://arxiv.org/abs/2108.08179) |
| AKAZE | OpenCV default, no training | MMA@3 / HEA@3 | 0.80 / 0.65 | [Efe et al. 2021](https://arxiv.org/abs/2108.08179) |
| SuperPoint + LightGlue | Official SuperPoint + LightGlue (MegaDepth) | Recall / precision / RANSAC AUC@1/5 | 94.3 / 88.9 / 38.3 / 79.6 | [LightGlue, Table 1](https://arxiv.org/abs/2306.13643) |
| ALIKED + LightGlue | Official ALIKED + LightGlue | AUC@3/5/10 px | 66.2 / 76.9 / 86.3 | [FlexGlue, Table 2](https://arxiv.org/abs/2610.32976) |
| LoFTR | Official LoFTR (MegaDepth) | Precision / RANSAC AUC@1/5 | 92.7 / 41.5 / 78.8 | [LightGlue, Table 1](https://arxiv.org/abs/2306.13643) |
| RoMa | Official RoMa v1 | RANSAC AUC@1/3/5 px | 44.7 / 72.6 / 81.4 | [MV-RoMa, Table 1](https://arxiv.org/abs/2603.27542) |
| FlexGlue | Official FlexGlue | AUC@3/5/10 px | 72.9 / 81.9 / 89.8 | [FlexGlue, Table 2](https://arxiv.org/abs/2610.32976) |
| DINOv2 / DINOv3 | Base ViT features + cosine | MMA / HEA | n/a | no standard HPatches matcher |

Notes:

- All learned rows use released pretrained weights and are **not fine-tuned on
  HPatches**.
- The metric column matters: classical rows use MMA/HEA, while the learned and
  dense rows use each paper's native precision/recall or homography-AUC setup.
  Do not compare values across different metrics.
- The classical values use MMA/HEA at 3 px with tuned ratio tests, mutual
  nearest neighbors, and 108 sequences. Our harness uses a simple ratio test at
  0.75 and all 116 sequences, so the classical numbers should be lower.
- DINOv2 and DINOv3 are foundation embeddings rather than standard HPatches
  matchers. No base-model HPatches sequence result exists; their numbers will
  only be comparable within this project.

### 3.1 Expected vs actual — implemented models

Only metrics that exist on both sides are compared. The Δ column mixes source
protocols and overstates the gap; see
[§3.3](#33-reproducing-the-published-hpatches-values-2026-11-09) for the audit.

| Model | Metric | Expected | Actual | Δ |
|---|---|---:|---:|---:|
| SIFT | MMA@3 | 0.870 | 0.734 | −0.136 |
| SIFT | HEA@3 | 0.760 | 0.705 | −0.055 |
| ORB | MMA@3 | 0.680 | 0.720 | +0.040 |
| ORB | HEA@3 | 0.400 | 0.529 | +0.129 |
| AKAZE | MMA@3 | 0.800 | 0.752 | −0.048 |
| AKAZE | HEA@3 | 0.650 | 0.586 | −0.064 |
| SuperPoint + LightGlue | AUC@1 | 0.383 | 0.170 | −0.213 |
| SuperPoint + LightGlue | AUC@5 | 0.796 | 0.600 | −0.196 |
| ALIKED + LightGlue | AUC@3 | 0.662 | 0.505 | −0.157 |
| ALIKED + LightGlue | AUC@5 | 0.769 | 0.643 | −0.126 |
| ALIKED + LightGlue | AUC@10 | 0.863 | 0.778 | −0.085 |
| LoFTR | AUC@1 | 0.415 | 0.269 | −0.146 |
| LoFTR | AUC@5 | 0.788 | 0.642 | −0.146 |
| RoMa | AUC@1 | 0.447 | 0.110 | −0.337 |
| RoMa | AUC@3 | 0.726 | 0.324 | −0.402 |
| RoMa | AUC@5 | 0.814 | 0.445 | −0.369 |
| FlexGlue | AUC@3 | 0.729 | 0.502 | −0.227 |
| FlexGlue | AUC@5 | 0.819 | 0.641 | −0.178 |
| FlexGlue | AUC@10 | 0.898 | 0.784 | −0.114 |

Observations:

- SIFT and AKAZE are slightly below the published classical values, while ORB
  is above them. This is consistent with protocol differences: the paper uses
  tuned ratio thresholds and mutual nearest neighbors, while this harness uses
  a simple ratio test at 0.75 with a 2,048-keypoint budget.
- SuperPoint + LightGlue and ALIKED + LightGlue are both below the published
  AUC values; between the two, the gap is largest for SuperPoint + LightGlue at
  AUC@1 and smallest for ALIKED + LightGlue at AUC@10.
- The learned models still produce strong MMA/HEA (ALIKED + LightGlue: 0.864 /
  0.778 at 3 px), so the AUC gap is more likely protocol-related than a
  completely broken matcher. Input resize, keypoint budget, RANSAC threshold,
  and corner-error definition all differ from the source papers.
- LoFTR, RoMa, and FlexGlue are also below their published AUC values, with
  LoFTR the closest (Δ ≈ −0.15) and RoMa the furthest (Δ ≈ −0.34 to −0.40).
  RoMa's gap is notable because it has the best MMA@3 of any model (0.937) but
  only 0.548 HEA@3: its correspondences are accurate, while the homography
  fitted from 2,048 dense matches is less stable. FlexGlue's gap shrinks with
  the threshold (−0.227 at 3 px to −0.114 at 10 px), so its main deficit is
  fine-grained corner accuracy.
- The measured AUC should be used for relative comparison inside this harness,
  not as a claim that the published AUC was reproduced.

### 3.2 Expected vs actual — not yet measured

LoFTR, RoMa, and FlexGlue are now implemented and measured, so their
expected-vs-actual rows moved to [§3.1](#31-expected-vs-actual--implemented-models).
Only the foundation-embedding baselines remain unmeasured.

| Model | Expected metric | Expected | Status |
|---|---|---|---|
| DINOv2 / DINOv3 | MMA / HEA | n/a | no standard HPatches result |

### 3.3 Reproducing the published HPatches values (2026-11-09)

Added after the initial trace; §2 stays as the recorded baseline.

**Trigger.** Every learned model landed below its published HPatches AUC by more
than the expected run-to-run variation.

**Main finding: the evaluation protocol accounts for most of the gap, not the
matchers.** The official HPatches homography protocol resizes images so the
**shorter side is 480 px** and reports corner error and the RANSAC inlier
threshold in that resized space:

- [LightGlue §5.1](https://arxiv.org/abs/2306.13643): "resize all images such
  that their smaller dimension is equal to 480 pixels".
- The [RoMa HPatches
  benchmark](https://github.com/Parskatt/RoMa/blob/master/romatch/benchmarks/hpatches_sequences_homog_benchmark.py)
  divides corner error and the inlier threshold by `min(H, W) / 480`:
  `ransacReprojThreshold = 3 * min(w2, h2) / 480`.

Our initial run keeps native resolution (up to 3,088 × 2,056) and a fixed 3 px
threshold. On the largest pair, 3 px is ≈ 0.70 px at 480, so an identical
homography scores much worse here. Two smaller differences:

- The official benchmark ignores eight sequences, the same list as LoFTR/RoMa:
  `i_contruction`, `i_crownnight`, `i_dc`, `i_pencils`, `i_whitebuilding`,
  `v_artisans`, `v_astronautis`, `v_talent`. Our run includes them.
- RoMa's benchmark samples 5,000 matches with certainty-weighted, KDE-balanced
  sampling. Our wrapper keeps the 2,048 highest-certainty grid points, which
  cluster spatially. This explains RoMa's MMA@3 = 0.937 with HEA@3 = 0.548: the
  correspondences are accurate, but the homography is fitted from a biased
  region.

#### Reanalysis with normalized corner error (no re-run)

Stored per-pair corner errors rescaled by `min(H, W) / 480`:

| Model | AUC@1 raw | AUC@1 norm | AUC@10 raw | AUC@10 norm | AUC@10 norm, 8 ignored |
|---|---:|---:|---:|---:|---:|
| SIFT | 0.236 | 0.310 | 0.726 | 0.796 | 0.793 |
| ORB | 0.123 | 0.179 | 0.584 | 0.665 | 0.661 |
| AKAZE | 0.134 | 0.203 | 0.627 | 0.711 | 0.701 |
| SuperPoint + LightGlue | 0.170 | 0.246 | 0.751 | 0.833 | 0.831 |
| ALIKED + LightGlue | 0.223 | 0.314 | 0.778 | 0.848 | 0.845 |
| LoFTR | 0.269 | 0.355 | 0.767 | 0.836 | 0.834 |
| RoMa | 0.110 | 0.164 | 0.606 | 0.712 | 0.717 |
| FlexGlue | 0.203 | 0.293 | 0.784 | 0.857 | 0.856 |

Normalization alone adds 0.07–0.11 AUC@10. Dropping the eight ignored sequences
moves AUC@10 by at most 0.01, so that is a minor factor.

#### Re-run under the published 480 px protocol

Both images resized so the shorter side is 480 px, the ground-truth homography
scaled by `H' = S_query · H · S_reference⁻¹`, the same 3 px RANSAC threshold now
in 480-space, and AUC computed on the resized corners. SuperPoint + LightGlue
uses 1,024 keypoints to match the LightGlue setup; the others keep 2,048.

| Model | AUC@1 | AUC@3 | AUC@5 | AUC@10 | Published | Δ at matched thresholds |
|---|---:|---:|---:|---:|---|---:|
| SuperPoint + LightGlue | 0.269 | 0.596 | 0.721 | 0.840 | 0.383 @1 / 0.796 @5 | −0.114 / −0.075 |
| ALIKED + LightGlue | 0.323 | 0.640 | 0.754 | 0.854 | 0.662 / 0.769 / 0.863 @3/5/10 | −0.022 / −0.015 / −0.009 |
| FlexGlue | 0.325 | 0.657 | 0.772 | 0.875 | 0.729 / 0.819 / 0.898 @3/5/10 | −0.072 / −0.047 / −0.023 |
| LoFTR | 0.388 | 0.669 | 0.767 | 0.856 | 0.415 @1 / 0.788 @5 | −0.027 / −0.021 |

Reading:

- ALIKED + LightGlue lands within 0.02 of the FlexGlue paper, so the protocol
  explains almost its whole initial gap.
- FlexGlue is closest at 10 px and furthest at 3 px. Its paper also reports a
  640 px variant, so our 480 px run is not a like-for-like reproduction.
- SuperPoint + LightGlue remains furthest below at 1 px. [LightGlue Table
  1](https://arxiv.org/abs/2306.13643) uses MAGSAC++ with per-method tuned inlier
  thresholds, while this harness uses fixed 3 px RANSAC.
- LoFTR is close under the same protocol.

**Not yet done.** RoMa has not been rerun with the 480 px protocol and balanced
sampling. Even with normalized error, its AUC@5 is 0.548 against 0.814
published, the largest remaining gap, consistent with the sampling issue above.

**Consequence for §3.1.** The expected values come from three different
protocols (LightGlue Table 1 with MAGSAC++, FlexGlue Table 2, MV-RoMa). The Δ
column is a rough sanity check, not a reproduction error.

**Next actions.**

- Decide whether the HPatches harness should adopt the 480-short-side,
  normalized-corner-error protocol. If yes, rerun all eight models and report
  those as the baseline, keeping §2 as the trace.
- Rerun RoMa with `sample(..., 5000)` balanced sampling and a normalized RANSAC
  threshold.
- Record each later harness or model change as a dated entry under §3.3.

## 4. FlickrLogos-32 Failure Analysis

The failure analysis splits every pair into hit, wrong box, no prediction,
false positive, or true negative.

| Model | Hit | Wrong box | No prediction | False positive |
|---|---:|---:|---:|---:|
| SIFT | 36.1% | 28.0% | 23.2% | 3.7% |
| ORB | 18.8% | 26.2% | 42.3% | 3.8% |
| AKAZE | 20.8% | 15.7% | 50.8% | 0.4% |

- SIFT fails most often with a wrong box: it finds the class but locks onto the
  wrong instance or background.
- ORB and AKAZE fail mostly by producing no prediction.
- Small logos are the main weakness. Below 1% of the query image, SIFT hits
  0.218 while ORB and AKAZE hit 0.000. At 5% or more, the rates are 0.460,
  0.292, and 0.326.
- Multi-instance queries are harder: SIFT 0.430 → 0.358, ORB 0.237 → 0.143,
  AKAZE 0.261 → 0.163 (single → multi).
- Worst SIFT classes include `apple`, `carlsberg`, `nvidia`, `pepsi`, `shell`.

## 5. References

- Week 1 design and bibliography:
  [week1-devnotes.md](week1-devnotes.md)
- U. Efe, K. G. Ince, A. A. Alatan. 2021. **Effect of Parameter Optimization on
  Classical and Learning-based Image Matching Methods.** *ICCVW*.
  [arXiv](https://arxiv.org/abs/2108.08179)
- V. Balntas, K. Lenc, A. Vedaldi, K. Mikolajczyk. 2017. **HPatches: A Benchmark
  and Evaluation of Handcrafted and Learned Local Descriptors.** *CVPR*.
  [arXiv](https://arxiv.org/abs/1704.05939)
- P. Lindenberger, P.-E. Sarlin, M. Pollefeys. 2023. **LightGlue: Local Feature
  Matching at Light Speed.** *ICCV*. [arXiv](https://arxiv.org/abs/2306.13643)
- J. Sun, Z. Shen, Y. Wang, H. Bao, X. Zhou. 2021. **LoFTR: Detector-Free Local
  Feature Matching with Transformers.** *CVPR*.
  [arXiv](https://arxiv.org/abs/2104.00680)
- J. Edstedt, Q. Sun, G. Bökman, M. Wadenbäck, M. Felsberg. 2024. **RoMa:
  Robust Dense Feature Matching.** *CVPR*.
  [arXiv](https://arxiv.org/abs/2305.15404)
- P. Lindenberger, P.-E. Sarlin, O. Miksik, M. Pollefeys. 2026. **FlexGlue:
  Sparse-to-Dense Feature Matching and Visual Disambiguation.** *arXiv*.
  [arXiv](https://arxiv.org/abs/2610.32976)
- J. Lee, S. Kang, S. Yoo. 2026. **MV-RoMa: From Pairwise Matching into
  Multi-View Track Reconstruction.** *arXiv*.
  [arXiv](https://arxiv.org/abs/2603.27542)

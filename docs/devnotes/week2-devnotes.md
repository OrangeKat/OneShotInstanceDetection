# Week 2 Devnotes: HPatches Validation and Failure Analysis

> **Status.** Week 2 integrated native HPatches metrics into the same benchmark
> and added a failure-analysis pass over the FlickrLogos-32 results.
>
> **Project arc.** Benchmark current solutions → diagnose failure modes → build
> the final pipeline / full model → publish a full report.
>
> **References.** Paper, dataset, and software links are inline or in
> [§6 References](#6-references).

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
- Added failure analysis over the FlickrLogos-32 results: per-class, logo-size,
  multi-instance, confidence, and latency breakdowns.

There is no separate validation executable. The command is the benchmark:

```bash
make data-hpatches
make hpatches
```

## 2. HPatches Validation

**Protocol.** 116 sequences, 580 reference/target pairs, 2,048 keypoints,
simple ratio test at 0.75, RANSAC threshold 3 px.

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

The learned models use LightGlue's own match-score filter
(`filter_threshold=0.1`); `--ratio` only applies to the classical matchers.

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

Per split, viewpoint is harder than illumination for homography accuracy:

| Model | Illumination HEA@3 | Viewpoint HEA@3 |
|---|---:|---:|
| SIFT | 0.849 | 0.566 |
| ORB | 0.670 | 0.393 |
| AKAZE | 0.719 | 0.458 |

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

Only metrics that exist on both sides are compared.

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

Observations:

- SIFT and AKAZE are slightly below the published classical values, while ORB
  is above them. This is consistent with protocol differences: the paper uses
  tuned ratio thresholds and mutual nearest neighbors, while this harness uses
  a simple ratio test at 0.75 with a 2,048-keypoint budget.
- Both learned models are below the published AUC values. The gap is largest
  for SuperPoint + LightGlue at AUC@1 and smallest for ALIKED + LightGlue at
  AUC@10.
- The learned models still produce strong MMA/HEA (ALIKED + LightGlue: 0.864 /
  0.778 at 3 px), so the AUC gap is more likely protocol-related than a
  completely broken matcher. Input resize, keypoint budget, RANSAC threshold,
  and corner-error definition all differ from the source papers.
- The measured AUC should be used for relative comparison inside this harness,
  not as a claim that the published AUC was reproduced.

### 3.2 Expected vs actual — not yet measured

| Model | Expected metric | Expected | Status |
|---|---|---|---|
| LoFTR | RANSAC AUC@1/5 | 0.415 / 0.788 | not implemented |
| RoMa | RANSAC AUC@1/3/5 | 0.447 / 0.726 / 0.814 | not implemented |
| FlexGlue | AUC@3/5/10 | 0.729 / 0.819 / 0.898 | not implemented |
| DINOv2 / DINOv3 | MMA / HEA | n/a | no standard HPatches result |

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

## 5. Next Steps

- Add the next model to `MODELS` and run both protocols; no new benchmark code
  should be needed.
- Improve small-logo handling: multi-scale matching, higher-resolution inputs,
  or a proposal/verification stage.
- Handle multi-instance queries with sequential RANSAC or proposal-verify.
- Use the failure analysis to choose the final pipeline, then write the final
  report.

## 6. References

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

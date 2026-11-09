# Week 1 Devnotes: One-Shot Instance Detection

> **Status.** The design below is implemented for the first dataset and the
> three classical baselines. See §6 for the current harness, results, and
> limitations.
>
> **Project arc.** Benchmark current solutions → diagnose failure modes → build
> the final pipeline / full model → publish a full report. One devlog per week,
> published on LinkedIn and kept in this folder.
>
> **References.** Paper, dataset, and software links are inline or in
> [§7 Bibliography](#7-bibliography).

## Contents

- [1. Goal](#1-goal)
- [2. Model Options](#2-model-options)
- [3. Dataset Options](#3-dataset-options)
- [4. Starting the Project](#4-starting-the-project)
- [5. Choosing the Models](#5-choosing-the-models)
- [6. Current Implementation](#6-current-implementation)
- [7. Bibliography](#7-bibliography)

---

## 1. Goal

Instance-level detection with minimal / no training.

| Aspect | Decision |
|---|---|
| Input | a reference image, or a crop of one specific instance |
| Output | a box for that same instance in a query image (or all occurrences) |
| Task | **instance-level matching** — not category detection, not tracking |
| Modality | RGB images only |
| Training | training-free by default; fine-tuning pretrained models is fine; from scratch is out of scope (whole project, not just week 1) |

"Find every instance" is not one RANSAC homography. A single fit returns one dominant instance, so the pipeline needs a multi-model step:

- sequential RANSAC (fit, remove inliers, repeat)
- J-Linkage / T-Linkage
- proposal-then-verify
- generalized Hough voting

Keep the evaluation protocol fixed across models:

```
reference crop → query image → predicted box → AP@0.5
```

## 2. Model Options

### 2.1 Local Feature Matching

Baseline pipeline:

```
keypoints → descriptors → match → ratio test / mutual NN → RANSAC homography → box
```

Keep this pipeline fixed so the models are actually comparable.

#### 2.1.1 Classical local features

| Descriptor | Matcher | Role |
|---|---|---|
| [SIFT](https://doi.org/10.1023/B:VISI.0000029664.99615.94) | BF-L2 or FLANN KD-tree | accuracy baseline |
| [ORB](https://doi.org/10.1109/ICCV.2011.6126544) | BF-Hamming | speed baseline |
| [AKAZE](https://doi.org/10.5244/C.27.13) | BF-Hamming | middle ground |

- FLANN works for binary descriptors only with LSH; KD-tree is for float descriptors.
- No training. CPU in OpenCV; GPU via [Kornia](https://kornia.github.io/) if needed.
- Weak on textureless objects, deformable objects, and severe viewpoint / illumination change.
- [NCC template matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html) is the cheap sanity baseline, especially for logos and products.

#### 2.1.2 Learned local features

| Role | Options |
|---|---|
| Front-end (detector + descriptor) | [SuperPoint](https://arxiv.org/abs/1712.07629), [ALIKED](https://arxiv.org/abs/2304.03608) · optional: [XFeat](https://arxiv.org/abs/2404.19174) |
| Matcher | [LightGlue](https://arxiv.org/abs/2306.13643) (default) · optional: [OmniGlue](https://arxiv.org/abs/2405.12979), [SuperGlue](https://arxiv.org/abs/1911.11763) (legacy) |

- SuperGlue's original implementation uses custom CUDA ops → avoid on ROCm. LightGlue is the safer default.
- OmniGlue is slower than LightGlue but is the more relevant cross-domain baseline.

**OmniGlue's DINOv2 guidance**

- SuperPoint supplies the keypoints and their local descriptors `d_i`; these are the descriptors that are refined and finally matched.
- DINOv2 dense feature maps are **bilinearly interpolated** at the SuperPoint keypoint locations → per-keypoint guidance descriptor `g_i`.
- `g_i` (not `d_i`) is used to prune the inter-image graph (top 50% most similar) and guide cross-attention.
- DINOv2 is **not** the final descriptor; the matching layer scores the refined SuperPoint `d_i` with a dot product + Sinkhorn.
- Position-guided attention: query/key use `d + p`, value uses `d` only.
- Ablations: merging DINO into the descriptors hurts; DINO guidance only on cross-attention helps.
- Designed for cross-domain generalization (object-centric, aerial, indoor), which matches the multi-category goal.

#### 2.1.3 Detector-free dense matching

| Model | Notes |
|---|---|
| [LoFTR](https://arxiv.org/abs/2104.00680) | standard dense baseline; pure PyTorch |
| [RoMa](https://arxiv.org/abs/2305.15404) (v1) | stronger; DINOv2-based; pure PyTorch |
| [RoMa v2](https://arxiv.org/abs/2511.15706) | better / faster on paper; custom CUDA kernel → ROCm risk; optional |
| [FlexGlue](https://arxiv.org/abs/2610.32976) | 2026; DINOv3-ConvNeXt encoder; sparse-to-dense; covisibility score rejects look-alike pairs; advanced, not a baseline |

Dense matchers output correspondences; derive the box via homography or matched-region clustering.

### 2.2 Foundation-model Embedding Matching

| Model | License | Size / fit | Notes |
|---|---|---|---|
| [DINOv2](https://arxiv.org/abs/2304.07193) | Apache 2.0 | ViT-S/B/L/g | patch tokens localize; CLS token retrieves |
| [DINOv3](https://arxiv.org/abs/2508.10104) | custom Meta license | ViT-S (21M) … ViT-7B (6.7B) | only ViT-S/B are realistic on 16 GB |

- Avoid language prompts: too ambiguous for instance-level matching, and they need larger models.
- "Cosine similarity" alone is underspecified. Pin down:
  - patch-level features for localization, CLS token for retrieval
  - which layer(s) to use
  - L2 normalization, mutual nearest neighbor, ratio test
  - multi-scale
  - box from RANSAC / homography or clustering
- OmniGlue's ablation is the useful lesson: DINOv2 as the sole descriptor is weak; it works better as candidate pruning / guidance for a local matcher.

## 3. Dataset Options

Goal: instance detection across **Buildings, Logos, People, Objects**. No single benchmark covers all four; each dataset has a different task and annotation format.

| Category | Dataset | Task / annotations | Metric | Note |
|---|---|---|---|---|
| Logos | [FlickrLogos-32](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos) | 32 classes, boxes, multiple logos per image | AP@0.5 | primary logo benchmark; rigid and textured |
| People | [CUHK-SYSU](https://github.com/ShuangLI59/person_search) | person search, boxes + IDs | mAP, top-1 | closest standard task to instance detection |
| Buildings | [Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) | retrieval, query ROIs | mAP, recall@k | not detection boxes |
| Objects | [RPC](https://arxiv.org/abs/1901.07249) | category-level product detection, boxes | mAP | no cross-image instance identity |
| Validation | [HPatches](https://github.com/hpatches/hpatches-dataset) | planar pairs, homographies | MMA, HEA, corner error | matcher validation; not detection AP |

**Protocol notes**

| Item | Protocol |
|---|---|
| Common | reference crop → query image → predicted box → AP@0.5 |
| Paris6k | report mAP / recall@k; project the query ROI through the estimated homography for a box IoU |
| HPatches | report MMA@1/3/5/10, HEA@1/3/5/10, and corner error; not detection AP |
| Mixing | do not mix category-level and instance-level AP in one table |
| Access | FlickrLogos-32: email request (the prep script uses the public OpenLogo mirror) · CUHK-SYSU: request form · Paris6k: original links may be dead · RPC: direct download · HPatches: direct download |

> **Current harness note.** FlickrLogos-32 is the first dataset wired into the
> manifest → box → AP path. See §6 for the exact split, results, and
> limitations.

## 4. Starting the Project

- The benchmarking tool was built first: dataset → pipeline → accuracy / speed
  → JSON + Markdown. Size and memory accounting are still planned.
- Baselines and metrics were fixed before tuning anything.

**Metrics**

The current harness records accuracy, speed, and match diagnostics. Size and
memory accounting are planned.

| Axis | Recorded now |
|---|---|
| Accuracy | hit rate at IoU@0.5, 11-point AP@0.5, mean IoU |
| Speed | per-pair latency mean / p50 / p95 |
| Diagnostics | match count and RANSAC inlier count (keypoint budget in config) |
| Per-sample output | query, predicted box, GT boxes, hit, latency |
| Planned | params + disk size, peak GPU / CPU memory |

- Reproducibility: fixed keypoint budget, input resolution, RANSAC threshold,
  and Lowe ratio. OpenCV RANSAC is not seeded, so small run-to-run variation is
  expected.
- Multi-instance: sequential RANSAC / J-Linkage / proposal-verify.

## 5. Choosing the Models

Want relevant models / best in each category, but keep at least one baseline per layer.

### 5.1 Baseline vs advanced

| Layer | Baseline | Advanced |
|---|---|---|
| Classical | SIFT + BF-L2 / FLANN-KDTree, ORB + BF-Hamming, AKAZE + BF-Hamming | NCC template matching |
| Learned sparse | SuperPoint + LightGlue, ALIKED + LightGlue | OmniGlue, XFeat |
| Dense | LoFTR, RoMa v1 | RoMa v2, FlexGlue |
| Foundation | DINOv2 ViT-B/14 patch cosine | DINOv3 ViT-B/16 |
| Retrieval (buildings) | DINOv2 CLS/GeM + FAISS | VLAD/GeM + local verification |
| People | DINOv2 embeddings + NN | person-search model |
| Multi-instance | sequential RANSAC | J-Linkage / proposal-verify |

### 5.2 Selected suite

| Family | Models |
|---|---|
| Local feature matching | SIFT/BF-L2 (FLANN-KDTree optional), ORB/BF-Hamming, AKAZE/BF-Hamming, SuperPoint/LightGlue, ALIKED/LightGlue, LoFTR, RoMa, FlexGlue |
| Foundation embeddings | DINOv2 / cosine, DINOv3 / cosine |

### 5.3 Rationale

- SIFT / ORB / AKAZE: training-free, interpretable, cover the accuracy/speed range.
- SuperPoint + LightGlue, ALIKED + LightGlue: learned sparse baselines.
- LoFTR / RoMa: dense baselines; RoMa v2 and FlexGlue are advanced, not baselines.
- DINOv2 / DINOv3: foundation features; best used for pruning / retrieval, not as the sole descriptor.

### 5.4 Fine-tuning candidates

Allowed, but not training from scratch:

- OmniGlue low-shot fine-tuning on the target domain — the paper shows gains from a few dozen target instances.
- LightGlue / SuperPoint fine-tuning on synthetic homographies or the target category.
- DINOv2 linear probe / adapter for people or product instances.
- Person-search models fine-tuned on CUHK-SYSU.

## 6. Current Implementation

### 6.1 Harness

- Entry point: `main.py`; implementation: `src/pipeline.py`.
- Dataset: manifest-backed `InstanceDataset` in `src/utils/dataset.py` —
  JSON or JSONL, absolute boxes, optional reference crop, negative queries.
- Models: `src/models/feature_matching/classical/{sift,orb,akaze}.py`, with shared
  matching and RANSAC helpers in `base.py`.
- Metrics: box protocol — hit rate at IoU@0.5, 11-point AP@0.5, mean IoU,
  match/inlier counts, latency.
- Outputs: `results.json`, `results.md`, and optional annotated examples.
- Analysis: `scripts/analyze_results.py` is a thin entry point; data, metrics,
  plots, tables, and report rendering live in `src/analysis/`.
- CLI: `--manifest`, `--root`, `--models`, `--output`, `--max-samples`,
  `--iou-threshold`, `--ransac-threshold`, `--max-keypoints`, `--ratio`.

### 6.2 Dataset split: FlickrLogos-32

- The official download requires an email request. The prep script
  (`scripts/prepare_flickrlogos32.py`) pulls the 2,240 positive FlickrLogos-32
  images from the public [`Voxel51/OpenLogo`](https://huggingface.co/datasets/Voxel51/OpenLogo)
  mirror and rebuilds the 32 classes.
- OpenLogo text labels `adidas_text` and `heineken_text` are folded into
  `adidas` and `heineken`.
- Split per class: one reference crop (the median-area logo box), 69 positive
  queries, and 10 cross-class negative queries.
- Total: **2,208 positives + 320 negatives = 2,528 pairs**.
- The official P1/P2/P3 split and the original 3,000 no-logo negatives are not
  preserved. The official split is therefore not comparable to published
  FlickrLogos-32 mAP.
- Median-area reference selection avoids huge close-up crops whose background
  texture dominates matching; selecting the largest logo crop performed
  substantially worse.

### 6.3 Models

- SIFT with BF-L2, ORB with BF-Hamming, AKAZE with BF-Hamming.
- Common settings: 2,048 keypoints, Lowe ratio 0.75, RANSAC reprojection
  threshold 3 px, 5,000 iterations, confidence 0.999.
- OpenCV 5.0 moved AKAZE to `cv2.xfeatures2d`, so the project depends on
  `opencv-contrib-python`.
- One RANSAC homography returns one box per reference/query pair. Multiple
  occurrences need a multi-model step (see §1).

### 6.4 Results

| Model | Hit rate | AP@0.5 | Mean IoU | Mean matches | Mean inliers | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT | 0.413 | 0.406 | 0.352 | 21.3 | 12.1 | 64.5 ms | 82.7 ms |
| ORB | 0.215 | 0.237 | 0.196 | 13.0 | 7.7 | 9.1 ms | 17.5 ms |
| AKAZE | 0.239 | 0.251 | 0.209 | 7.3 | 5.1 | 48.9 ms | 63.3 ms |

These values sit in the expected range for a training-free one-shot matcher:
SIFT is strongest, AKAZE is second, ORB is fastest but weakest. They are below
supervised FlickrLogos-32 detectors (~0.84–0.87 mAP;
[Zhang et al. 2021](https://arxiv.org/abs/2108.13775)) and trained few-shot
models (~0.57–0.67 mAP;
[Fehervari & Appalaraju 2019](https://arxiv.org/abs/1811.08009);
[Bhunia et al. 2019](https://arxiv.org/abs/1811.01395)), which is expected
without training or the official multi-reference split.

### 6.5 Failure analysis

`scripts/analyze_results.py` writes plots, CSVs, and `analysis.md` to
`outputs/flickrlogos32/analysis/`. Failure categories are hit, wrong box, no
prediction, false positive, and true negative.

| Model | Hit | Wrong box | No prediction | False positive |
|---|---:|---:|---:|---:|
| SIFT | 36.1% | 28.0% | 23.2% | 3.7% |
| ORB | 18.8% | 26.2% | 42.3% | 3.8% |
| AKAZE | 20.8% | 15.7% | 50.8% | 0.4% |

- **SIFT** fails most often with a wrong box: it finds the class but locks onto
  the wrong instance or background.
- **ORB and AKAZE** fail mostly by producing no prediction at all.
- **Small logos are the main weakness.** For logos below 1% of the query image,
  SIFT hits 0.218 while ORB and AKAZE hit 0.000. For logos at least 5%, the
  rates are SIFT 0.460, ORB 0.292, AKAZE 0.326.
- **Multi-instance queries are harder** for every model: SIFT 0.430 → 0.358,
  ORB 0.237 → 0.143, AKAZE 0.261 → 0.163 (single → multi).
- **Inlier count is a useful but imperfect confidence score**: its correlation
  with IoU is 0.67 for SIFT, 0.66 for ORB, and 0.73 for AKAZE.
- Worst SIFT classes: `apple`, `carlsberg`, `nvidia`, `pepsi`, `shell`. Best
  classes: `paulaner`, `rittersport`, `starbucks`, `fosters`, `becks`.

Generated plots: failure breakdown, per-class hit-rate and mean-IoU heatmaps,
IoU distribution, hit rate by logo size and GT-instance count, inliers vs IoU,
precision-recall, and latency distribution.

### 6.6 Limitations and next steps

- One reference crop per class, not the official 10-image reference set.
- No official train/val/test split; cross-class negatives replace the original
  no-logo negatives.
- AP is computed per reference/query pair with inlier count as confidence, not
  per-image detection AP over proposals.
- Single-instance output only; no multi-instance handling yet.
- Classical features are weak on low-texture logos and large scale/domain gaps.
- Size and memory reporting are not implemented.
- Next: NCC baseline, SuperPoint/LightGlue, CUHK-SYSU, multi-instance
  handling, and size/memory accounting.

## 7. Bibliography

### Classical local features

- D. G. Lowe. 2004. **Distinctive Image Features from Scale-Invariant Keypoints.** *IJCV* 60(2). [doi](https://doi.org/10.1023/B:VISI.0000029664.99615.94)
- E. Rublee, V. Rabaud, K. Konolige, G. Bradski. 2011. **ORB: An Efficient Alternative to SIFT or SURF.** *ICCV*. [doi](https://doi.org/10.1109/ICCV.2011.6126544)
- P. F. Alcantarilla, J. Nuevo, A. Bartoli. 2013. **Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces.** *BMVC*. [doi](https://doi.org/10.5244/C.27.13)

### Learned local features and matchers

- D. DeTone, T. Malisiewicz, A. Rabinovich. 2018. **SuperPoint: Self-Supervised Interest Point Detection and Description.** *CVPR Workshops*. [arXiv](https://arxiv.org/abs/1712.07629)
- P.-E. Sarlin, D. DeTone, T. Malisiewicz, A. Rabinovich. 2020. **SuperGlue: Learning Feature Matching with Graph Neural Networks.** *CVPR*. [arXiv](https://arxiv.org/abs/1911.11763)
- P. Lindenberger, P.-E. Sarlin, M. Pollefeys. 2023. **LightGlue: Local Feature Matching at Light Speed.** *ICCV*. [arXiv](https://arxiv.org/abs/2306.13643)
- H. Jiang, A. Karpur, B. Cao, Q. Huang, A. Araujo. 2024. **OmniGlue: Generalizable Feature Matching with Foundation Model Guidance.** *CVPR*. [arXiv](https://arxiv.org/abs/2405.12979)
- X. Zhao, X. Wu, J. Miao, W. Chen, P. C. Y. Chen, Z. Li. 2023. **ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation.** *IEEE TIM*. [arXiv](https://arxiv.org/abs/2304.03608)
- G. Potje, F. Cadar, A. Araujo, R. Martins, E. R. Nascimento. 2024. **XFeat: Accelerated Features for Lightweight Image Matching.** *CVPR*. [arXiv](https://arxiv.org/abs/2404.19174)

### Dense matching

- J. Sun, Z. Shen, Y. Wang, H. Bao, X. Zhou. 2021. **LoFTR: Detector-Free Local Feature Matching with Transformers.** *CVPR*. [arXiv](https://arxiv.org/abs/2104.00680)
- J. Edstedt, Q. Sun, G. Bökman, M. Wadenbäck, M. Felsberg. 2024. **RoMa: Robust Dense Feature Matching.** *CVPR*. [arXiv](https://arxiv.org/abs/2305.15404)
- J. Edstedt, D. Nordström, Y. Zhang, G. Bökman, J. Astermark, A. Heyden, V. Larsson, M. Wadenbäck, M. Felsberg, F. Kahl. 2026. **RoMa v2: Harder Better Faster Denser Feature Matching.** *ECCV*. [arXiv](https://arxiv.org/abs/2511.15706)
- P. Lindenberger, P.-E. Sarlin, O. Miksik, M. Pollefeys. 2026. **FlexGlue: Sparse-to-Dense Feature Matching and Visual Disambiguation.** *arXiv*. [arXiv](https://arxiv.org/abs/2610.32976)

### Foundation models

- M. Oquab, T. Darcet, T. Moutakanni, et al. 2024. **DINOv2: Learning Robust Visual Features without Supervision.** *TMLR*. [arXiv](https://arxiv.org/abs/2304.07193)
- O. Siméoni, H. V. Vo, M. Seitzer, et al. 2025. **DINOv3.** *arXiv*. [arXiv](https://arxiv.org/abs/2508.10104)

### Robust estimation / multi-model fitting

- M. A. Fischler, R. C. Bolles. 1981. **Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography.** *CACM* 24(6). [doi](https://doi.org/10.1145/358669.358692)
- D. Barath, J. Noskova, M. Ivashechkin, J. Matas. 2020. **MAGSAC++, a Fast, Reliable and Accurate Robust Estimator.** *CVPR*. [arXiv](https://arxiv.org/abs/1912.05909)
- R. Toldo, A. Fusiello. 2008. **Robust Multiple Structures Estimation with J-Linkage.** *ECCV*. [doi](https://doi.org/10.1007/978-3-540-88682-2_41)

### Logo detection / retrieval

- I. Fehervari, S. Appalaraju. 2019. **Scalable Logo Recognition using Proxies.** *WACV*. [arXiv](https://arxiv.org/abs/1811.08009)
- A. K. Bhunia, A. K. Bhunia, S. Ghose, A. Das, P. P. Roy, U. Pal. 2019. **A Deep One-Shot Network for Query-based Logo Retrieval.** *arXiv*. [arXiv](https://arxiv.org/abs/1811.01395)
- B. Zhang, W. Min, J. Wang, S. Hou, Q. Hou, Y. Zheng, S. Jiang. 2021. **Discriminative Semantic Feature Pyramid Network with Guided Anchoring for Logo Detection.** *arXiv*. [arXiv](https://arxiv.org/abs/2108.13775)

### Benchmarks / evaluation

- U. Efe, K. G. Ince, A. A. Alatan. 2021. **Effect of Parameter Optimization on Classical and Learning-based Image Matching Methods.** *ICCVW*. [arXiv](https://arxiv.org/abs/2108.08179)

### Datasets

- V. Balntas, K. Lenc, A. Vedaldi, K. Mikolajczyk. 2017. **HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors.** *CVPR*. [arXiv](https://arxiv.org/abs/1704.05939) · [dataset](https://github.com/hpatches/hpatches-dataset)
- J. Philbin, O. Chum, M. Isard, J. Sivic, A. Zisserman. 2008. **Lost in Quantization: Improving Particular Object Retrieval in Large Scale Image Databases.** *CVPR*. [doi](https://doi.org/10.1109/CVPR.2008.4587635) · [Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)
- S. Romberg, L. G. Pueyo, R. Lienhart, R. van Zwol. 2011. **Scalable Logo Recognition in Real-World Images.** *ICMR*. [doi](https://doi.org/10.1145/1991996.1992028) · [dataset](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos)
- T. Xiao, S. Li, B. Wang, L. Lin, X. Wang. 2017. **Joint Detection and Identification Feature Learning for Person Search.** *CVPR*. [arXiv](https://arxiv.org/abs/1604.01850) · [dataset](https://github.com/ShuangLI59/person_search)
- X.-S. Wei, Q. Cui, L. Yang, P. Wang, L. Liu. 2019. **RPC: A Large-Scale Retail Product Checkout Dataset.** *arXiv*. [arXiv](https://arxiv.org/abs/1901.07249)

### Software / docs

- OpenCV. **Template Matching.** [docs](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- OpenCV. **OpenCV 5.0 documentation.** [docs](https://docs.opencv.org/5.0/)
- Kornia. **Differentiable Computer Vision Library.** [docs](https://kornia.github.io/)
- Voxel51. **OpenLogo.** [dataset](https://huggingface.co/datasets/Voxel51/OpenLogo) — public mirror used to obtain the FlickrLogos-32 positive images.
- Voxel51. **OpenLogo.** [dataset](https://huggingface.co/datasets/Voxel51/OpenLogo) — public mirror used to obtain the FlickrLogos-32 positive images.

# One-Shot Instance Detection

Find a specific object instance in RGB query images, given a single reference
image or crop.

> **Status:** the benchmark harness and SIFT / ORB / AKAZE baselines are
> implemented. FlickrLogos-32 is the first dataset; current results are below.
> Design notes and the full bibliography are in
> [`docs/devnotes/week1-devnotes.md`](docs/devnotes/week1-devnotes.md).

## Overview

Most detectors answer *"is there a dog in this image?"* This project answers
*"where is **this exact** dog / logo / building / product?"* — one reference
crop, one query image, a box around the same instance.

This is **instance-level matching**, not category detection and not tracking.
The reference defines the instance; no class label is needed.

| Aspect | Decision |
|---|---|
| Input | a reference image, or a crop of one specific instance |
| Output | a box for that same instance in a query image (or all occurrences) |
| Task | instance-level matching |
| Modality | RGB images only |
| Training | training-free by default; fine-tuning pretrained models is allowed; training from scratch is out of scope |

Typical pipeline:

```
reference crop → keypoints / descriptors → match → robust estimate → box
```

A single RANSAC homography returns only the dominant instance. Finding every
occurrence needs a multi-model step: sequential RANSAC, J-Linkage / T-Linkage,
proposal-then-verify, or generalized Hough voting.

## Project arc

This is not only a benchmark. The sequence is:

1. **Benchmark current solutions** — a fixed harness and comparable classical /
   learned baselines.
2. **Diagnose the gaps** — per-class, logo-size, multi-instance, confidence, and
   latency analysis to find where methods actually break.
3. **Build the final pipeline** — a full model / end-to-end approach informed by
   those failures, not just the next model on a list.
4. **Write the final report** — methodology, results, failure analysis, and
   conclusions.

Each week gets one devlog, published on LinkedIn and kept in `docs/devnotes/`.

## Baseline suite

The benchmark keeps the pipeline fixed so models are actually comparable.
Each family has at least one simple baseline and one stronger entry.

Implemented now: **SIFT + BF-L2**, **ORB + BF-Hamming**, **AKAZE +
BF-Hamming**, all followed by a ratio test and RANSAC homography. The rest of
the suite is planned.

| Layer | Baseline | Advanced |
|---|---|---|
| Classical | SIFT + BF-L2 / FLANN-KDTree, ORB + BF-Hamming, AKAZE + BF-Hamming | NCC template matching |
| Learned sparse | SuperPoint + LightGlue, ALIKED + LightGlue | OmniGlue, XFeat |
| Dense | LoFTR, RoMa v1 | RoMa v2, FlexGlue |
| Foundation | DINOv2 ViT-B/14 patch cosine | DINOv3 ViT-B/16 |
| Retrieval (buildings) | DINOv2 CLS/GeM + FAISS | VLAD/GeM + local verification |
| People | DINOv2 embeddings + NN | person-search model |
| Multi-instance | sequential RANSAC | J-Linkage / proposal-verify |

Selected suite:

- **Local feature matching** — SIFT/BF-L2 (FLANN-KDTree optional),
  ORB/BF-Hamming, AKAZE/BF-Hamming, SuperPoint/LightGlue,
  ALIKED/LightGlue, LoFTR, RoMa, FlexGlue.
- **Foundation embeddings** — DINOv2 / cosine, DINOv3 / cosine.

Notes:

- ORB and AKAZE use BF-Hamming. FLANN needs LSH for binary descriptors;
  KD-tree is for float descriptors such as SIFT.
- OpenCV 5 moved AKAZE to `cv2.xfeatures2d`, so the project depends on
  `opencv-contrib-python` rather than `opencv-python`.
- SuperGlue's original implementation uses custom CUDA ops, so LightGlue is
  the safer default on ROCm.
- DINOv2 is best used for pruning / retrieval, not as the sole descriptor.
  OmniGlue uses it as bilinearly interpolated per-keypoint guidance `g_i`,
  while the final matching still uses refined SuperPoint `d_i`.
- Fine-tuning candidates: OmniGlue low-shot adaptation, LightGlue/SuperPoint
  on synthetic homographies, DINOv2 probes/adapters, and person-search models
  on CUHK-SYSU.

## Datasets

Five datasets cover four instance categories. No single benchmark covers all
of them, and each has a different task and annotation format.

| Category | Dataset | Task / annotations | Metric |
|---|---|---|---|
| Logos | [FlickrLogos-32](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos) | 32 classes, boxes, multiple logos per image | AP@0.5 |
| People | [CUHK-SYSU](https://github.com/ShuangLI59/person_search) | person search, boxes + IDs | mAP, top-1 |
| Buildings | [Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) | retrieval, query ROIs | mAP, recall@k |
| Objects | [RPC](https://arxiv.org/abs/1901.07249) | category-level product detection, boxes | mAP |
| Validation | [HPatches](https://github.com/hpatches/hpatches-dataset) | planar pairs, homographies | MMA, HEA, corner error |

Protocol notes:

- Common protocol: reference crop → query image → predicted box → AP@0.5.
- Paris6k: report mAP / recall@k, and project the query ROI through the
  estimated homography for a box IoU.
- HPatches: matcher validation only; report MMA@1/3/5/10, HEA@1/3/5/10, and
  corner error. Do not report detection AP for it.
- Do not mix category-level and instance-level AP in one table.
- Access: FlickrLogos-32 requires an email request (the prep script uses the
  public OpenLogo mirror); CUHK-SYSU requires a request form; Paris6k original
  links may be dead; RPC is directly downloadable; HPatches is directly
  downloadable.

## Evaluation

The current harness records accuracy, speed, and match diagnostics, and writes
JSON plus Markdown. Size and memory accounting are planned.

| Axis | Recorded now |
|---|---|
| Accuracy | box protocol: hit rate at IoU@0.5, 11-point AP@0.5, mean IoU; HPatches: MMA, HEA, corner error |
| Speed | per-pair latency mean / p50 / p95 |
| Diagnostics | match count, RANSAC inlier count; keypoint budget in config |
| Per-sample output | query, predicted box, GT boxes, hit, latency |
| Planned | params + disk size, peak GPU / CPU memory |

Reproducibility requires a fixed keypoint budget, input resolution, RANSAC
threshold, and Lowe ratio. OpenCV RANSAC is not seeded, so small run-to-run
variation is expected.

## Pipeline

`main.py` is the benchmark entry point; the implementation lives in
`src/pipeline.py`. The Makefile wraps the common commands — run `make help`
to list them.

```bash
make bench
make hpatches
```

Common targets:

| Target | Action |
|---|---|
| `make sync` | `uv lock && uv sync --all-groups` |
| `make check` | Ruff lint + format check |
| `make bench` | FlickrLogos-32 box benchmark |
| `make bench-smoke` | 20-sample smoke test |
| `make hpatches` | HPatches matcher validation |
| `make examples` | annotated prediction examples |
| `make analyze` | failure-analysis plots and report |
| `make clean` | remove local caches |
| `make tag-week WEEK=N` | create and push the weekly tag |

Override variables for a run, for example:

```bash
make bench MODELS="sift"
make hpatches MODELS="sift orb"
make bench MANIFEST=path/to/manifest.json OUTPUT=outputs/custom
```

Run `main.py` directly for flags not exposed by the Makefile.

| Option | Default | Meaning |
|---|---|---|
| `--protocol` | `box` | `box` (AP@0.5) or `hpatches` (MMA/HEA/corner error) |
| `--manifest` | required for `box` | JSON or JSONL reference/query manifest |
| `--root` | manifest directory | base directory for relative paths |
| `--hpatches-dir` | `data/hpatches` | directory containing `hpatches-sequences-release` |
| `--models` | `all` | `sift`, `orb`, `akaze`, `all` |
| `--output` | `outputs/benchmark` | directory for `results.json` and `results.md` |
| `--max-samples` | all | limit samples (box) or pairs (HPatches) |
| `--iou-threshold` | `0.5` | IoU threshold for a box hit |
| `--ransac-threshold` | `3.0` | RANSAC reprojection threshold in pixels |
| `--max-keypoints` | `2048` | keypoint budget per image |
| `--ratio` | `0.75` | Lowe ratio-test threshold |

### Manifest format

Each record is one reference/query pair:

```json
[
  {
    "dataset": "flickrlogos32",
    "instance_id": "adidas",
    "category": "logo",
    "reference": "images/adidas/4991278064.jpg",
    "reference_box": [403, 271, 609, 396],
    "query": "images/adidas/144503924.jpg",
    "query_boxes": [[43, 17, 235, 195]],
    "metadata": {}
  }
]
```

- Paths are relative to the manifest directory (or `--root`).
- Boxes are absolute `[x1, y1, x2, y2]` pixels.
- `reference_box` is optional; without it the full reference image is the crop.
- `query_boxes` may be empty for a negative query.
- JSONL is supported with one record per line.

### Outputs

| Path | Contents |
|---|---|
| `results.json` | config, per-model aggregate metrics, per-sample records |
| `results.md` | Markdown summary table |
| `examples/index.md` | annotated good, bad, and false-positive examples |
| `analysis/analysis.md` | failure analysis, plots, and per-class CSVs |

## Repository layout

| Path | Contents |
|---|---|
| `docs/devnotes/week1-devnotes.md` | goal, model options, datasets, metrics, model selection, current implementation, bibliography |
| `docs/rocm-setup.md` | full ROCm / WSL2 install and troubleshooting guide |
| `src/pipeline.py` | benchmark implementation and CLI |
| `src/models/feature_matching/` | SIFT, ORB, AKAZE matchers and shared matching helpers |
| `src/utils/dataset.py` | manifest-backed reference/query dataset |
| `src/utils/logger.py` | stderr logger helper |
| `src/analysis/` | failure-analysis data, metrics, plots, and report helpers |
| `src/validation/` | HPatches sequence loading, MMA/HEA/corner-error metrics, and report |
| `main.py` | benchmark CLI entry point; delegates to `src/pipeline.py` |
| `Makefile` | common setup, benchmark, validation, and analysis commands |
| `scripts/prepare_rocm_wheels.py` | retags + vendors AMD's legacy-tagged ROCm wheel closure for `uv lock` |
| `scripts/prepare_flickrlogos32.py` | downloads the FlickrLogos-32 subset from OpenLogo and writes the benchmark manifest |
| `scripts/prepare_hpatches.py` | downloads HPatches and writes the box-AP manifest |
| `scripts/visualize_examples.py` | renders good, bad, and false-positive prediction examples |
| `scripts/analyze_results.py` | writes failure plots, CSVs, and a Markdown analysis |
| `scripts/gpu_check.py` | backend/GPU diagnostics: HIP vs CUDA build, driver, groups, tools |
| `scripts/setup_wsl_rocm.sh` | WSL2 + AMD Radeon setup checks (kernel, HIP SDK, `/dev/kfd`, permissions) |
| `data/` | input images / datasets (gitignored) |
| `outputs/` | results, checkpoints, logs (gitignored) |
| `vendor/pytorch-rocm/` | vendored, retagged ROCm wheels (gitignored; ~1.9 GB) |
| `pyproject.toml` | torch pins per OS + `[tool.uv]` find-links configuration |

## Setup

### Quick start — Linux + AMD ROCm

```bash
make rocm-wheels                            # once; vendors the retagged AMD ROCm closure
make sync                                   # uv lock + uv sync --all-groups
make gpu                                    # expect device: cuda (ROCm/HIP build)
make data-flickr                            # download + build data/flickrlogos32/manifest.json
make bench                                  # run the benchmark
```

A ~1.9 GB `.venv` is created; make sure you have disk headroom. On WSL2, run
`bash scripts/setup_wsl_rocm.sh` first, then follow the quick start.

### Benchmark data — FlickrLogos-32

FlickrLogos-32 is the first dataset. The official download requires an email
request, so `prepare_flickrlogos32.py` instead pulls the 2,240 positive
FlickrLogos-32 images from the public `Voxel51/OpenLogo` mirror and rebuilds
the 32 logo classes.

```bash
make data-flickr
```

This writes `data/flickrlogos32/images/` and
`data/flickrlogos32/manifest.json`. The manifest has 2,208 positive pairs
(one reference crop per class, 69 queries per class) plus 320 cross-class
negative queries. Each class reference is its median-area logo crop.

Prep options: `--data-dir`, `--samples` (cached OpenLogo metadata),
`--negatives-per-class` (default 10), `--seed`, and `--skip-download`.

Run the benchmark with:

```bash
make bench
make bench MODELS="sift"
```

Save annotated prediction examples, including failures and false positives:

```bash
make examples
```

The images and a Markdown index are written to
`outputs/flickrlogos32/examples/`. Visualization options: `--models`,
`--good`, `--bad`, `--negatives`.

### Matcher validation — HPatches

HPatches is used to check whether the matchers themselves behave as expected.
The benchmark runs through the same pipeline and model registry, but switches
to the metrics used in the papers: MMA@1/3/5/10, HEA@1/3/5/10, and corner error.

```bash
make data-hpatches
make hpatches
make hpatches MODELS="sift"
```

This downloads the 116 sequences (580 reference/target pairs) into
`data/hpatches/` and writes `outputs/hpatches/results.json` plus `results.md`.
The report includes a literature reference table and a per-split breakdown.

### CUDA / CPU / macOS

The non-Linux dependency markers install stock PyPI torch. To switch a Linux
environment to CUDA or CPU, remove the `==...+rocm*` pins and the
`[tool.uv] find-links` entry, then re-lock. `torch.cuda.is_available()` then
means an NVIDIA GPU.

### Full ROCm / WSL2 guide

Driver installation, WSL2 caveats, backend switching, and sanity checks are
in [`docs/rocm-setup.md`](docs/rocm-setup.md).

## Current results — FlickrLogos-32

2,528 pairs: 2,208 positives and 320 cross-class negatives, one reference crop
per class.

| Model | Hit rate | AP@0.5 | Mean IoU | Mean matches | Mean inliers | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT | 0.413 | 0.406 | 0.352 | 21.3 | 12.1 | 64.5 ms | 82.7 ms |
| ORB | 0.215 | 0.237 | 0.196 | 13.0 | 7.7 | 9.1 ms | 17.5 ms |
| AKAZE | 0.239 | 0.251 | 0.209 | 7.3 | 5.1 | 48.9 ms | 63.3 ms |

Outputs: `outputs/flickrlogos32/results.json`, `results.md`, and
`examples/index.md`.

These numbers are a **project-specific one-shot benchmark**, not official
FlickrLogos-32 mAP: one reference per class, no P1/P2/P3 split, and
cross-class negatives instead of the original 3,000 no-logo images.

## HPatches validation results

116 sequences, 580 pairs, 2,048 keypoints, Lowe ratio 0.75.

| Model | MMA@1 | MMA@3 | MMA@5 | MMA@10 | HEA@1 | HEA@3 | HEA@5 | HEA@10 | Median CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIFT | 0.505 | 0.734 | 0.777 | 0.798 | 0.441 | 0.705 | 0.809 | 0.891 | 1.344 px |
| ORB | 0.385 | 0.720 | 0.789 | 0.814 | 0.245 | 0.529 | 0.667 | 0.772 | 2.515 px |
| AKAZE | 0.435 | 0.752 | 0.824 | 0.858 | 0.309 | 0.586 | 0.700 | 0.828 | 2.025 px |

For reference, Efe et al. 2021 report SIFT MMA@3 / HEA@3 of 0.87 / 0.76,
ORB 0.68 / 0.40, and AKAZE 0.80 / 0.65. Their protocol uses tuned ratio tests,
mutual nearest neighbors, 108 sequences, and OpenCV 4.5.2, so exact agreement
is not expected. The values are in the expected range, and the full report
includes the literature table plus per-split metrics.

Adding another matcher only requires registering it in `MODELS` and running the
same command — no separate validation script.

## Failure analysis

```bash
make analyze
```

This writes plots, CSVs, and `analysis.md` to `outputs/flickrlogos32/analysis/`.
The analysis splits failures into hit, wrong box, no prediction, false positive,
and true negative, and breaks results down by class, logo size, and number of
GT instances.

Current failure patterns:

| Model | Hit | Wrong box | No prediction | False positive |
|---|---:|---:|---:|---:|
| SIFT | 36.1% | 28.0% | 23.2% | 3.7% |
| ORB | 18.8% | 26.2% | 42.3% | 3.8% |
| AKAZE | 20.8% | 15.7% | 50.8% | 0.4% |

- **SIFT** usually finds the class but places the box on the wrong instance or
  background when it fails.
- **ORB and AKAZE** fail mostly by producing no prediction at all.
- Small logos are the main weakness: for logos below 1% of the query image,
  SIFT hits 0.218, while ORB and AKAZE hit 0.000.
- Queries with multiple GT instances are harder than single-instance queries
  for all three models.
- Worst SIFT classes include `apple`, `carlsberg`, `nvidia`, `pepsi`, and
  `shell`; best classes include `paulaner`, `rittersport`, and `starbucks`.

Generated plots include per-class hit-rate / IoU heatmaps, failure breakdown,
IoU distribution, hit rate by logo size and instance count, inliers vs IoU,
precision-recall, and latency.

## Roadmap

- [x] Define goal, scope, datasets, and model suite
- [x] Benchmarking harness (dataset → pipeline → metrics → JSON + Markdown)
- [x] FlickrLogos-32 preparation and one-shot benchmark
- [x] Classical baselines: SIFT, ORB, AKAZE
- [x] HPatches matcher validation (MMA / HEA / corner error)
- [x] Prediction example visualizations (good, bad, false positives)
- [x] Failure analysis: class, size, instance-count, confidence, and latency plots
- [ ] Classical baseline: NCC
- [ ] Learned sparse baselines: SuperPoint/LightGlue, ALIKED/LightGlue
- [ ] Dense baselines: LoFTR, RoMa
- [ ] Foundation embeddings: DINOv2, DINOv3
- [ ] Multi-instance handling
- [ ] Fine-tuning experiments
- [ ] Final pipeline + full model
- [ ] Final report
- [ ] Weekly devlogs (ongoing)

## Devlogs

One devlog per week, published on LinkedIn and kept here.

Weekly states are tagged in git. To see the project as it was at the end of a
previous week:

```bash
git tag --list 'week-*'
git checkout week-1
```

Return to the latest state with `git checkout main`.

- [Devlog index](docs/devnotes/README.md)
- [Week 1](docs/devnotes/week1-devnotes.md) — goal, design, datasets, metrics,
  first baseline results, and failure analysis.
- [Week 2](docs/devnotes/week2-devnotes.md) — HPatches validation integrated
  into the bench, MMA/HEA/corner error, literature comparison.

## Documentation

- [`docs/devnotes/week1-devnotes.md`](docs/devnotes/week1-devnotes.md) — design notes and bibliography
- [`docs/rocm-setup.md`](docs/rocm-setup.md) — ROCm / WSL2 environment guide

## License

MIT — see [`LICENSE`](LICENSE).

# AGENTS.md

## Project

One-shot instance detection: given a reference image or crop, find that exact
instance in RGB query images. Not category detection, not tracking.

Project arc:

1. Benchmark current solutions.
2. Diagnose failure modes and gaps.
3. Build the final pipeline / full model.
4. Write the final report.

One devlog per week, kept in `docs/devnotes/` and summarized on LinkedIn.

## Hard rules

- Do not write implementation code unless the user explicitly asks.
- Use only these datasets: FlickrLogos-32, CUHK-SYSU, Paris6k, RPC, HPatches.
- HPatches is a matcher-validation dataset: report MMA, HEA, and corner error,
  not detection AP.
- RGB images only.
- Training-free by default. Fine-tuning pretrained models is allowed. Training
  from scratch is out of scope.
- Keep the evaluation protocol fixed across models.
- Do not commit `data/`, `outputs/`, `local_files/`, or `vendor/pytorch-rocm/`.
- Do not add empty-directory `.gitkeep` placeholders.
- Keep files focused and readable. When touching a large file, prefer
  extracting a focused module instead of growing it further.

## Layout

| Path | Contents |
|---|---|
| `main.py` | benchmark CLI entry point; delegates to `src/pipeline.py` |
| `Makefile` | common setup, benchmark, validation, and analysis commands |
| `src/pipeline.py` | box-AP and HPatches protocols, model registry, reports |
| `src/models/feature_matching/` | classical, LightGlue, LoFTR, RoMa, FlexGlue matchers |
| `src/models/feature_matching/lightglue/_vendor/` | vendored LightGlue components (Apache 2.0) |
| `src/models/feature_matching/roma/_vendor/` | vendored RoMa inference code (MIT) |
| `src/models/feature_matching/flexglue/_vendor/` | vendored FlexGlue inference code (MIT) |
| `src/analysis/` | failure-analysis data, metrics, plots, tables, report |
| `src/validation/` | HPatches sequence loading, MMA/HEA/corner-error metrics |
| `src/utils/dataset.py` | manifest-backed reference/query dataset |
| `src/utils/logger.py` | stderr logger |
| `scripts/prepare_flickrlogos32.py` | FlickrLogos-32 download + manifest |
| `scripts/prepare_hpatches.py` | HPatches download + box-AP manifest |
| `scripts/prepare_cuhk_sysu.py` | CUHK-SYSU person-search manifest |
| `scripts/prepare_rpc.py` | RPC category-level manifest |
| `scripts/prepare_paris6k.py` | Paris6k retrieval manifest |
| `scripts/visualize_examples.py` | good/bad/false-positive prediction images |
| `scripts/analyze_results.py` | failure-analysis plots, CSVs, and report |
| `scripts/prepare_rocm_wheels.py` | vendor/retag AMD ROCm wheels for `uv lock` |
| `scripts/gpu_check.py` | ROCm / CUDA / device diagnostics |
| `docs/devnotes/` | weekly devlogs and devlog index |
| `docs/rocm-setup.md` | ROCm / WSL2 setup guide |
| `data/` | datasets (gitignored) |
| `outputs/` | benchmark results (gitignored) |
| `local_files/` | private drafts and notes (gitignored) |

## Setup

```bash
make rocm-wheels   # once on a fresh clone
make sync
make gpu
```

- Python 3.12 is required by the ROCm wheel pins.
- OpenCV 5 moved AKAZE to `cv2.xfeatures2d`; the project depends on
  `opencv-contrib-python`, not `opencv-python`.
- SuperPoint/LightGlue and ALIKED/LightGlue use `kornia` and download weights
  to `~/.cache/torch/hub/checkpoints` on first use.
- LightGlue runs with `flash=False` by default for ROCm compatibility.
- `--ratio` applies to classical matchers; LightGlue uses `filter_threshold=0.1`.
- RoMa and FlexGlue use vendored inference code and download official weights on
  first use. LoFTR uses Kornia's outdoor weights.
- On WSL2, run `bash scripts/setup_wsl_rocm.sh` first.

## Benchmarks

### Box protocol — FlickrLogos-32

```bash
make data-flickr
make bench
make bench MODELS="sift"
```

- 32 logo classes, 2,208 positive pairs + 320 cross-class negatives.
- One median-area reference crop per class; 69 positive queries per class.
- The official P1/P2/P3 split and the 3,000 no-logo negatives are not preserved.
- Metrics: hit rate at IoU@0.5, 11-point AP@0.5, mean IoU, match/inlier counts,
  latency mean/p50/p95.
- Outputs: `results.json`, `results.md`.

### HPatches validation

```bash
make data-hpatches
make hpatches
make hpatches MODELS="sift"
```

- 116 sequences, 580 reference/target pairs.
- Metrics: MMA@1/3/5/10, HEA@1/3/5/10, corner error.
- The report includes a literature reference table and per-split results.
- No separate validation script: use the same pipeline and `MODELS` registry.

### Other datasets

- CUHK-SYSU is request-only. `make data-cuhk` expects the official layout
  under `data/cuhk-sysu` (`Image/SSM`,
  `annotation/test/train_test/TestG50.mat`) and builds probe/gallery pairs.
- RPC is downloaded from the public Hugging Face mirror by `make data-rpc`.
  It is category-level, so report it separately from instance-level AP.
- Paris6k is retrieval-only: `make data-paris` downloads the revisitop images
  and ground truth, and `make paris` runs
  `--protocol retrieval` (mAP / recall@k). Do not report box AP for it.

### Examples and failure analysis

```bash
make examples
make analyze
```

- Examples: `outputs/flickrlogos32/examples/index.md`.
- Analysis: `outputs/flickrlogos32/analysis/analysis.md` plus plots and CSVs.

## Adding a model

1. Implement `FeatureMatcher.match(reference, query) -> MatchResult` in
   `src/models/feature_matching/` (classical or LightGlue subpackage).
2. Export it from the relevant `__init__.py`.
3. Register it in `MODELS` in `src/pipeline.py`.
4. Run both protocols:

```bash
make bench MODELS="loftr"
make hpatches MODELS="roma"
```

5. Update README/devnotes results if the change is user-visible.

Use the shared helpers in `base.py`: `ratio_test`, `estimate_homography`,
`keypoint_points`, `limit_keypoints`, and `to_gray`.

## Style

- Ruff, line length 88, Python 3.12.
- Lint rules: `E4`, `E7`, `E9`, `F`, `I`, `ANN`, `D`.
- Google-style docstrings and type hints on all function signatures.
- Keep protocol, dataset, and model code separate.
- Prefer small modules over one large file.

```bash
make check
make format
```

## Verification

- Run `make check` after code changes.
- Smoke-test with a subset before a full run: `make bench-smoke`.
- Confirm output JSON/Markdown exists and is well-formed.
- RANSAC is not seeded, so small run-to-run variation is expected.

## Git

- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
  `chore:`.
- One logical change per commit. Imperative subject; body explains what and why.
- Tag the end of each week as `week-N` with `make tag-week WEEK=N`; push the tag.
- Remote fetch uses HTTPS; push uses SSH (`git@github.com:OrangeKat/...`).
- Do not commit datasets, results, or private notes.
- Update the README and the current week devlog when behavior or results change.

## Documentation

- `README.md` — user-facing overview, setup, commands, current results.
- `docs/devnotes/README.md` — devlog index.
- `docs/devnotes/weekN-devnotes.md` — weekly progress, decisions, failures,
  metrics, and next steps.
- `docs/rocm-setup.md` — environment-specific setup and troubleshooting.

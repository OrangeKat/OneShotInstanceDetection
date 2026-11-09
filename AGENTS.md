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
| `src/pipeline.py` | box-AP and HPatches protocols, model registry, reports |
| `src/models/feature_matching/` | SIFT, ORB, AKAZE and shared matching helpers |
| `src/analysis/` | failure-analysis data, metrics, plots, tables, report |
| `src/validation/` | HPatches sequence loading, MMA/HEA/corner-error metrics |
| `src/utils/dataset.py` | manifest-backed reference/query dataset |
| `src/utils/logger.py` | stderr logger |
| `scripts/prepare_flickrlogos32.py` | FlickrLogos-32 download + manifest |
| `scripts/prepare_hpatches.py` | HPatches download + box-AP manifest |
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
python3 scripts/prepare_rocm_wheels.py   # once on a fresh clone
uv lock && uv sync --all-groups
uv run python scripts/gpu_check.py
```

- Python 3.12 is required by the ROCm wheel pins.
- OpenCV 5 moved AKAZE to `cv2.xfeatures2d`; the project depends on
  `opencv-contrib-python`, not `opencv-python`.
- On WSL2, run `bash scripts/setup_wsl_rocm.sh` first.

## Benchmarks

### Box protocol — FlickrLogos-32

```bash
uv run python scripts/prepare_flickrlogos32.py
uv run python main.py \
  --manifest data/flickrlogos32/manifest.json \
  --models sift orb akaze \
  --output outputs/flickrlogos32
```

- 32 logo classes, 2,208 positive pairs + 320 cross-class negatives.
- One median-area reference crop per class; 69 positive queries per class.
- The official P1/P2/P3 split and the 3,000 no-logo negatives are not preserved.
- Metrics: hit rate at IoU@0.5, 11-point AP@0.5, mean IoU, match/inlier counts,
  latency mean/p50/p95.
- Outputs: `results.json`, `results.md`.

### HPatches validation

```bash
uv run python scripts/prepare_hpatches.py
uv run python main.py \
  --protocol hpatches \
  --models sift orb akaze \
  --output outputs/hpatches
```

- 116 sequences, 580 reference/target pairs.
- Metrics: MMA@1/3/5/10, HEA@1/3/5/10, corner error.
- The report includes a literature reference table and per-split results.
- No separate validation script: use the same pipeline and `MODELS` registry.

### Examples and failure analysis

```bash
uv run python scripts/visualize_examples.py
uv run python scripts/analyze_results.py
```

- Examples: `outputs/flickrlogos32/examples/index.md`.
- Analysis: `outputs/flickrlogos32/analysis/analysis.md` plus plots and CSVs.

## Adding a model

1. Implement `FeatureMatcher.match(reference, query) -> MatchResult` in
   `src/models/feature_matching/`.
2. Export it from `src/models/feature_matching/__init__.py`.
3. Register it in `MODELS` in `src/pipeline.py`.
4. Run both protocols:

```bash
uv run python main.py --manifest data/flickrlogos32/manifest.json --models <name>
uv run python main.py --protocol hpatches --models <name>
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
uv run ruff check .
uv run ruff format .
```

Known pre-existing Ruff findings in `scripts/prepare_rocm_wheels.py`: `I001`
(import order) and `F841` (unused `linux`). Do not fix unless asked.

## Verification

- Run `ruff check` and `ruff format --check` after code changes.
- Smoke-test with a subset before a full run:
  `uv run python main.py --manifest data/flickrlogos32/manifest.json --max-samples 20`.
- Confirm output JSON/Markdown exists and is well-formed.
- RANSAC is not seeded, so small run-to-run variation is expected.

## Git

- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
  `chore:`.
- One logical change per commit. Imperative subject; body explains what and why.
- Tag the end of each week as `week-N` and push the tag.
- Remote fetch uses HTTPS; push uses SSH (`git@github.com:OrangeKat/...`).
- Do not commit datasets, results, or private notes.
- Update the README and the current week devlog when behavior or results change.

## Documentation

- `README.md` — user-facing overview, setup, commands, current results.
- `docs/devnotes/README.md` — devlog index.
- `docs/devnotes/weekN-devnotes.md` — weekly progress, decisions, failures,
  metrics, and next steps.
- `docs/rocm-setup.md` — environment-specific setup and troubleshooting.

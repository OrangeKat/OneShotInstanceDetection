# AGENTS.md

## Project

One-shot instance detection: given a reference image or crop, find that exact
instance in RGB query images. Not category detection, not tracking.

## Rules

- Do not write implementation code unless the user explicitly asks.
- Use only these datasets: FlickrLogos-32, CUHK-SYSU, Paris6k, RPC, HPatches.
- HPatches is used for matcher validation with MMA / HEA / corner error, not
  detection AP.
- RGB images only.
- Training-free by default. Fine-tuning pretrained models is allowed.
  Training from scratch is out of scope.
- Keep the evaluation protocol fixed across models.

## Layout

- `src/pipeline.py` — benchmark implementation and CLI.
- `main.py` — benchmark CLI entry point (delegates to `src/pipeline.py`).
- `src/models/feature_matching/` — SIFT, ORB, AKAZE matchers.
- `src/analysis/` — failure-analysis data, metrics, plots, and report helpers.
- `src/utils/dataset.py` — manifest-backed reference/query dataset.
- `src/utils/logger.py` — stderr logger.
- `docs/devnotes/week1-devnotes.md` — design, models, datasets, metrics, bibliography.
- `docs/rocm-setup.md` — ROCm / WSL2 setup.
- `data/` — datasets (gitignored).
- `outputs/` — results (gitignored).

## Commands

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format .
uv run python scripts/gpu_check.py
uv run python scripts/prepare_flickrlogos32.py
uv run python main.py --manifest data/flickrlogos32/manifest.json
uv run python scripts/prepare_hpatches.py
uv run python main.py --protocol hpatches --output outputs/hpatches
uv run python scripts/visualize_examples.py
uv run python scripts/analyze_results.py
```

## Style

- Ruff with Google-style docstrings.
- Type hints on all function signatures.

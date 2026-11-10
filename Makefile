SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= uv run python
RUFF ?= uv run ruff
MODELS ?= sift orb akaze superpoint_lightglue aliked_lightglue loftr roma flexglue
MAX_SIDE ?= 640
GPU_MEM_FRACTION ?= 0
MANIFEST ?= data/flickrlogos32/manifest.json
OUTPUT ?= outputs/flickrlogos32
HPATCHES_OUTPUT ?= outputs/hpatches
PARIS_OUTPUT ?= outputs/paris6k
MODEL ?= sift
INDEX ?= 0
ARGS ?=
ANALYZE_MANIFEST ?= data/flickrlogos32/manifest.json
ANALYZE_RESULTS ?= outputs/flickrlogos32/results.json
ANALYZE_OUTPUT ?= outputs/flickrlogos32/analysis
WEEK ?=

.PHONY: help rocm-wheels sync lint format check gpu \
	data-flickr data-hpatches data-prw data-rpc data-paris \
	bench bench-smoke hpatches paris bench-all \
	examples analyze diagnose compare sweep figures all \
	clean tag-week

help: ## Show available targets
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make bench MODELS=\"sift\""
	@echo "  make bench-smoke"
	@echo "  make tag-week WEEK=2"

rocm-wheels: ## Vendor and retag AMD ROCm wheels (once on a fresh clone)
	python3 scripts/prepare_rocm_wheels.py

sync: ## Lock and sync all dependency groups
	uv lock
	uv sync --all-groups

lint: ## Run Ruff checks
	$(RUFF) check .

format: ## Format code with Ruff
	$(RUFF) format .

check: ## Check linting and formatting without modifying files
	$(RUFF) check .
	$(RUFF) format --check .

gpu: ## Run the ROCm / GPU diagnostics
	$(PYTHON) scripts/gpu_check.py

data-flickr: ## Download FlickrLogos-32 and build the manifest
	$(PYTHON) scripts/prepare_flickrlogos32.py

data-hpatches: ## Download HPatches and build the manifest
	$(PYTHON) scripts/prepare_hpatches.py

data-prw: ## Build the PRW person-search manifest (data must be downloaded)
	$(PYTHON) scripts/prepare_prw.py

data-rpc: ## Download RPC shards and build the category-level manifest
	$(PYTHON) scripts/prepare_rpc.py

data-paris: ## Download Paris6k and build the retrieval manifest
	$(PYTHON) scripts/prepare_paris6k.py

bench: ## Run the FlickrLogos-32 box benchmark
	$(PYTHON) main.py \
		--manifest $(MANIFEST) \
		--models $(MODELS) \
		--max-side $(MAX_SIDE) \
		--gpu-mem-fraction $(GPU_MEM_FRACTION) \
		--output $(OUTPUT)

bench-smoke: ## Run a 20-sample box smoke test
	$(PYTHON) main.py \
		--manifest $(MANIFEST) \
		--models $(MODELS) \
		--max-samples 20 \
		--max-side $(MAX_SIDE) \
		--gpu-mem-fraction $(GPU_MEM_FRACTION) \
		--output $(OUTPUT)-smoke

hpatches: ## Run HPatches matcher validation
	$(PYTHON) main.py \
		--protocol hpatches \
		--models $(MODELS) \
		--max-side $(MAX_SIDE) \
		--gpu-mem-fraction $(GPU_MEM_FRACTION) \
		--output $(HPATCHES_OUTPUT)

paris: ## Run Paris6k retrieval evaluation
	$(PYTHON) main.py \
		--protocol retrieval \
		--models $(MODELS) \
		--max-side $(MAX_SIDE) \
		--gpu-mem-fraction $(GPU_MEM_FRACTION) \
		--output $(PARIS_OUTPUT)

bench-all: ## Run every available dataset sequentially
	$(PYTHON) scripts/run_all.py --models $(MODELS) $(ARGS)

examples: ## Render good, bad, and false-positive examples
	$(PYTHON) scripts/visualize_examples.py

analyze: ## Generate failure-analysis plots and report
	$(PYTHON) scripts/analyze_results.py \
		--manifest $(ANALYZE_MANIFEST) \
		--results $(ANALYZE_RESULTS) \
		--output $(ANALYZE_OUTPUT) $(ARGS)

diagnose: ## Diagnose one pair (MODEL=sift INDEX=0 ARGS="--seeds 5 --oracle")
	$(PYTHON) scripts/diagnose.py --model $(MODEL) --index $(INDEX) $(ARGS)

compare: ## Compare models across dataset result files
	$(PYTHON) scripts/compare_datasets.py $(ARGS)

sweep: ## Run a sensitivity sweep (ARGS="--keypoints 512 1024")
	$(PYTHON) scripts/sweep.py $(ARGS)

figures: ## Regenerate the per-dataset devlog figures
	$(PYTHON) scripts/plot_dataset_summary.py

all: check bench hpatches examples analyze ## Run checks and both benchmarks

clean: ## Remove local Python and Ruff caches
	find src scripts -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache .pytest_cache

tag-week: ## Create and push a weekly tag (usage: make tag-week WEEK=2)
	@test -n "$(WEEK)" || (echo "usage: make tag-week WEEK=2"; exit 1)
	git tag -a week-$(WEEK) -m "Week $(WEEK)"
	git push origin week-$(WEEK)

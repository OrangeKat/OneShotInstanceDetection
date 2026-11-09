SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= uv run python
RUFF ?= uv run ruff
MODELS ?= sift orb akaze superpoint_lightglue aliked_lightglue loftr roma flexglue
MANIFEST ?= data/flickrlogos32/manifest.json
OUTPUT ?= outputs/flickrlogos32
HPATCHES_OUTPUT ?= outputs/hpatches
WEEK ?=

.PHONY: help rocm-wheels sync lint format check gpu \
	data-flickr data-hpatches \
	bench bench-smoke hpatches \
	examples analyze all \
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

bench: ## Run the FlickrLogos-32 box benchmark
	$(PYTHON) main.py \
		--manifest $(MANIFEST) \
		--models $(MODELS) \
		--output $(OUTPUT)

bench-smoke: ## Run a 20-sample box smoke test
	$(PYTHON) main.py \
		--manifest $(MANIFEST) \
		--models $(MODELS) \
		--max-samples 20 \
		--output $(OUTPUT)-smoke

hpatches: ## Run HPatches matcher validation
	$(PYTHON) main.py \
		--protocol hpatches \
		--models $(MODELS) \
		--output $(HPATCHES_OUTPUT)

examples: ## Render good, bad, and false-positive examples
	$(PYTHON) scripts/visualize_examples.py

analyze: ## Generate failure-analysis plots and report
	$(PYTHON) scripts/analyze_results.py

all: check bench hpatches examples analyze ## Run checks and both benchmarks

clean: ## Remove local Python and Ruff caches
	find src scripts -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache .pytest_cache

tag-week: ## Create and push a weekly tag (usage: make tag-week WEEK=2)
	@test -n "$(WEEK)" || (echo "usage: make tag-week WEEK=2"; exit 1)
	git tag -a week-$(WEEK) -m "Week $(WEEK)"
	git push origin week-$(WEEK)

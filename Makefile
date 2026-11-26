# Ekam — one command per pipeline stage. Run from the repo root.
# Windows: `winget install ezwinports.make` (works from PowerShell or Git Bash).
# Pass Hydra overrides with ARGS, e.g.:  make data ARGS="synth.n_entities=20000"

UV      ?= uv
RUN     := $(UV) run python -m
ARGS    ?=

# Keep every cache inside the repo (off a small C: drive) and force UTF-8 on Windows consoles.
export UV_CACHE_DIR            := .cache/uv
export HF_HOME                 := .cache/huggingface
export TORCH_HOME              := .cache/torch
export PYTHONUTF8              := 1
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help setup data data-quick block train m2 resolve eval demo test test-fast lint mlflow clean-synth

help:
	@echo "setup      install pinned deps (uv sync)"
	@echo "data       M1: download MCA + benchmarks, ingest, synth sources, entity split, audit"
	@echo "data-quick same on a 20k-entity sample (fast smoke run)"
	@echo "test       full pytest suite (includes checks on generated data if present)"
	@echo "lint       ruff"
	@echo "mlflow     local MLflow UI on http://127.0.0.1:5000"
	@echo "block      M2: normalise, rule + FAISS-kNN blocking, PC/RR curves (docs/reports/M2_blocking.*)"
	@echo "train      M2: LR / LightGBM / Splink baselines + benchmarks (docs/reports/M2_baselines.*)"
	@echo "m2         block then train (one command for the M2 deliverables)"
	@echo "resolve | eval | demo   (milestones 3-5)"

setup:
	$(UV) sync

data:
	$(RUN) ekam.pipeline.data $(ARGS)

data-quick:
	$(RUN) ekam.pipeline.data synth.n_entities=20000 $(ARGS)

block:
	$(RUN) ekam.pipeline.block $(ARGS)

train:
	$(RUN) ekam.pipeline.train $(ARGS)

m2: block train

resolve eval demo:
	@echo "'make $@' arrives in a later milestone (see docs/DECISIONS.md and README)."; exit 1

test:
	$(UV) run pytest $(ARGS)

test-fast:
	$(UV) run pytest -m "not slow" $(ARGS)

lint:
	$(UV) run ruff check src tests

mlflow:
	$(UV) run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

clean-synth:
	$(UV) run python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['data/synth','data/processed']]"

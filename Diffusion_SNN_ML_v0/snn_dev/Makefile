# Makefile for SNN Diffusion Project
# Ensures all commands run using the 'diff' conda environment

# Path to the specific python interpreter in the 'diff' environment
PYTHON := /home/sebas_dev_linux/miniconda3/envs/diff/bin/python
PIP := /home/sebas_dev_linux/miniconda3/envs/diff/bin/pip

.PHONY: check-env run-pipeline help

help:
	@echo "Available commands:"
	@echo "  make check-env      : Verify python version and environment"
	@echo "  make run-pipeline   : Run the extraction and segmentation pipeline"
	@echo "                        (usage: make run-pipeline LIMIT=10)"

check-env:
	@echo "Checking Python environment..."
	@which $(PYTHON)
	@$(PYTHON) --version
	@echo "Environment 'diff' is ready."

# Example: make run-pipeline LIMIT=20
LIMIT ?= 0
run-pipeline:
	@echo "Running pipeline with LIMIT=$(LIMIT)..."
	$(PYTHON) src/pipeline.py --limit $(LIMIT)

analyze-data:
	@echo "Running dataset analysis..."
	$(PYTHON) src/analyze_data.py

run-baseline:
	@echo "Running baseline SNN training..."
	$(PYTHON) src/baseline.py

train-diffusion:
	@echo "Starting Class-Conditional Diffusion Training..."
	$(PYTHON) src/train_diffusion.py

# Usage: make generate-synthetic CHECKPOINT=diffusion_checkpoints/checkpoint-epoch-100
generate-synthetic:
	@echo "Generating and filtering synthetic data..."
	$(PYTHON) src/generate_synthetic.py $(CHECKPOINT)

train-mixed:
	@echo "Training SNN on Mixed (Real + Synthetic) dataset..."
	$(PYTHON) src/train_mixed.py


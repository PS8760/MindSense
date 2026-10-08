# MindSense — build automation.
# Every target is verified to run on a clean clone (see README quickstart).

PY ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYV := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest

.DEFAULT_GOAL := help

.PHONY: help setup data train notebooks lint test app docker clean smoke

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv + install runtime & training dependencies
	@if $(PYV) -m pip --version >/dev/null 2>&1; then \
		$(PYV) -m pip install -r requirements-train.txt; \
	elif command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(PYV) -r requirements-train.txt; \
	else \
		echo "venv has no pip and uv not found; recreate with: uv venv --python 3.11 .venv"; exit 1; \
	fi
	$(PYV) -m ipykernel install --user --name mindsense --display-name "Python (mindsense)" || true
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

data: ## Run the data pipeline (Exp 1): download with fallback ladder + clean + harmonize
	$(PYV) scripts/download_data.py
	$(PYV) scripts/run_pipeline.py

train: ## Train all models (Exp 3/4/6/7) and export artifacts + metrics
	$(PYV) -m mindsense.models.train_all

notebooks: ## Execute all 10 notebooks top-to-bottom
	bash scripts/run_all_notebooks.sh

lint: ## Ruff lint + format check
	$(RUFF) check src app tests scripts
	$(RUFF) format --check src app tests scripts || $(RUFF) format src app tests scripts

test: ## Run the pytest suite
	$(PYTEST) tests -q

app: ## Launch the Streamlit app locally
	$(VENV)/bin/streamlit run app/Home.py --server.port 8501

docker: ## Build and run the Docker image
	docker build -t mindsense .
	docker run --rm -p 8501:8501 mindsense

smoke: ## Fast CI-style check: lint + tests (no heavy training)
	$(RUFF) check src app tests scripts
	$(PYTEST) tests -q

clean: ## Remove caches and executed notebook outputs
	rm -rf .pytest_cache .ruff_cache **/__pycache__
	find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} +

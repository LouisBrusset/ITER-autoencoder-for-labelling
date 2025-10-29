SHELL := /bin/bash

.PHONY: run test test-quick clean clean-minimal test-clean

# run: clean data/models/tests then start the backend server
run:
	@echo "Cleaning data and models before starting..."
	@$(MAKE) clean
	@echo "Starting backend (uvicorn) on port 8000"
	@uvicorn src.autoencoder_for_labelling.main:app --reload --host 0.0.0.0 --port 8000

# test: run tests with coverage and generate report
test:
	@echo "Cleaning data and models before starting..."
	@$(MAKE) clean
	@echo "Running tests with coverage..."
	@uv run pytest --cov=autoencoder_for_labelling --cov-report=term-missing --cov-report=html

# test-quick: run tests without coverage for faster feedback
test-quick:
	@echo "Cleaning data and models before starting..."
	@$(MAKE) clean
	@echo "Running tests quickly..."
	@uv run pytest -v

# clean: remove generated data, model artifacts, and coverage files
clean:
	@echo "Removing dataset and uploaded/synthetic files in data and models"
	@rm -f data/current_dataset.npz || true
	@rm -rf data/synthetic/* || true
	@rm -rf data/uploaded/* || true
	@rm -f models/current_model.pth || true
	@rm -f models/current_architecture.json || true
	@rm -rf models/saved_model/* || true
	@rm -rf models/saved_architechture/* || true

	@echo "Removing results files..."
	@rm -rf results/labels/* || true
	@rm -rf results/latents/* || true
	@rm -rf results/projections2d/* || true
	@rm -rf results/reconstructions/* || true
	@rm -f results/current_*.npz || true

	@echo "Removing coverage files from tests..."
	@rm -rf .coverage || true
	@rm -rf htmlcov/ || true
	@rm -rf .pytest_cache/ || true
	@rm -rf __pycache__/ || true
	@rm -rf */__pycache__/ || true
	@rm -rf */*/__pycache__/ || true
	@rm -rf */*/*/__pycache__/ || true

	@echo "Clean complete."

# clean-minimal: remove only some files
clean-minimal:
	@echo "Removing dataset and uploaded/synthetic files in data and models"
	@rm -f data/current_dataset.npz || true
	@rm -f models/current_model.pth || true
	@rm -f models/current_architecture.json || true
	@rm -f results/current_*.npz || true
	@echo "Clean minimal complete."



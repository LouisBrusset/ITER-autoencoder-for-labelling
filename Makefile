SHELL := /bin/bash

.PHONY: run test clean

# run: clean data/models then start the backend server (foreground)
run:
	@echo "Cleaning data and models before starting..."
	@$(MAKE) clean
	@echo "Starting backend (uvicorn) on port 8000"
	@cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# test: start a temporary backend in background, run smoke curl checks, then stop it
test:
	@echo "To implement..."
	
# clean: remove generated data and model artifacts so you can start fresh
clean:
	@echo "Removing dataset and uploaded/synthetic files in backend/data and backend/models"
	@rm -f backend/data/current_dataset.npz || true
	@rm -rf backend/data/synthetic/* || true
	@rm -rf backend/data/uploaded/* || true
	@rm -f backend/models/current_model.pth || true
	@rm -rf backend/models/saved/* || true
	@echo "Clean complete."

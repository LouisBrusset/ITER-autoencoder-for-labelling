from pathlib import Path
import os
import os
import asyncio
import numpy as np

import pytest
from fastapi.testclient import TestClient
from autoencoder_for_labelling.main import app

client = TestClient(app)


def _write_dataset(filename: str, n_samples: int = 20, input_dim: int = 6, val_ratio: float = 0.2):
	os.makedirs("data/synthetic", exist_ok=True)
	data = np.random.rand(n_samples, input_dim)
	labels = np.zeros(n_samples, dtype=int)
	n_val = max(1, int(n_samples * val_ratio))
	is_train = np.ones(n_samples, dtype=bool)
	is_train[-n_val:] = False
	np.savez(Path("data/synthetic") / filename, data=data, labels=labels, is_train=is_train)


def _cleanup(*paths):
	for p in paths:
		try:
			fp = Path(p)
			if fp.exists():
				if fp.is_file():
					fp.unlink()
				else:
					# remove directory contents
					for child in fp.glob('test_*'):
						try:
							child.unlink()
						except Exception:
							pass
		except Exception:
			pass


def test_model_options_and_current_model_behavior():
	"""Model listing and current-model when none exists."""
	# ensure saved_model dir exists
	os.makedirs("models/saved_model", exist_ok=True)
	# create a dummy test model file
	test_model = Path("models/saved_model/test_dummy_model.pth")
	if test_model.exists():
		test_model.unlink()
	test_model.write_bytes(b"dummy")

	resp = client.get("/model-options")
	assert resp.status_code == 200
	data = resp.json()
	assert "saved_models" in data
	assert any(f.startswith("test_dummy_model") for f in data["saved_models"]) 

	# current-model should return loaded False if no current_model.pth
	cur = client.get("/current-model")
	assert cur.status_code == 200
	assert cur.json().get("loaded") in (False, None)

	# cleanup
	_cleanup(str(test_model))


def test_start_training_requires_dataset():
	"""If no dataset loaded, starting training should fail with 400."""
	_cleanup("data/current_dataset.npz")
	resp = client.post("/start-training", params={"epochs": 1})
	assert resp.status_code == 400


def test_start_training_and_reset_with_monkeypatch(monkeypatch):
	"""Start training when dataset is loaded. Monkeypatch the heavy run_training to a light stub.

	This prevents long-running work and uncontrolled model file creation while still
	exercising the router logic (metrics set, task scheduled).
	"""
	# prepare and load a small dataset (filename must start with test_)
	filename = "test_train_dataset.npz"
	_write_dataset(filename, n_samples=12, input_dim=4, val_ratio=0.25)
	load = client.post("/load-dataset", params={"filename": filename, "data_type": "synthetic"})
	assert load.status_code == 200

	# monkeypatch the run_training coroutine used by the router to a lightweight stub
	async def dummy_run_training(train_data, val_data, epochs, learning_rate, encoding_dim, **kwargs):
		# simulate a short background task
		training_metrics = __import__('autoencoder_for_labelling.services.training_service', fromlist=['training_metrics']).training_metrics
		training_metrics["is_training"] = True
		await asyncio.sleep(0.01)
		training_metrics["is_training"] = False

	monkeypatch.setattr("autoencoder_for_labelling.routers.training_router.run_training", dummy_run_training)

	# start training: should return 200 and set metrics
	resp = client.post("/start-training", params={"epochs": 1, "learning_rate": 0.01, "encoding_dim": 4})
	assert resp.status_code == 200
	j = resp.json()
	assert j.get("status") == "Training started"

	# immediately query training-status: metrics should have been updated by router before task scheduling
	status = client.get("/training-status")
	assert status.status_code == 200
	st = status.json()
	assert "is_training" in st

	# call reset-training to clear state
	r = client.post("/reset-training")
	assert r.status_code == 200
	s2 = client.get("/training-status").json()
	assert s2.get("is_training") is False

	# cleanup dataset files
	_cleanup("data/synthetic/" + filename, "data/current_dataset.npz")



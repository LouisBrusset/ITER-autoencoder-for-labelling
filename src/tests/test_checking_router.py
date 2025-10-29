from pathlib import Path
import os
import numpy as np

import pytest
from fastapi.testclient import TestClient
from autoencoder_for_labelling.main import app

client = TestClient(app)


def _write_synthetic_file(filename: str, n_samples: int = 10, input_dim: int = 6, is_train=None):
	"""Helper: write a synthetic npz to data/synthetic/<filename>. The file's name should start with 'test_'."""
	os.makedirs("data/synthetic", exist_ok=True)
	data = np.random.rand(n_samples, input_dim)
	labels = np.zeros(n_samples, dtype=int)
	if is_train is None:
		is_train = np.ones(n_samples, dtype=bool)
	np.savez(Path("data/synthetic") / filename, data=data, labels=labels, is_train=is_train)


def _cleanup_files(*paths):
	for p in paths:
		try:
			if Path(p).exists():
				Path(p).unlink()
		except Exception:
			pass


def test_no_dataset_returns_400():
	"""When no current dataset is present, endpoint should return 400."""
	# ensure current dataset removed
	_cleanup_files("data/current_dataset.npz")
	resp = client.get("/check-sample-plot")
	assert resp.status_code == 400


def test_check_sample_plot_global_index():
	"""Plot for a global index should return PNG image bytes."""
	filename = "test_check_dataset.npz"
	_write_synthetic_file(filename)

	# load dataset via API (this will create data/current_dataset.npz)
	load = client.post("/load-dataset", params={"filename": filename, "data_type": "synthetic"})
	assert load.status_code == 200

	resp = client.get("/check-sample-plot?index=0")
	assert resp.status_code == 200
	assert resp.headers.get("content-type") == "image/png"
	assert len(resp.content) > 100  # non-trivial png

	# cleanup
	_cleanup_files("data/synthetic/" + filename, "data/current_dataset.npz")


def test_check_sample_plot_train_subset_index():
	"""When subset='train', index refers to training subset indices."""
	filename = "test_check_dataset_train.npz"
	n = 8
	is_train = np.array([True, True, False, False, True, True, False, True], dtype=bool)
	_write_synthetic_file(filename, n_samples=n, input_dim=4, is_train=is_train)

	load = client.post("/load-dataset", params={"filename": filename, "data_type": "synthetic"})
	assert load.status_code == 200

	# request first train sample
	resp = client.get("/check-sample-plot?index=0&subset=train")
	assert resp.status_code == 200
	assert resp.headers.get("content-type") == "image/png"
	assert len(resp.content) > 100

	_cleanup_files("data/synthetic/" + filename, "data/current_dataset.npz")


def test_check_sample_plot_index_out_of_range_and_no_train_samples():
	"""Covers two error conditions: global index out of range and no train samples for subset='train'."""
	# prepare a small dataset
	filename = "test_check_dataset_errors.npz"
	n = 5
	# create a dataset where all samples are validation (no train)
	is_train_all_false = np.zeros(n, dtype=bool)
	_write_synthetic_file(filename, n_samples=n, input_dim=3, is_train=is_train_all_false)

	load = client.post("/load-dataset", params={"filename": filename, "data_type": "synthetic"})
	assert load.status_code == 200

	# global index out of range
	resp = client.get(f"/check-sample-plot?index={n}")
	assert resp.status_code == 400

	# subset='train' when there are no training samples should return 400
	resp2 = client.get("/check-sample-plot?index=0&subset=train")
	assert resp2.status_code == 400

	_cleanup_files("data/synthetic/" + filename, "data/current_dataset.npz")





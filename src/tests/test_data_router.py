from pathlib import Path
import os
import io
import numpy as np

import pytest
from fastapi.testclient import TestClient
from autoencoder_for_labelling.main import app

client = TestClient(app)


def _cleanup_data():
	"""Remove test-created dataset and upload files and current_dataset.npz.

	Tests should create files prefixed with `test_` so this helper can safely
	remove transient artifacts without affecting real data.
	"""
	try:
		updir = Path("data/uploaded")
		if updir.exists():
			for p in updir.glob("test_*.npz"):
				try:
					p.unlink()
				except Exception:
					pass
		sdir = Path("data/synthetic")
		if sdir.exists():
			for p in sdir.glob("test_*.npz"):
				try:
					p.unlink()
				except Exception:
					pass
		cur = Path("data/current_dataset.npz")
		if cur.exists():
			try:
				cur.unlink()
			except Exception:
				pass
	except Exception:
		pass


def test_get_data_options():
	"""Basic smoke test for /data-options endpoint."""
	resp = client.get("/data-options")
	assert resp.status_code == 200
	data = resp.json()
	assert "uploaded_files" in data
	assert "synthetic_files" in data


def test_create_and_delete_synthetic_data():
	"""Create a small synthetic dataset and then delete the created file."""
	# create synthetic data via API
	params = {"n_samples": 100, "input_dim": 5, "n_anomalies": 5, "val_ratio": 0.1}
	resp = client.post("/create-synthetic-data", params=params)
	assert resp.status_code == 200
	result = resp.json()
	assert result["status"] == "success"
	filename = result["filename"]

	# file should exist in data/synthetic
	synthetic_path = Path("data/synthetic") / filename
	assert synthetic_path.exists()

	# rename the created synthetic file to start with test_ so tests are easy to clean
	test_name = f"test_{filename}"
	test_path = synthetic_path.parent / test_name
	synthetic_path.rename(test_path)
	assert test_path.exists()

	# cleanup by calling delete-dataset endpoint on the test-prefixed filename
	del_resp = client.delete("/delete-dataset", params={"filename": test_name, "data_type": "synthetic"})
	assert del_resp.status_code == 200
	del_result = del_resp.json()
	assert del_result["status"] == "success"
	assert not test_path.exists()


def test_load_and_current_dataset_cycle():
	"""Save a small npz into data/synthetic, load it with API, then check current-dataset and cleanup."""
	# prepare a small dataset and write to data/synthetic
	os.makedirs("data/synthetic", exist_ok=True)
	data = np.random.rand(30, 4)
	labels = np.zeros(30, dtype=int)
	filename = "test_small_dataset.npz"
	filepath = Path("data/synthetic") / filename
	np.savez(filepath, data=data, labels=labels)

	# load it via API
	resp = client.post("/load-dataset", params={"filename": filename, "data_type": "synthetic"})
	assert resp.status_code == 200
	load_result = resp.json()
	assert load_result["status"] == "success"

	# current-dataset should report loaded True and the filename
	cur = client.get("/current-dataset")
	assert cur.status_code == 200
	cur_json = cur.json()
	assert cur_json.get("loaded", False) is True
	assert filename in cur_json.get("filename", "")
	assert "shape" in cur_json

	# cleanup transient test files
	_cleanup_data()


def test_upload_data_npz_and_cleanup():
	"""Upload a small npz file using the upload endpoint and ensure it is saved to data/uploaded."""
	# prepare an in-memory npz
	buf = io.BytesIO()
	a = np.arange(12).reshape(3, 4)
	np.savez(buf, data=a)
	buf.seek(0)

	# use a filename that begins with test_ so it's easy to find/cleanup
	files = {"file": ("test_uploaded_test.npz", buf.read(), "application/octet-stream")}
	resp = client.post("/upload-data", files=files)
	assert resp.status_code == 200
	resj = resp.json()
	assert resj["status"] == "success"
	uploaded_path = Path("data/uploaded") / "test_uploaded_test.npz"
	assert uploaded_path.exists()

	# cleanup transient test files
	_cleanup_data()


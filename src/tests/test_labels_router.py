from pathlib import Path
import os
import json
import numpy as np

import pytest
from fastapi.testclient import TestClient
from autoencoder_for_labelling.main import app

client = TestClient(app)



def _write_test_dataset(filename: str, n_samples: int = 10, input_dim: int = 4, val_ratio: float = 0.3):
	os.makedirs("data/synthetic", exist_ok=True)
	data = np.random.rand(n_samples, input_dim)
	labels = np.zeros(n_samples, dtype=int)
	n_val = max(1, int(n_samples * val_ratio))
	is_train = np.ones(n_samples, dtype=bool)
	is_train[-n_val:] = False
	np.savez_compressed(Path("data/synthetic") / filename, data=data, labels=labels, is_train=is_train)


def _cleanup_files(*paths):
	for p in paths:
		try:
			fp = Path(p)
			if fp.exists():
				fp.unlink()
		except Exception:
			pass


def test_save_labels_errors_and_success_flow():
	# ensure no dataset -> save-labels should return 400
	_cleanup_files("data/current_dataset.npz")
	bad_payload = {"labels": [1, 0]}
	r = client.post("/save-labels", json=bad_payload)
	assert r.status_code == 400

	# create and load a test dataset
	ds_name = "test_labels_dataset.npz"
	_write_test_dataset(ds_name, n_samples=8, input_dim=3, val_ratio=0.25)
	load = client.post("/load-dataset", params={"filename": ds_name, "data_type": "synthetic"})
	assert load.status_code == 200

	# missing indices -> 400
	payload = {"labels": [1, 1]}
	r2 = client.post("/save-labels", json=payload)
	assert r2.status_code == 400

	# mismatched lengths -> 400
	payload2 = {"labels": [1, 0, 1], "indices": [0, 1]}
	r3 = client.post("/save-labels", json=payload2)
	assert r3.status_code == 400

	# out-of-range index -> 400
	payload3 = {"labels": [1], "indices": [999]}
	r4 = client.post("/save-labels", json=payload3)
	assert r4.status_code == 400

	# happy path: apply labels to indices
	payload_ok = {"labels": [1, 2], "indices": [0, 2]}
	r5 = client.post("/save-labels", json=payload_ok)
	assert r5.status_code == 200
	j = r5.json()
	assert j.get("n_labels") == 2

	# confirm data/current_dataset.npz was updated with labels
	cur = np.load("data/current_dataset.npz", allow_pickle=True)
	assert 'labels' in cur
	labels_arr = np.array(cur['labels'], dtype=int)
	assert labels_arr[0] == 1
	assert labels_arr[2] == 2

	# create a current_latents.npz covering full dataset and ensure labels propagate
	n = labels_arr.shape[0]
	latents = np.random.rand(n, 2)
	indices = np.arange(n, dtype=int)
	os.makedirs('results', exist_ok=True)
	np.savez_compressed('results/current_latents.npz', data=latents, indices=indices)

	# apply another label which should update current_latents.npz labels
	payload_ok2 = {"labels": [3], "indices": [1]}
	r6 = client.post("/save-labels", json=payload_ok2)
	assert r6.status_code == 200

	meta = np.load('results/current_latents.npz', allow_pickle=True)
	# if labels key present, it should be consistent with dataset labels where possible
	if 'labels' in meta:
		meta_labels = np.array(meta['labels'], dtype=int)
		assert meta_labels[1] == 3

	# test export-all-labels: should create a JSON file under results/labels
	exp = client.get('/export-all-labels')
	assert exp.status_code == 200
	expj = exp.json()
	fpath = expj.get('file')
	assert fpath is not None and Path(fpath).exists()

	# load saved JSON and confirm keys
	with open(fpath, 'r') as fh:
		payload = json.load(fh)
	assert 'trainset' in payload and 'validationset' in payload

	# cleanup test files
	_cleanup_files('data/synthetic/' + ds_name, 'data/current_dataset.npz', 'results/current_latents.npz')
	# remove exported labels file
	try:
		if fpath:
			Path(fpath).unlink()
	except Exception:
		pass



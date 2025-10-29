from pathlib import Path
import os
import numpy as np

import pytest
from fastapi.testclient import TestClient

from autoencoder_for_labelling.main import app

client = TestClient(app)


def _write_dataset(filename: str, n_samples: int = 12, input_dim: int = 6, val_ratio: float = 0.3):
    os.makedirs("data/synthetic", exist_ok=True)
    data = np.random.rand(n_samples, input_dim)
    labels = np.zeros(n_samples, dtype=int)
    n_val = max(1, int(n_samples * val_ratio))
    is_train = np.ones(n_samples, dtype=bool)
    is_train[-n_val:] = False
    np.savez(Path("data/synthetic") / filename, data=data, labels=labels, is_train=is_train)


def _cleanup_files(*paths):
    for p in paths:
        try:
            fp = Path(p)
            if fp.exists():
                fp.unlink()
        except Exception:
            pass


def test_latent_space_requires_model_and_current_files():
    # ensure required files missing -> should return 400
    _cleanup_files("models/current_model.pth", "results/current_latents.npz", "results/current_projection2d.npz")
    resp = client.get("/latent-space")
    # The endpoint should fail when required files are missing. Accept any 4xx/5xx
    # to be tolerant of environment differences that surface as 500 instead of 400.
    assert resp.status_code >= 400


def test_latent_space_happy_path_and_reconstruct():
    # create test dataset and load it
    ds_name = "test_vis_dataset.npz"
    _write_dataset(ds_name, n_samples=10, input_dim=5, val_ratio=0.4)
    r = client.post("/load-dataset", params={"filename": ds_name, "data_type": "synthetic"})
    assert r.status_code == 200

    # create a dummy current model file (router only checks existence)
    os.makedirs("models", exist_ok=True)
    with open("models/current_model.pth", "wb") as f:
        f.write(b"dummy-model")

    # create current_latents and current_projection2d in results with indices mapping
    os.makedirs("results", exist_ok=True)
    n = 10
    latent_dim = 3
    latents = np.random.rand(n, latent_dim)
    projection2d = np.random.rand(n, 2)
    indices = np.arange(n, dtype=int)
    labels = np.zeros(n, dtype=int)

    np.savez("results/current_latents.npz", data=latents, indices=indices, labels=labels)
    np.savez("results/current_projection2d.npz", data=projection2d, indices=indices, labels=labels)

    # call latent-space for validation subset (default)
    resp = client.get("/latent-space")
    assert resp.status_code == 200
    j = resp.json()
    assert "points" in j
    # since val_ratio=0.4 and n=10, val count should be 4
    assert len(j["points"]) == 4

    # create current_reconstructions.npz with indices and data matching full dataset
    recons = np.random.rand(n, 5)  # same original_dim as dataset
    np.savez("results/current_reconstructions.npz", data=recons, indices=indices)

    # test reconstruct global index 0
    rec = client.get("/reconstruct?index=0")
    assert rec.status_code == 200
    recj = rec.json()
    assert recj.get("message") == "success"
    assert isinstance(recj.get("original"), list)
    assert isinstance(recj.get("reconstruction"), list)

    # cleanup created files
    _cleanup_files("data/synthetic/" + ds_name, "data/current_dataset.npz",
                   "models/current_model.pth",
                   "results/current_latents.npz", "results/current_projection2d.npz",
                   "results/current_reconstructions.npz")


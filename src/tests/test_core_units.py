"""Unit tests for core modules: dataset generation, VAE loss, model forward/encode/decode and trainer.

This single file contains lightweight, fast tests that exercise the public
functions/classes in:
- autoencoder_for_labelling.data.dataset.generate_synthetic_data
- autoencoder_for_labelling.training.vae_loss.vae_loss
- autoencoder_for_labelling.models.autoencoder.Convolutional_Beta_VAE
- autoencoder_for_labelling.training.trainer.train_autoencoder

Tests are intentionally small (few samples / small dims) so they run quickly
on CI and local dev machines.
"""
from __future__ import annotations

import numpy as np
import torch

import pytest

from autoencoder_for_labelling.data.dataset import generate_synthetic_data
from autoencoder_for_labelling.training.vae_loss import vae_loss
from autoencoder_for_labelling.models.autoencoder import Convolutional_Beta_VAE
from autoencoder_for_labelling.training.trainer import train_autoencoder


def test_generate_synthetic_data_shapes_and_anomalies():
    """generate_synthetic_data returns arrays of the expected shape and anomaly count."""
    n_samples = 20
    input_dim = 6
    n_anomalies = 3
    data, labels, is_train = generate_synthetic_data(n_samples=n_samples, input_dim=input_dim, n_anomalies=n_anomalies, val_ratio=0.25)

    assert isinstance(data, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(is_train, np.ndarray)

    assert data.shape == (n_samples, input_dim)
    assert labels.shape == (n_samples,)
    assert is_train.shape == (n_samples,)

    # label sum should equal the number of anomalies we asked for
    assert int(labels.sum()) == n_anomalies


def test_vae_loss_returns_three_tensors_and_values():
    """vae_loss returns (total, recon_loss, kld_loss) and they are tensors/scalars."""
    batch = 4
    input_dim = 5
    latent_dim = 2
    recon = torch.randn(batch, input_dim)
    x = torch.randn(batch, input_dim)
    mu = torch.randn(batch, latent_dim)
    logvar = torch.randn(batch, latent_dim)

    total, recon_loss, kld_loss = vae_loss(recon, x, mu, logvar, beta=0.05)

    assert isinstance(total, torch.Tensor)
    assert isinstance(recon_loss, torch.Tensor)
    assert isinstance(kld_loss, torch.Tensor)
    # values should be finite numbers
    assert torch.isfinite(total)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kld_loss)


def test_autoencoder_forward_encode_decode_shapes():
    """Check forward/encode/decode shapes for a small dense (non-conv) autoencoder."""
    input_dim = 8
    encoding_dim = 3
    # use conv_layers=0 to exercise the dense path which is simpler in tests
    model = Convolutional_Beta_VAE(input_dim=input_dim, encoding_dim=encoding_dim, conv_layers=0)

    batch = 5
    x = torch.randn(batch, input_dim)

    out = model(x)
    # forward returns (recon, mu, logvar)
    assert isinstance(out, tuple) and len(out) == 3
    recon, mu, logvar = out
    assert recon.shape == (batch, input_dim)
    assert mu.shape[0] == batch
    assert logvar.shape[0] == batch

    # encode and decode helpers
    mu_enc, logvar_enc = model.encode(x)
    assert mu_enc.shape[0] == batch
    dec = model.decode(mu_enc)
    # decode should return tensor shaped (batch, input_dim)
    assert dec.shape[0] == batch and dec.shape[1] == input_dim


def test_train_autoencoder_one_epoch_runs_and_returns_losses():
    """Run a single epoch of train_autoencoder on a small synthetic dataset.

    This ensures the training loop and loss plumbing works end-to-end.
    """
    input_dim = 8
    encoding_dim = 2
    model = Convolutional_Beta_VAE(input_dim=input_dim, encoding_dim=encoding_dim, conv_layers=0)

    # small train/val tensors
    train_data = torch.randn(6, input_dim)
    val_data = torch.randn(2, input_dim)

    train_loss, val_loss = train_autoencoder(model, train_data, val_data, epoch=1, learning_rate=1e-2, beta=0.01, verbose=False)

    assert isinstance(train_loss, float)
    # val_loss may be None if val_data empty; here we provided val_data so expect float
    assert isinstance(val_loss, float) or val_loss is None
    assert train_loss >= 0.0
    if val_loss is not None:
        assert val_loss >= 0.0

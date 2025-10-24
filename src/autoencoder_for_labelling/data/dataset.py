import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_data(
        n_samples: int = 1000,
        input_dim: int = 20,
        n_anomalies: int = 50,
        val_ratio: float = 0.2
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset with normal and anomalous data points and return a combined
    numpy array along with labels and a boolean is_train mask.

    Returns:
        data: np.ndarray shape (n_samples, input_dim)
        labels: np.ndarray shape (n_samples,) with 0/1 (anomaly) values but train entries
                will be set to 0 (labels are only meaningful for validation set)
        is_train: np.ndarray[bool] shape (n_samples,) True for train samples, False for val
    """
    # basic normal cluster
    normal_data, _ = make_blobs(
        n_samples=n_samples - n_anomalies,
        n_features=input_dim,
        centers=3,
        cluster_std=0.5
    )
    anomaly_data = np.random.normal(5, 2, (n_anomalies, input_dim))
    all_data = np.vstack([normal_data, anomaly_data])

    # Normalization
    scaler = MinMaxScaler()
    all_data = scaler.fit_transform(all_data)

    # Shuffle and split
    rng = np.random.default_rng()
    perm = rng.permutation(n_samples)
    all_data = all_data[perm]

    # labels: mark anomalies (we created anomalies as last n_anomalies before shuffle)
    labels = np.zeros(n_samples, dtype=int)
    labels[-n_anomalies:] = 1
    labels = labels[perm]

    # Determine train/val split
    if val_ratio < 0 or val_ratio >= 1:
        val_ratio = 0.2
    n_val = max(1, int(n_samples * val_ratio))
    is_train = np.ones(n_samples, dtype=bool)
    # choose last n_val entries as validation set (but shuffled already)
    val_idx = np.arange(n_samples - n_val, n_samples)
    is_train[val_idx] = False

    return all_data, labels, is_train
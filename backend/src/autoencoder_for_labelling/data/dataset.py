import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_data(
        n_samples: int = 1000, 
        input_dim: int = 20, 
        n_anomalies: int = 50
        ) -> tuple[torch.FloatTensor]:
    """
    Generate synthetic dataset with normal and anomalous data points.
    """
    normal_data, _ = make_blobs(
        n_samples=n_samples - n_anomalies, 
        n_features=input_dim, 
        centers=1, 
        cluster_std=0.5
    )
    anomaly_data = np.random.normal(5, 2, (n_anomalies, input_dim))
    all_data = np.vstack([normal_data, anomaly_data])
    
    # Normalization
    scaler = MinMaxScaler()
    all_data = scaler.fit_transform(all_data)
    
    train_data = torch.FloatTensor(all_data)
    return train_data, train_data  # Same data for train and test for simplicity
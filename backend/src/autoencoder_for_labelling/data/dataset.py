import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_data(n_samples=1000, input_dim=20, n_anomalies=50):
    """
    Génère des données synthétiques avec quelques anomalies
    """
    # Données normales
    normal_data, _ = make_blobs(
        n_samples=n_samples - n_anomalies, 
        n_features=input_dim, 
        centers=1, 
        cluster_std=0.5
    )
    
    # Données anormales (bruit)
    anomaly_data = np.random.normal(5, 2, (n_anomalies, input_dim))
    
    # Combiner les données
    all_data = np.vstack([normal_data, anomaly_data])
    
    # Normaliser entre 0 et 1
    scaler = MinMaxScaler()
    all_data = scaler.fit_transform(all_data)
    
    # Convertir en tenseurs PyTorch
    train_data = torch.FloatTensor(all_data)
    
    return train_data, train_data  # Pour simplifier, on utilise les mêmes données pour train et test
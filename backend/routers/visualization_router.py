import os

from fastapi import APIRouter, HTTPException

import numpy as np
import torch
import umap

from src.autoencoder_for_labelling.models.autoencoder import SimpleAutoencoder

router = APIRouter()


@router.get("/latent-space")
async def get_latent_space():
    """Compute and return 2D UMAP projection of the latent space for the current dataset and model."""
    try:
        if not os.path.exists("models/current_model.pth") or not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="First load a dataset and a model")
        
        data = np.load("data/current_dataset.npz")
        dataset = torch.FloatTensor(data['data'])
        # prefer labels saved in results/ if present
        labels = None
        results_dir = 'results'
        if os.path.exists(results_dir):
            files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.startswith('labels_') and f.endswith('.json')]
            if files:
                latest = max(files, key=os.path.getmtime)
                try:
                    import json
                    with open(latest, 'r') as fh:
                        payload = json.load(fh)
                        # support new format with 'label'+'index' or older 'labels'
                        if 'label' in payload and len(payload['label']) == dataset.shape[0]:
                            labels = np.array(payload['label'])
                        elif 'labels' in payload and len(payload['labels']) == dataset.shape[0]:
                            labels = np.array(payload['labels'])
                except Exception:
                    labels = None
        if labels is None:
            labels = data['labels'] if 'labels' in data else np.zeros(len(data['data']))
        
        input_dim = dataset.shape[1]
        state = torch.load('models/current_model.pth')
        encoding_dim = None
        if isinstance(state, dict):
            if 'encoder.2.weight' in state:
                encoding_dim = state['encoder.2.weight'].shape[0]
        if encoding_dim is None:
            encoding_dim = 8
        model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
        model.load_state_dict(state)
        model.eval()
        
        # Calculate latent space & UMAP projection to 2D
        with torch.no_grad():
            latent_representations = model.encoder(dataset).numpy()
        reducer = umap.UMAP(n_components=2, random_state=42)
        latent_2d = reducer.fit_transform(latent_representations)

        # Prepare data for Chart.js
        points = []
        for i, (x, y) in enumerate(latent_2d):
            points.append({
                "x": float(x), 
                "y": float(y), 
                "label": int(labels[i]) if i < len(labels) else 0, 
                "original_index": i
            })
        
        return {
            "points": points,
            "latent_dim": int(latent_representations.shape[1]),
            "original_dim": int(dataset.shape[1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reconstruct")
async def reconstruct_sample(index: int = 0):
    """Retrun original and reconstructed data for a given sample index."""
    try:
        # Check files
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="No dataset loaded")
        if not os.path.exists("models/current_model.pth"):
            raise HTTPException(status_code=400, detail="No model loaded")
        
        # Load data
        data = np.load("data/current_dataset.npz")
        X = data['data']
        n_samples = X.shape[0]
        if index < 0 or index >= n_samples:
            raise HTTPException(status_code=400, detail=f"Index out of range (0..{n_samples-1})")
        
        sample = torch.FloatTensor(X[index:index+1])

        # load model state and instantiate model
        state = torch.load('models/current_model.pth')
        encoding_dim = None
        if isinstance(state, dict) and 'encoder.2.weight' in state:
            encoding_dim = state['encoder.2.weight'].shape[0]
        if encoding_dim is None:
            encoding_dim = 8

        input_dim = X.shape[1]
        model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            reconstructed = model(sample).numpy()[0]

        original = X[index].tolist()
        reconstruction = reconstructed.tolist()

        return {
            "index": int(index), 
            "original": original, 
            "reconstruction": reconstruction, 
            "message": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

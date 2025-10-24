import os

from fastapi import APIRouter, HTTPException

import numpy as np
import torch
import umap

from autoencoder_for_labelling.models.autoencoder import SimpleAutoencoder

router = APIRouter()


@router.get("/latent-space")
async def get_latent_space(subset: str | None = 'validation', deterministic: bool = True):
    """
    Compute and return 2D UMAP projection of the latent space for the current dataset and model.

    Query parameter `subset` controls which samples are used for projection:
        - 'train' or 'training' : use the training subset
        - 'validation' or 'val' : use the validation subset (default)
        - 'all' : use the entire dataset
    """
    try:
        if not os.path.exists("models/current_model.pth") or not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="First load a dataset and a model")
        
        data = np.load("data/current_dataset.npz", allow_pickle=True)
        all_data = data['data']

        # Determine train/validation indices; legacy support: if no is_train assume all training
        if 'is_train' in data:
            is_train = np.array(data['is_train'], dtype=bool)
        else:
            is_train = np.ones(len(all_data), dtype=bool)
        train_indices = np.where(is_train)[0]
        val_indices = np.where(~is_train)[0]

        # choose subset
        s = (subset or 'validation').lower()
        if s in ('validation', 'val'):
            if len(val_indices) == 0:
                raise HTTPException(status_code=400, detail='No validation samples available — create dataset with a validation split')
            dataset = torch.FloatTensor(all_data[val_indices])
            working_indices = val_indices
        elif s in ('train', 'training'):
            if len(train_indices) == 0:
                raise HTTPException(status_code=400, detail='No training samples available')
            dataset = torch.FloatTensor(all_data[train_indices])
            working_indices = train_indices
        elif s in ('all', 'global', 'dataset'):
            dataset = torch.FloatTensor(all_data)
            working_indices = np.arange(len(all_data))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown subset '{subset}'. Use 'train', 'validation' or 'all'.")

        # prefer labels saved in results/labels if present (labels will refer to global indices)
        labels = None
        results_dir = os.path.join('results', 'labels')
        if os.path.exists(results_dir):
            files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.startswith('labels_') and f.endswith('.json')]
            if files:
                latest = max(files, key=os.path.getmtime)
                try:
                    import json
                    with open(latest, 'r') as fh:
                        payload = json.load(fh)
                        # payload contains 'label' and 'index' lists — map them into full labels array
                        if 'label' in payload and 'index' in payload:
                            full_labels = np.zeros(len(all_data), dtype=int)
                            for idx, lab in zip(payload['index'], payload['label']):
                                if 0 <= idx < len(full_labels):
                                    full_labels[int(idx)] = int(lab)
                            # extract labels for the chosen working indices
                            labels = full_labels[working_indices]
                        else:
                            labels = None
                except Exception:
                    labels = None
        if labels is None:
            # if no results file, prefer labels field from dataset, but only for validation indices
            if 'labels' in data:
                full_labels = data['labels']
                labels = full_labels[working_indices]
            else:
                labels = np.zeros(len(working_indices), dtype=int)
        
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
        if deterministic:
            reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        else:
            reducer = umap.UMAP(n_components=2, init='spectral', n_neighbors=15, n_jobs=-1, random_state=None)
        latent_2d = reducer.fit_transform(latent_representations)

        # Prepare data for Chart.js
        points = []
        for i_local, (x, y) in enumerate(latent_2d):
            global_i = int(working_indices[i_local])
            points.append({
                "x": float(x),
                "y": float(y),
                "label": int(labels[i_local]) if i_local < len(labels) else 0,
                "original_index": global_i
            })
        
        return {
            "points": points,
            "latent_dim": int(latent_representations.shape[1]),
            "original_dim": int(dataset.shape[1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reconstruct")
async def reconstruct_sample(index: int = 0, subset: str | None = None):
    """Return original and reconstructed data for a given sample index.

    Parameters:
    - index: integer. Interpretation depends on `subset`:
        * subset='val' -> index is taken relative to the validation subset (0..n_val-1)
        * subset='train' -> index is taken relative to the training subset (0..n_train-1)
        * subset omitted -> index is treated as a global dataset index (0..n-1)
    """
    try:
        # Check files
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="No dataset loaded")
        if not os.path.exists("models/current_model.pth"):
            raise HTTPException(status_code=400, detail="No model loaded")
        
        # Load data and determine indices
        data = np.load("data/current_dataset.npz", allow_pickle=True)
        all_data = data['data']
        if 'is_train' in data:
            is_train = np.array(data['is_train'], dtype=bool)
        else:
            is_train = np.ones(len(all_data), dtype=bool)

        train_indices = np.where(is_train)[0]
        val_indices = np.where(~is_train)[0]

        # Interpret index depending on requested subset
        if subset == 'val':
            if len(val_indices) == 0:
                raise HTTPException(status_code=400, detail='No validation samples available — create dataset with a validation split')
            if index < 0 or index >= len(val_indices):
                raise HTTPException(status_code=400, detail=f"Index out of range (0..{len(val_indices)-1}) for validation set")
            global_idx = int(val_indices[index])
        elif subset == 'train':
            if len(train_indices) == 0:
                raise HTTPException(status_code=400, detail='No training samples available')
            if index < 0 or index >= len(train_indices):
                raise HTTPException(status_code=400, detail=f"Index out of range (0..{len(train_indices)-1}) for training set")
            global_idx = int(train_indices[index])
        else:
            # treat index as a global index
            n_samples = all_data.shape[0]
            if index < 0 or index >= n_samples:
                raise HTTPException(status_code=400, detail=f"Index out of range (0..{n_samples-1})")
            global_idx = int(index)

        sample = torch.FloatTensor(all_data[global_idx:global_idx+1])

        # load model state and instantiate model
        state = torch.load('models/current_model.pth')
        encoding_dim = None
        if isinstance(state, dict) and 'encoder.2.weight' in state:
            encoding_dim = state['encoder.2.weight'].shape[0]
        if encoding_dim is None:
            encoding_dim = 8

        # input dim comes from the full dataset
        input_dim = all_data.shape[1]
        model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            recon_t = model(sample)
            reconstructed = recon_t.detach().cpu().numpy()[0]

        # original sample from the global dataset
        original = all_data[global_idx].tolist()
        reconstruction = reconstructed.tolist()

        return {
            "index": int(global_idx),
            "original": original,
            "reconstruction": reconstruction,
            "message": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os
import re
import asyncio

from fastapi import APIRouter, HTTPException

import torch
import numpy as np

from autoencoder_for_labelling.services.training_service import training_metrics, run_training, perform_full_inference
from autoencoder_for_labelling.models.autoencoder import SimpleAutoencoder

router = APIRouter()


@router.get("/model-options")
async def get_model_options():
    """Give saved model options"""
    saved_models = os.listdir("models/saved")
    return {"saved_models": saved_models}


@router.delete("/delete-model")
async def delete_model(model_filename: str):
    """Delete a specific saved model file"""
    try:
        model_path = f"models/saved/{model_filename}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # If it was the current model, remove that as well
        if os.path.exists("models/current_model.pth"):
            try:
                current_state = torch.load("models/current_model.pth")
                saved_state = torch.load(model_path)

                def state_dicts_equal(a, b):
                    # Both must be dict-like
                    if not isinstance(a, dict) or not isinstance(b, dict):
                        return False
                    if set(a.keys()) != set(b.keys()):
                        return False
                    for k in a.keys():
                        va = a[k]
                        vb = b[k]
                        # If tensors, use torch.equal or allclose
                        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                            try:
                                if va.shape != vb.shape:
                                    return False
                                if not torch.equal(va, vb):
                                    # fallback to allclose for floating types
                                    if not torch.allclose(va, vb, atol=1e-6, rtol=1e-5):
                                        return False
                            except Exception:
                                return False
                        else:
                            # fallback to simple equality for non-tensor values
                            if va != vb:
                                return False
                    return True
                
                if state_dicts_equal(current_state, saved_state):
                    try:
                        os.remove("models/current_model.pth")
                    except Exception:
                        # ignore errors removing the current model file
                        pass
            except Exception:
                # If any error occurs while comparing/loading, don't block deletion of the saved model
                pass

        os.remove(model_path)
        
        return {
            "status": "success", 
            "model_filename": model_filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-model")
async def get_current_model():
    """Return info about the currently loaded model (if any)."""
    try:
        if not os.path.exists("models/current_model.pth"):
            return {"loaded": False}
        
        state = torch.load("models/current_model.pth")
        encoding_dim = None
        if isinstance(state, dict) and 'encoder.2.weight' in state:
            encoding_dim = int(state['encoder.2.weight'].shape[0])
        
        # try to infer input_dim from current dataset
        input_dim = None
        if os.path.exists("data/current_dataset.npz"):
            try:
                d = np.load("data/current_dataset.npz")
                if 'data' in d:
                    input_dim = int(d['data'].shape[1])
            except Exception:
                input_dim = None

        return {
            "loaded": True, 
            "encoding_dim": encoding_dim, 
            "input_dim": input_dim
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-training")
async def start_training(epochs: int = 100, learning_rate: float = 0.001, encoding_dim: int = 8):
    """Start training the autoencoder model"""
    if training_metrics["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Check if a dataset is loaded
    if not os.path.exists("data/current_dataset.npz"):
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    data = np.load("data/current_dataset.npz", allow_pickle=True)
    all_data = data['data']
    # if is_train is present, split accordingly, else assume all data is train and val empty
    if 'is_train' in data:
        is_train = data['is_train']
    else:
        is_train = np.ones(len(all_data), dtype=bool)
    train_idx = np.where(is_train)[0]
    val_idx = np.where(~is_train)[0]
    train_data = torch.FloatTensor(all_data[train_idx])
    val_data = torch.FloatTensor(all_data[val_idx]) if len(val_idx) > 0 else None
    
    training_metrics.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": epochs,
        "train_loss_history": [],
        "val_loss_history": [],
        "current_train_loss": 0.0,
        "current_val_loss": None,
        "epochs_data": []
    })

    asyncio.create_task(run_training(train_data, val_data, epochs, learning_rate, encoding_dim))
    
    return {"status": "Training started", "total_epochs": epochs}


@router.get("/training-status")
async def get_training_status():
    return training_metrics


@router.post("/load-model")
async def load_model(model_filename: str):
    """Load a specific saved model file as the current model"""
    try:
        model_path = f"models/saved/{model_filename}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="First load a dataset")
        
        # Check if a dataset is loaded to know the dimension
        data = np.load("data/current_dataset.npz")
        input_dim = data['data'].shape[1]
        
        # Load the saved state dict first so we can infer encoding dim
        state = torch.load(model_path)
        encoding_dim = None
        if isinstance(state, dict):
            if 'encoder.2.weight' in state:     # common key for second encoder Linear layer
                encoding_dim = state['encoder.2.weight'].shape[0]
            else:
                # try to parse from filename pattern _dim_<n>
                m = re.search(r'_dim_(\d+)', model_filename)
                if m:
                    encoding_dim = int(m.group(1))
        if encoding_dim is None:
            encoding_dim = 8

        # Create model with inferred encoding_dim and load weights
        model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
        model.load_state_dict(state)
        model.eval()

        # Store the model state as current model
        torch.save(state, "models/current_model.pth")

        return {"status": "success", "model_loaded": model_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-training")
async def reset_training():
    training_metrics.update({
        "is_training": False,
        "current_epoch": 0,
        "total_epochs": 0,
        "train_loss_history": [],
        "val_loss_history": [],
        "current_train_loss": 0.0,
        "current_val_loss": None,
        "epochs_data": []
    })
    return {"status": "Training reset"}


@router.post("/run-inference")
async def run_inference(deterministic: bool = True):
    """Trigger inference using the current model and dataset and save latents/reconstructions/projection files.

    Query/JSON param `deterministic` controls deterministic UMAP.
    """
    try:
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="No dataset loaded")
        if not os.path.exists("models/current_model.pth"):
            raise HTTPException(status_code=400, detail="No model loaded")

        result = await perform_full_inference(deterministic=deterministic)
        return {"status": "inference_complete", "files": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inference-options")
async def inference_options():
    """Return available saved inference artifacts for the currently loaded dataset (latents, projections, reconstructions)."""
    try:
        current_dataset = None
        if os.path.exists("data/current_dataset.npz"):
            try:
                import numpy as _np
                d = _np.load("data/current_dataset.npz", allow_pickle=True)
                current_dataset = str(d.get('filename', 'current_dataset'))
            except Exception:
                current_dataset = None

        def list_matches(dirpath):
            out = []
            if not os.path.exists(dirpath):
                return out
            for f in os.listdir(dirpath):
                if not f.endswith('.npz'):
                    continue
                full = os.path.join(dirpath, f)
                try:
                    import numpy as _np
                    meta = _np.load(full, allow_pickle=True)
                    dsname = str(meta.get('filename', ''))
                    ts = int(meta.get('timestamp', os.path.getmtime(full))) if 'timestamp' in meta else int(os.path.getmtime(full))
                    if current_dataset is None or dsname == current_dataset:
                        out.append({'file': full, 'dataset': dsname, 'timestamp': int(ts)})
                except Exception:
                    # ignore unreadable files
                    pass
            out.sort(key=lambda x: x['timestamp'], reverse=True)
            return out

        latents = list_matches(os.path.join('results','latents'))
        projections = list_matches(os.path.join('results','projections2d'))
        reconstructions = list_matches(os.path.join('results','reconstructions'))

        return { 'latents': latents, 'projections': projections, 'reconstructions': reconstructions }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

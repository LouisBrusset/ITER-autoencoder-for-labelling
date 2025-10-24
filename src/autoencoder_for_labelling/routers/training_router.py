import os
import re
import asyncio

from fastapi import APIRouter, HTTPException

import torch
import numpy as np

from autoencoder_for_labelling.services.training_service import training_metrics, run_training
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
    
    data = np.load("data/current_dataset.npz")
    train_data = torch.FloatTensor(data['data'])
    
    training_metrics.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": epochs,
        "loss_history": [],
        "current_loss": 0.0,
        "epochs_data": [],
        "loss_data": []
    })

    asyncio.create_task(run_training(train_data, epochs, learning_rate, encoding_dim))
    
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
        "loss_history": [],
        "current_loss": 0.0,
        "epochs_data": [],
        "loss_data": []
    })
    return {"status": "Training reset"}

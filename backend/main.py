### Standard library imports
import os
import json
import asyncio
from typing import Optional

#    import warnings
#    import pkg_resources

# Conf os + warnings
#    os.environ['PYTORCH_NVFUSER_DISABLE'] = '1'
#    os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
#    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
#    warnings.filterwarnings("ignore", category=FutureWarning, module="pkg_resources")

### Third-party imports
# API framework
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response

# Data analysis imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Machine learning imports
import torch
import torch.nn as nn
import umap
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### Package in backend/src
from src.autoencoder_for_labelling.models.autoencoder import SimpleAutoencoder
from src.autoencoder_for_labelling.data.dataset import generate_synthetic_data
from src.autoencoder_for_labelling.training.trainer import train_autoencoder


# === GLOBAL SETUP ===

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/uploaded", exist_ok=True)
os.makedirs("data/synthetic", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)

# Global state
training_metrics = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss_history": [],
    "current_loss": 0.0,
    "epochs_data": [],
    "loss_data": []
}




@app.get("/")
async def root():
    return {"message": "Autoencoder Training API is running!"}


# === SECTION 1 : DATA ===

@app.get("/data-options")
async def get_data_options():
    """Give available data options"""
    uploaded_files = os.listdir("data/uploaded")
    synthetic_files = os.listdir("data/synthetic")
    
    return {
        "uploaded_files": uploaded_files,
        "synthetic_files": synthetic_files
    }


@app.get("/current-dataset")
async def get_current_dataset():
    """Yield the currently loaded dataset"""
    try:
        if os.path.exists("data/current_dataset.npz"):
            data = np.load("data/current_dataset.npz")            
            response_data = {
                "loaded": True,
                "filename": str(data.get('filename', 'current_dataset.npz'))
            }
            
            dataset = data['data']
            if hasattr(dataset, 'shape'):
                response_data["shape"] = list(dataset.shape)
            else:
                response_data["shape"] = "Unknown"
                
            return response_data
        return {"loaded": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-dataset")
async def load_dataset(filename: str, data_type: str = "synthetic"):
    """Load a specific dataset file as the current dataset"""
    try:
        if data_type == "synthetic":
            filepath = f"data/synthetic/{filename}"
        else:
            filepath = f"data/uploaded/{filename}"
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found in directory")

        # Load the data
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            data = df.values
            labels = np.zeros(len(data))
        else:
            loaded = np.load(filepath)
            if 'data' in loaded:
                data = loaded['data']
            elif 'arr_0' in loaded:
                data = loaded['arr_0']
            else:
                data = loaded[loaded.files[0]]
            
            labels = loaded['labels'] if 'labels' in loaded else np.zeros(len(data))

        # Type conversion safe for serialization
        data_for_save = data.copy() if hasattr(data, 'copy') else data
        labels_for_save = labels.copy() if hasattr(labels, 'copy') else labels
        
        np.savez("data/current_dataset.npz", 
                data=data_for_save, 
                labels=labels_for_save, 
                filename=filename)
        
        response_data = {
            "status": "success",
            "filename": filename,
        }
        if hasattr(data, 'shape'):
            response_data["data_shape"] = list(data.shape)
        if hasattr(labels, 'shape'):
            response_data["labels_shape"] = list(labels.shape)
            
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-dataset")
async def delete_dataset(filename: str, data_type: str = "synthetic"):
    """Delete a specific dataset file"""
    try:
        if data_type == "synthetic":
            filepath = f"data/synthetic/{filename}"
        else:
            filepath = f"data/uploaded/{filename}"
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="No file found in directory")
        
        os.remove(filepath)

        # If it was the current dataset, remove that as well
        if os.path.exists("data/current_dataset.npz"):
            current_data = np.load("data/current_dataset.npz")
            if current_data.get('filename', '') == filename:
                os.remove("data/current_dataset.npz")
        
        return {
            "status": "success",
            "filename": filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-synthetic-data")
async def create_synthetic_data(n_samples: int = 1000, input_dim: int = 20, n_anomalies: int = 50):
    """Create synthetic data and save to file"""
    try:
        train_tensor, _ = generate_synthetic_data(n_samples=n_samples, input_dim=input_dim, n_anomalies=n_anomalies)
        all_data = train_tensor.numpy()
        labels = np.zeros(n_samples)
        labels[-n_anomalies:] = 1

        filename = f"synthetic_data_{n_samples}_{input_dim}.npz"
        filepath = f"data/synthetic/{filename}"
        np.savez(filepath, data=all_data, labels=labels)

        return {
            "status": "success",
            "filename": filename,
            "data_shape": all_data.shape,
            "anomalies_count": n_anomalies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload a file and save to uploaded data directory"""
    try:
        if not file.filename.endswith(('.csv', '.npz')):
            raise HTTPException(status_code=400, detail="Only CSV and NPZ files are supported")
        
        filepath = f"data/uploaded/{file.filename}"
        
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            data_shape = df.shape
        else:
            data = np.load(filepath)
            data_shape = data['data'].shape if 'data' in data else data.shape
        
        return {
            "status": "success",
            "filename": file.filename,
            "data_shape": data_shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# === SECTION 2 : CHECKING ===

@app.get('/check-sample-plot')
async def check_sample_plot(index: int = 0):
    """Return a PNG image plotting original for a dataset sample index."""
    try:
        # Ensure dataset exists
        if not os.path.exists('data/current_dataset.npz'):
            raise HTTPException(status_code=400, detail='No dataset loaded')

        data = np.load('data/current_dataset.npz', allow_pickle=True)
        X = data['data']
        n = X.shape[0]
        if index < 0 or index >= n:
            raise HTTPException(status_code=400, detail=f'Index out of range (0..{n-1})')

        sample = torch.FloatTensor(X[index:index+1])
        orig = X[index]

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(orig, label='Original', color='tab:blue')
        ax.set_xlabel('Feature index')
        ax.set_ylabel('Value')
        title = f'Sample {index} - Original'
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.read(), media_type='image/png')

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# === SECTION 3 : TRAINING ===

@app.get("/model-options")
async def get_model_options():
    """Give saved model options"""
    saved_models = os.listdir("models/saved")
    return {"saved_models": saved_models}

@app.delete("/delete-model")
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

@app.get("/current-model")
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

@app.post("/start-training")
async def start_training(epochs: int = 100, learning_rate: float = 0.001, encoding_dim: int = 8):
    """Start training the autoencoder model"""
    if training_metrics["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")

    # Check if a dataset is loaded
    if not os.path.exists("data/current_dataset.npz"):
        raise HTTPException(status_code=400, detail="No dataset loaded")

    # Load the data
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


async def run_training(train_data, epochs, learning_rate, encoding_dim):
    try:
        input_dim = train_data.shape[1]
        model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)

        for epoch in range(1, epochs + 1):
            # Use the package trainer helper for a single epoch
            loss = train_autoencoder(model, train_data, epoch, learning_rate, verbose=False)

            # Store metrics
            training_metrics["current_epoch"] = epoch
            training_metrics["current_loss"] = float(loss)
            training_metrics["loss_history"].append(float(loss))
            training_metrics["epochs_data"].append(epoch)
            training_metrics["loss_data"].append(float(loss))

            await asyncio.sleep(0.05)

        # Save the trained model state dict
        model_path = f"models/saved/model_epochs_{epochs}_dim_{encoding_dim}.pth"
        torch.save(model.state_dict(), model_path)
        # also save as current model for visualization
        torch.save(model.state_dict(), "models/current_model.pth")
        
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        training_metrics["is_training"] = False


@app.get("/training-status")
async def get_training_status():
    return training_metrics


@app.post("/load-model")
async def load_model(model_filename: str):
    """Load a specific saved model file as the current model"""
    try:
        model_path = f"models/saved/{model_filename}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")

        # Check if a dataset is loaded to know the dimension
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="First load a dataset")
        
        data = np.load("data/current_dataset.npz")
        input_dim = data['data'].shape[1]
        
        # Load the saved state dict first so we can infer encoding dim
        state = torch.load(model_path)
        encoding_dim = None
        if isinstance(state, dict):
            # common key for second encoder Linear layer
            if 'encoder.2.weight' in state:
                encoding_dim = state['encoder.2.weight'].shape[0]
            else:
                # try to parse from filename pattern _dim_<n>
                import re
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
    
@app.post("/reset-training")
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




# === SECTION 4 : VISUALISATION ===

@app.get("/latent-space")
async def get_latent_space():
    """Compute and return the 2D latent space projection of the current dataset using UMAP."""
    try:
        # Check if a model and dataset are loaded
        if not os.path.exists("models/current_model.pth") or not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="First load a dataset and a model")

        # Load the data and model
        data = np.load("data/current_dataset.npz")
        dataset = torch.FloatTensor(data['data'])
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
            "original_dim": int(dataset.shape[1]),
            "latent_vectors": latent_representations.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/reconstruct")
async def reconstruct_sample(index: int = 0):
    """Return original and reconstructed sample vectors for a given index."""
    try:
        # check files
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="No dataset loaded")
        if not os.path.exists("models/current_model.pth"):
            raise HTTPException(status_code=400, detail="No model loaded")

        # load data
        data = np.load("data/current_dataset.npz")
        X = data['data']
        n_samples = X.shape[0]
        if index < 0 or index >= n_samples:
            raise HTTPException(status_code=400, detail=f"Index out of range (0..{n_samples-1})")

        sample = torch.FloatTensor(X[index:index+1])  # keep batch dim

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


### Standard library imports
import os
import json
import asyncio
from typing import Optional

### Third-party imports
# API framework
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Data analysis imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Machine learning imports
import torch
import torch.nn as nn
import umap

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




@app.post("/create-synthetic-data")
async def create_synthetic_data(n_samples: int = 1000, input_dim: int = 20, n_anomalies: int = 50):
    """Crée des données synthétiques"""
    try:
        # Use the package helper to generate synthetic data (returns torch tensors)
        train_tensor, _ = generate_synthetic_data(n_samples=n_samples, input_dim=input_dim, n_anomalies=n_anomalies)
        # convert to numpy for saving and for the frontend expectations
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
    """Upload un fichier de données"""
    try:
        # Vérifier l'extension
        if not file.filename.endswith(('.csv', '.npz')):
            raise HTTPException(status_code=400, detail="Seuls les fichiers CSV et NPZ sont supportés")
        
        filepath = f"data/uploaded/{file.filename}"
        
        # Sauvegarder le fichier
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        # Charger pour vérification
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




@app.get("/current-dataset")
async def get_current_dataset():
    """Retourne le dataset actuellement chargé"""
    try:
        if os.path.exists("data/current_dataset.npz"):
            data = np.load("data/current_dataset.npz")
            return {
                "loaded": True,
                "shape": data['data'].shape,
                "filename": data.get('filename', 'current_dataset.npz')
            }
        return {"loaded": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/load-dataset")
async def load_dataset(filename: str, data_type: str = "synthetic"):
    """Charge un dataset spécifique"""
    try:
        if data_type == "synthetic":
            filepath = f"data/synthetic/{filename}"
        else:
            filepath = f"data/uploaded/{filename}"
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        # Charger les données
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            data = df.values
            labels = np.zeros(len(data))  # Pas de labels par défaut
        else:
            loaded = np.load(filepath)
            data = loaded['data'] if 'data' in loaded else loaded['arr_0']
            labels = loaded['labels'] if 'labels' in loaded else np.zeros(len(data))
        
        # Sauvegarder comme dataset courant
        np.savez("data/current_dataset.npz", data=data, labels=labels, filename=filename)
        
        return {
            "status": "success",
            "filename": filename,
            "data_shape": data.shape,
            "labels_shape": labels.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))







# === SECTION 2 : ENTRAÎNEMENT ===

@app.get("/model-options")
async def get_model_options():
    """Retourne les modèles sauvegardés"""
    saved_models = os.listdir("models/saved")
    return {"saved_models": saved_models}




@app.post("/start-training")
async def start_training(epochs: int = 100, learning_rate: float = 0.001, encoding_dim: int = 8):
    """Démarre l'entraînement"""
    if training_metrics["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Vérifier qu'un dataset est chargé
    if not os.path.exists("data/current_dataset.npz"):
        raise HTTPException(status_code=400, detail="Aucun dataset chargé")
    
    # Charger les données
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
    """Charge un modèle pré-entraîné"""
    try:
        model_path = f"models/saved/{model_filename}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Modèle non trouvé")
        
        # Vérifier qu'un dataset est chargé pour connaître la dimension
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="Chargez d'abord un dataset")
        
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
    





# === SECTION 3 : VISUALISATION ===

@app.get("/latent-space")
async def get_latent_space():
    """Calcule et retourne le latent space projeté en 2D"""
    try:
        # Vérifier qu'un modèle et des données sont chargés
        if not os.path.exists("models/current_model.pth") or not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="Chargez un modèle et des données d'abord")
        
        # Charger les données et le modèle
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
        
        # Calculer le latent space
        with torch.no_grad():
            latent_representations = model.encoder(dataset).numpy()
        
        # Projection UMAP en 2D
        reducer = umap.UMAP(n_components=2, random_state=42)
        latent_2d = reducer.fit_transform(latent_representations)
        
        # Préparer les données pour Chart.js
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
            "latent_dim": latent_representations.shape[1],
            "original_dim": dataset.shape[1]
        }
        
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






@app.get("/")
async def root():
    return {"message": "Autoencoder Training API is running!"}
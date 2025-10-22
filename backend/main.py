from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import asyncio
from src.autoencoder_for_labelling.models.autoencoder import SimpleAutoencoder
from src.autoencoder_for_labelling.data.dataset import generate_synthetic_data
from src.autoencoder_for_labelling.training.trainer import train_autoencoder
import torch

app = FastAPI()

# Autorise les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# État global pour stocker les métriques d'entraînement
training_metrics = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss_history": [],
    "current_loss": 0.0
}

# Endpoint pour démarrer l'entraînement
@app.post("/start-training")
async def start_training():
    if training_metrics["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Réinitialiser les métriques
    training_metrics.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": 100,  # 100 epochs pour l'exemple
        "loss_history": [],
        "current_loss": 0.0
    })
    
    # Démarrer l'entraînement en arrière-plan
    asyncio.create_task(run_training())
    
    return {"status": "Training started", "total_epochs": training_metrics["total_epochs"]}

# Endpoint pour récupérer l'état actuel de l'entraînement
@app.get("/training-status")
async def get_training_status():
    return training_metrics

# Endpoint pour obtenir le plot d'entraînement
@app.get("/training-plot")
async def get_training_plot():
    try:
        plt.figure(figsize=(10, 6))
        
        if training_metrics["loss_history"]:
            epochs = range(1, len(training_metrics["loss_history"]) + 1)
            plt.plot(epochs, training_metrics["loss_history"], 'b-', label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Autoencoder Training - Epoch {training_metrics["current_epoch"]}')
            plt.legend()
            plt.grid(True)
        
        # Conversion en image base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {"image": f"data:image/png;base64,{image_base64}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Fonction d'entraînement asynchrone
async def run_training():
    try:
        # Générer des données synthétiques
        train_data, test_data = generate_synthetic_data(n_samples=1000, input_dim=20)
        
        # Initialiser le modèle
        model = SimpleAutoencoder(input_dim=20, encoding_dim=8)
        
        # Entraînement
        for epoch in range(1, training_metrics["total_epochs"] + 1):
            # Simulation de l'entraînement (remplacez par votre vrai entraînement)
            loss = train_autoencoder(model, train_data, epoch)
            
            # Mettre à jour les métriques
            training_metrics["current_epoch"] = epoch
            training_metrics["current_loss"] = loss
            training_metrics["loss_history"].append(loss)
            
            # Pause pour simuler l'entraînement
            await asyncio.sleep(0.1)  # 100ms entre les epochs pour la démo
            
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        training_metrics["is_training"] = False

# Endpoint pour réinitialiser l'entraînement
@app.post("/reset-training")
async def reset_training():
    training_metrics.update({
        "is_training": False,
        "current_epoch": 0,
        "loss_history": [],
        "current_loss": 0.0
    })
    return {"status": "Training reset"}
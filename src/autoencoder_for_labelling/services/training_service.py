import asyncio

import torch

from autoencoder_for_labelling.models.autoencoder import SimpleAutoencoder
from autoencoder_for_labelling.training.trainer import train_autoencoder

# Shared training metrics state
training_metrics = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss_history": [],
    "val_loss_history": [],
    "current_train_loss": 0.0,
    "current_val_loss": None,
    "epochs_data": []
}


async def run_training(train_data, val_data, epochs, learning_rate, encoding_dim, verbose=False):
    try:
        input_dim = train_data.shape[1]
        model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
        
        for epoch in range(1, epochs + 1):
            train_loss, val_loss = train_autoencoder(model, train_data, val_data, epoch, learning_rate, verbose=False)

            training_metrics["current_epoch"] = epoch
            training_metrics["current_train_loss"] = float(train_loss)
            training_metrics["current_val_loss"] = float(val_loss) if val_loss is not None else None
            training_metrics["train_loss_history"].append(float(train_loss))
            if val_loss is not None:
                training_metrics["val_loss_history"].append(float(val_loss))
            training_metrics["epochs_data"].append(epoch)
            
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

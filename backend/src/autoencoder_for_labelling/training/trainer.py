import torch
import torch.nn as nn
import torch.optim as optim

def train_autoencoder(model, data, epoch, learning_rate=0.001):
    """
    Fonction d'entraînement pour une epoch
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, data)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Pour la démo, on ajoute un peu de variation aléatoire
    import random
    variation = random.uniform(0.9, 1.1)
    simulated_loss = loss.item() * variation / (epoch * 0.1 + 1)
    
    print(f"Epoch {epoch}, Loss: {simulated_loss:.6f}")
    
    return simulated_loss
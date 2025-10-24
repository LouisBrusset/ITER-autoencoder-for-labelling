import torch.nn as nn
import torch.optim as optim
import torch

def train_autoencoder(
                      model: nn.Module, 
                      data: torch.Tensor, 
                      epoch: int, 
                      learning_rate: float = 0.001,
                      verbose: bool = True
                      ) -> float:
    """
    Trains the autoencoder model for one epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, data)
    loss.backward()
    optimizer.step()
    
    if verbose:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return loss.item()
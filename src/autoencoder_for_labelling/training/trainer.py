import torch.nn as nn
import torch.optim as optim
import torch

def train_autoencoder(
                      model: nn.Module,
                      train_data: torch.Tensor,
                      val_data: torch.Tensor | None,
                      epoch: int,
                      learning_rate: float = 0.001,
                      verbose: bool = True
                      ) -> tuple[float, float | None]:
    """
    Trains the autoencoder model for one epoch on train_data and optionally
    evaluates on val_data. Returns (train_loss, val_loss).
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_data)
    loss.backward()
    optimizer.step()

    train_loss = float(loss.item())
    val_loss = None
    if val_data is not None and len(val_data) > 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = float(criterion(val_outputs, val_data).item())
        model.train()

    if verbose:
        if val_loss is None:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}")
        else:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_loss, val_loss
import torch.nn as nn
import torch.optim as optim
import torch
from autoencoder_for_labelling.training.vae_loss import vae_loss


def train_autoencoder(
                      model: nn.Module,
                      train_data: torch.Tensor,
                      val_data: torch.Tensor | None,
                      epoch: int,
                      learning_rate: float = 0.001,
                      beta: float = 0.01,
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
    # If model returns (recon, mu, logvar) treat as VAE
    if isinstance(outputs, tuple) and len(outputs) == 3:
        recon, mu, logvar = outputs
        loss, recon_loss, kld_loss = vae_loss(recon, train_data, mu, logvar, beta=beta)
    else:
        recon = outputs
        loss = criterion(recon, train_data)
    loss.backward()
    optimizer.step()

    train_loss = float(loss.item())
    val_loss = None
    if val_data is not None and len(val_data) > 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            if isinstance(val_outputs, tuple) and len(val_outputs) == 3:
                v_recon, v_mu, v_logvar = val_outputs
                v_loss, v_recon_loss, v_kld_loss = vae_loss(v_recon, val_data, v_mu, v_logvar, beta=beta)
                val_loss = float(v_loss.item())
            else:
                val_loss = float(criterion(val_outputs, val_data).item())
        model.train()

    if verbose:
        if val_loss is None:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}")
        else:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_loss, val_loss
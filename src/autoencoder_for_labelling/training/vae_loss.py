import torch

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.01):
    """Compute VAE loss = MSE(recon, x) + beta * KLD.

    KLD is the KL divergence between N(mu, var) and N(0,1):
        KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Returns a tuple (total_loss, recon_loss, kld_loss) where each is a scalar tensor.
    """
    # reconstruction loss (MSE averaged over batch)
    mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')

    # KLD per batch (sum over latent dims, mean over batch)
    # standard formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld_element = 1 + logvar - mu.pow(2) - logvar.exp()
    # sum over latent dims
    kld_sum = -0.5 * torch.sum(kld_element, dim=1)
    # mean over batch
    kld_loss = torch.mean(kld_sum)

    total = mse_loss + beta * kld_loss
    return total, mse_loss, kld_loss

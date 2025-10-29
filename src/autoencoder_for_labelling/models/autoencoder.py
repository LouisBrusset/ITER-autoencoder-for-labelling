import torch.nn as nn
from typing import List, Optional


class SimpleAutoencoder(nn.Module):
    """Flexible autoencoder supporting variable encoder/decoder depths and widths.

    Parameters
    - input_dim: int, number of input features
    - encoding_dim: int, size of latent vector
    - encoder_layer_sizes: Optional[List[int]] intermediate layer sizes between input and latent
    - decoder_layer_sizes: Optional[List[int]] intermediate layer sizes between latent and output

    The class builds encoder: input -> encoder_layer_sizes... -> encoding_dim
    and decoder: encoding_dim -> decoder_layer_sizes... -> input_dim
    Activation: ReLU for hidden layers, Sigmoid on final output.
    """
    def __init__(self, input_dim: int = 20, encoding_dim: int = 8,
                 encoder_layer_sizes: Optional[List[int]] = None,
                 decoder_layer_sizes: Optional[List[int]] = None):
        super().__init__()
        # Normalize None to empty lists
        encoder_layer_sizes = encoder_layer_sizes or []
        decoder_layer_sizes = decoder_layer_sizes or []

        # Build encoder layers
        enc_layers = []
        prev = input_dim
        for sz in encoder_layer_sizes:
            enc_layers.append(nn.Linear(prev, int(sz)))
            enc_layers.append(nn.ReLU())
            prev = int(sz)
        # final mapping to encoding_dim
        enc_layers.append(nn.Linear(prev, encoding_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder layers: from encoding_dim back to input_dim
        dec_layers = []
        prev = encoding_dim
        for sz in decoder_layer_sizes:
            dec_layers.append(nn.Linear(prev, int(sz)))
            dec_layers.append(nn.ReLU())
            prev = int(sz)
        # final mapping to input_dim
        dec_layers.append(nn.Linear(prev, input_dim))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
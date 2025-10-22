import torch
import torch.nn as nn
import torch.optim as optim

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=20, encoding_dim=8):
        super(SimpleAutoencoder, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()  # Pour avoir une sortie entre 0 et 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
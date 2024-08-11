#!/usr/local/bin/python3

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def run_autoencoder(data):
    input_dim = data.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_tensor = torch.FloatTensor(data.values)

    epochs = 100
    losses = []

    for epoch in range(epochs):
        output = model(data_tensor)
        loss = criterion(output, data_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

    latent_features_path = '/var/www/html/skocsis2/LFOmics/data/processed/latent.csv'
    latent_features = model.encoder(data_tensor).detach().numpy()
    pd.DataFrame(latent_features).to_csv(latent_features_path, index=False)

    plot_loss_curve(losses)

    return latent_features_path

def plot_loss_curve(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/var/www/html/skocsis2/LFOmics/data/results/loss_curve.png')
    plt.close()

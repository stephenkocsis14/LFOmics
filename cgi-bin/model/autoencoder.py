#!/usr/local/bin/python3

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def run_autoencoder(df, hidden_dim=10):
    # Convert DataFrame to tensor
    input_data = torch.tensor(df.values, dtype=torch.float32)
    
    # Define the model
    input_dim = input_data.shape[1]
    model = SimpleAutoencoder(input_dim, hidden_dim)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Track losses for plotting
    losses = []
    
    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        encoded, decoded = model(input_data)
        loss = criterion(decoded, input_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store the loss value
        losses.append(loss.item())
    
    # Extract the latent features
    latent_features = encoded.detach().numpy()
    decoded_features = decoded.detach().numpy()
    
    # Convert to DataFrame
    latent_features_df = pd.DataFrame(latent_features, index=df.index, columns=[f'Latent_{i+1}' for i in range(hidden_dim)])
    decoded_features_df = pd.DataFrame(decoded_features, index=df.index, columns=df.columns)
    
    # Select top 1000 genes by variance across latent features
    top_1000_genes = latent_features_df.var(axis=1).nlargest(1000).index
    top_1000_df = latent_features_df.loc[top_1000_genes]
    
    # Map the top 1000 latent features back to their gene approximations
    gene_approximations = decoded_features_df.loc[top_1000_genes]
    
    # Save the results
    latent_features_path = '/var/www/html/skocsis2/LFOmics/results/latent_features_top_1000.csv'
    gene_approximations_path = '/var/www/html/skocsis2/LFOmics/results/gene_approximations_top_1000.csv'
    top_1000_df.to_csv(latent_features_path, index=True)
    gene_approximations.to_csv(gene_approximations_path, index=True)
    
    # Plot the training loss
    plt.figure()
    plt.plot(losses)
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    loss_plot_path = '/var/www/html/skocsis2/LFOmics/results/loss_curve.png'
    plt.savefig(loss_plot_path)
    plt.close()
    
    return latent_features_path, gene_approximations_path, loss_plot_path

#!/usr/local/bin/python3

import cgi
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
import mysql.connector

# Constants
UPLOAD_DIR = '../data/uploads/'
PROCESSED_DIR = '../data/processed/'
RESULTS_DIR = '../data/results/'

DB_HOST = 'localhost'
DB_USER = 'skocsis2'
DB_PASSWORD = 'spCk7725!'
DB_NAME = 'skocsis2'
SCHEMA_NAME = 'multiomics'

# Ensure the results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Save uploaded file
def save_uploaded_file(uploaded_file, upload_dir):
    try:
        file_name = os.path.basename(uploaded_file.filename)
        file_path = os.path.join(upload_dir, file_name)
        with open(file_path, 'wb') as output_file:
            output_file.write(uploaded_file.file.read())
        return file_path
    except Exception as e:
        print("Content-Type: text/html")
        print()
        print(f"<p>Error saving uploaded file: {str(e)}</p>")
        raise

# Load, preprocess, and save the input data
def process_input_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.set_index('GeneSymbol', inplace=True)
        processed_data = data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        processed_file_path = os.path.join(PROCESSED_DIR, 'processed_data.csv')
        processed_data.to_csv(processed_file_path)
        return processed_data
    except Exception as e:
        print("Content-Type: text/html")
        print()
        print(f"<p>Error processing input data: {str(e)}</p>")
        raise

# Define the autoencoder model
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

# Run the autoencoder model
def run_autoencoder(df, hidden_dim=10):
    try:
        input_data = torch.tensor(df.values, dtype=torch.float32)
        input_dim = input_data.shape[1]
        model = SimpleAutoencoder(input_dim, hidden_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = []
        num_epochs = 100
        for epoch in range(num_epochs):
            encoded, decoded = model(input_data)
            loss = criterion(decoded, input_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        latent_features = encoded.detach().numpy()
        decoded_features = decoded.detach().numpy()
        
        # Map the latent features and decoded features back to the GeneSymbol index
        latent_features_df = pd.DataFrame(latent_features, index=df.index, columns=[f'Latent_{i+1}' for i in range(hidden_dim)])
        decoded_features_df = pd.DataFrame(decoded_features, index=df.index, columns=df.columns)
        
        # Save the results to files
        latent_features_path = os.path.join(RESULTS_DIR, 'latent_features.csv')
        gene_approximations_path = os.path.join(RESULTS_DIR, 'decoded_features.csv')
        latent_features_df.to_csv(latent_features_path, index=True)
        decoded_features_df.to_csv(gene_approximations_path, index=True)
        
        # Plot the training loss
        plt.figure()
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        loss_plot_path = os.path.join(RESULTS_DIR, 'loss_curve.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        return latent_features_path, gene_approximations_path, loss_plot_path
    except Exception as e:
        print("Content-Type: text/html")
        print()
        print(f"<p>Error running autoencoder model: {str(e)}</p>")
        raise

# Run GSEApy analysis
def run_gseapy_analysis(gene_approximations_path):
    try:
        gene_approximations = pd.read_csv(gene_approximations_path, index_col=0)
        
        # Select the top contributing genes for GSEA (e.g., by variance)
        top_genes = gene_approximations.var(axis=1).nlargest(1000).index.tolist()
        
        # Perform GSEApy analysis
        gsea_results = gp.enrichr(
            gene_list=top_genes,
            gene_sets='KEGG_2019_Human',
            outdir=None  # No output files
        )

        # Save the results
        enrichment_results_path = os.path.join(RESULTS_DIR, 'enrichment_results.csv')
        gsea_results.res2d.to_csv(enrichment_results_path, index=False)
        
        # Plot the top 10 enrichment scores
        top_10_results = gsea_results.res2d.head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Combined Score', y='Term', data=top_10_results)
        plt.title('Top 10 Enrichment Scores')
        plt.xlabel('Combined Score')
        plt.ylabel('Gene Set')
        
        # Adjust plot to avoid cutting off labels
        plt.tight_layout()
        
        # Save the plot as an image
        plot_path = os.path.join(RESULTS_DIR, 'top_10_enrichment_scores.png')
        plt.savefig(plot_path)
        plt.close()

        return enrichment_results_path, plot_path
    except Exception as e:
        print("Content-Type: text/html")
        print()
        print(f"<p>Error running GSEApy analysis: {str(e)}</p>")
        raise

# Save results files to the database
def save_results_files_to_db(latent_features_path, gene_approximations_path, enrichment_results_path, loss_plot_path, plot_path):
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()

        cursor.execute(f"USE {SCHEMA_NAME};")

        # Insert records into the results_files table
        files_to_insert = [
            (latent_features_path, 'Latent Features CSV'),
            (gene_approximations_path, 'Decoded Features CSV'),
            (enrichment_results_path, 'GSEA Enrichment Results CSV'),
            (loss_plot_path, 'Training Loss Plot PNG'),
            (plot_path, 'Top 10 Enrichment Scores Plot PNG')
        ]

        for file_path, description in files_to_insert:
            cursor.execute("""
                INSERT INTO results_files (file_path, description) 
                VALUES (%s, %s)
            """, (file_path, description))

        conn.commit()
    except Exception as e:
        print("Content-Type: text/html")
        print()
        print(f"<p>Error saving files to the database: {str(e)}</p>")
        conn.rollback()
  finally:
        cursor.close()
        conn.close()

# Main script logic
def main():
    print("Content-Type: text/html")
    print()
    
    try:
        form = cgi.FieldStorage()
        uploaded_file = form['fileUpload']
        
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
        
        # Process the input data
        processed_data = process_input_data(file_path)
        
        # Run the autoencoder model
        latent_features_path, gene_approximations_path, loss_plot_path = run_autoencoder(processed_data)
        
        # Run GSEApy analysis
        enrichment_results_path, plot_path = run_gseapy_analysis(gene_approximations_path)
        
        # Save results files to the database
        save_results_files_to_db(latent_features_path, gene_approximations_path, enrichment_results_path, loss_plot_path, plot_path)
        
        # Redirect to the results page after successful processing
        print(f"""
        <html>
        <head>
            <meta http-equiv="refresh" content="0; url=../templates/results.html">
        </head>
        <body>
            <p>File uploaded, data processed, autoencoder training completed, GSEApy analysis done, and results saved successfully.</p>
        </body>
        </html>
        """)
        
    except Exception as e:
        print("Content-Type: text/html")
        print()
        print(f"<p>Unexpected error: {str(e)}</p>")

if __name__ == "__main__":
    main()

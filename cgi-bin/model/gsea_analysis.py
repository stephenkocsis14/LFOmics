#!/usr/local/bin/python3

import gseapy as gp
import pandas as pd
import matplotlib.pyplot as plt

def run_gseapy_analysis(latent_features_path):
    latent_features = pd.read_csv(latent_features_path)
    gene_list = latent_features.columns.tolist()

    enr = gp.enrichr(gene_list=gene_list, description='pathway', gene_sets='KEGG_2016')

    # Save results
    decoded_features_path = '/var/www/html/skocsis2/LFOmics/data/processed/decoded_top_latent.csv'
    enrichment_results_path = '/var/www/html/skocsis2/LFOmics/data/results/enrichment.csv'
    enr.results.to_csv(enrichment_results_path, index=False)
    pd.DataFrame(latent_features).to_csv(decoded_features_path, index=False)

    # Generate visualizations
    plot_enrichment_scores(enr.results)
    plot_p_value_distribution(enr.results)

    return decoded_features_path, enrichment_results_path

def plot_enrichment_scores(results):
    # Sort results by enrichment score and take top 10
    top_results = results.sort_values('Enrichment Score', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_results['Term'], top_results['Enrichment Score'], color='blue')
    plt.xlabel('Enrichment Score')
    plt.title('Top 10 Enriched KEGG Pathways')
    plt.gca().invert_yaxis()
    plt.savefig('/var/www/html/skocsis2/LFOmics/data/results/enrichment_scores.png')
    plt.close()

def plot_p_value_distribution(results):
    plt.figure(figsize=(10, 6))
    plt.hist(results['P-value'], bins=50, color='green', edgecolor='black')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('P-value Distribution for Enriched KEGG Pathways')
    plt.savefig('/var/www/html/skocsis2/LFOmics/data/results/p_value_distribution.png')
    plt.close()

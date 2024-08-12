#!/usr/local/bin/python3

import gseapy as gp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_gseapy_analysis(gene_approximations_path):
    # Load the gene approximations
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
    enrichment_results_path = '/var/www/html/skocsis2/LFOmics/results/enrichment_results.csv'
    gsea_results.res2d.to_csv(enrichment_results_path, index=False)
    
    # Plot the top 10 enrichment scores
    top_10_results = gsea_results.res2d.head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Combined Score', y='Term', data=top_10_results)
    plt.title('Top 10 Enrichment Scores')
    plt.xlabel('Combined Score')
    plt.ylabel('Gene Set')
    
    # Save the plot as an image
    plot_path = '/var/www/html/skocsis2/LFOmics/results/top_10_enrichment_scores.png'
    plt.savefig(plot_path)
    
    # Clear the plot to prevent overlap in case of subsequent plotting
    plt.clf()
    
    return enrichment_results_path, plot_path

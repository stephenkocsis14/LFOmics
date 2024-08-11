# LFOmics (Latent Feature Omics)

LFOmics is a powerful web tool designed for integrating multiomic data using autoencoder-based latent feature extraction and performing functional enrichment analysis with GSEApy enrichr on the top latent features. This tool facilitates advanced multiomic data analysis, enabling researchers to uncover hidden patterns and gain functional insights from complex biological datasets.

## Table of Contents

- [Introduction](#introduction)
- [Background](#background)
- [Features](#features)
- [Test Data Sources](#test-data-sources)
- [Data Preparation](#data-preparation)
- [Running the Tool](#running-the-tool)
- [Results](#results)
- [Clone Repository](#clone-repository)

## Introduction

Multiomic data integration is a crucial step in understanding complex biological systems. LFOmics leverages the power of autoencoders to extract latent features from multiomic datasets, providing a unified view of the data. The extracted features are then subjected to functional enrichment analysis using GSEApy enrichr to identify significantly enriched pathways and biological processes.

## Background

LFOmics was created to address the growing need for robust tools capable of integrating and analyzing multiomic data. Multiomics involves the simultaneous study of different types of 'omics' data (e.g., genomics, transcriptomics, proteomics) to gain a comprehensive view of biological systems. The complexity and high dimensionality of such data pose significant challenges for traditional analysis methods.

To overcome these challenges, LFOmics employs an autoencoder-based approach to extract latent features from multiomic datasets. Autoencoders are neural networks that compress data into a lower-dimensional space, capturing essential features while reducing noise. These latent features provide a more manageable representation of the data, enabling more effective downstream analysis.

Once the latent features are extracted, LFOmics leverages GSEApy Enrichr for functional enrichment analysis. Gene Set Enrichment Analysis (GSEA) is a statistical method used to determine whether a predefined set of genes shows statistically significant differences between two biological states. By applying GSEA to the latent features, LFOmics identifies key pathways and biological processes that are significantly enriched, providing valuable insights into the underlying biology.

LFOmics is designed to be user-friendly, with an intuitive web interface that allows researchers to upload their data, run analyses, and receive results with minimal effort. This tool is ideal for researchers looking to uncover hidden patterns in complex multiomic datasets and gain a deeper understanding of the biological systems they study.

## Features

- **Autoencoder-Based Latent Feature Extraction:** Efficiently extracts hidden non-linear patterns from multiomic data using a simple neural network.
- **GSEApy enrichr Functional Enrichment Analysis:** Performs functional enrichment analysis on top latent features to identify key biological pathways.
- **Interactive Web Interface:** User-friendly web interface for data upload, analysis, and getting results.
- **Support for Multiple Omic Types:** Handles diverse types of omic data, including those which gene symbols are features.
- **Intermediate Files Provided:** Provides CSV files for all of the intermediate matrices and functional enrichment results.

## Test Data Sources

LFOmics was developed and tested using publicly available RNA sequencing datasets:

- **Bulk RNA-seq Data**: The bulk RNA-seq data used in this project was sourced from the study titled "ESR1 and p53 Interactome Defines Mechanisms of Therapeutic Response to Tamoxifen Therapy in Luminal Breast Cancer Patients" under the accession number [GSE263089](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263089). This dataset provided valuable insights into gene expression patterns across different biological conditions.

- **Single-cell RNA-seq Data**: The single-cell RNA-seq (scRNA-seq) data was obtained from the study titled "Proteogenomic integration of single-cell RNA and protein analysis identifies novel tumour-infiltrating lymphocyte phenotypes in breast cancer" under the accession number [GSE199219](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE199219). This dataset allowed for the exploration of gene expression at the single-cell level, contributing to the understanding of cellular heterogeneity.

These datasets were instrumental in demonstrating the capabilities of LFOmics in integrating and analyzing complex multiomic data.

## Data Preperation

Before using LFOmics, please ensure that your data is correctly formatted:

- **CSV File Format**: Your data should be in a CSV file format, where:
  - **Rows represent gene symbols**: Each row should correspond to a unique gene, identified by its official gene symbol.
  - **Columns represent samples**: Each column should correspond to a sample, which could be from different conditions, time points, or replicates.

- **Combining Multiomics Data**: If you are integrating multiple types of omic data (e.g., transcriptomics, proteomics), ensure that these datasets are already combined into a single dataframe with gene symbols as the row identifiers. The columns should be consistent across all types of data, representing the same set of samples.

- **Example Format**:
  
  ```plaintext
  GENE_SYMBOL, Sample_1, Sample_2, Sample_3, ...
  BRCA1, 10.5, 9.8, 11.2, ...
  TP53, 12.3, 13.1, 14.0, ...
  ```

## Running the Tool

To use LFOmics, follow these simple steps:

### Step 1: Prepare Your Data

Ensure your data is formatted as a CSV file with gene symbols as rows and samples as columns. Refer to the [Data Preparation](#data-preparation) section for more details.

### Step 2: Access the LFOmics Interface

Connect to the JHU VPN and open your web browser and navigate to the following URL to access the LFOmics interface:

```plaintext
URL will go here
```

## Results

After submitting your data through the LFOmics interface, you will receive an email once the analysis is complete. The results will be available on the results page, where you can download the following files and view visualizations:

### Your Results

The analysis has been completed. You can download your results below:

- **Latent Feature Matrix**: A CSV file containing the latent features extracted by the autoencoder.

- **Decoded Latent Features**: A CSV file mapping the latent features back to their corresponding gene symbols.

- **Enrichment Analysis Results**: A CSV file with the results of the GSEApy Enrichr analysis, highlighting the top enriched pathways and biological processes.

### Visualizations

- **Model Training Loss**: An image showing the training loss curve of the autoencoder, which indicates how well the model performed during training.

- **Top Enrichment Scores**: A bar plot visualizing the top enriched pathways based on their enrichment scores.

- **P-value Distribution**: A histogram displaying the distribution of p-values across the enriched pathways, helping to assess the statistical significance of the results.

Each of these outputs is designed to provide you with a comprehensive view of the data analysis process, from the extraction of key features to the identification of significant biological pathways.

## Clone Repository
   ```bash
   git clone https://github.com/stephenkocsis14/LFOmics.git
   cd LFOmics
   ```


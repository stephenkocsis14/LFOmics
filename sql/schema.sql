-- Database: multiomics

CREATE DATABASE IF NOT EXISTS multiomics;
USE multiomics;

-- Table to store information about the uploaded input matrices
CREATE TABLE input_matrices (
    matrix_id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    num_genes INT,
    num_samples INT
);

-- Table to store the latent feature matrices
CREATE TABLE latent_features (
    feature_id INT AUTO_INCREMENT PRIMARY KEY,
    matrix_id INT,
    latent_feature_name VARCHAR(255),
    sample_id VARCHAR(255),
    feature_value FLOAT,
    FOREIGN KEY (matrix_id) REFERENCES input_matrices(matrix_id) ON DELETE CASCADE
);

-- Table to store the decoded latent features
CREATE TABLE decoded_latent_features (
    decoded_feature_id INT AUTO_INCREMENT PRIMARY KEY,
    feature_id INT,
    gene_symbol VARCHAR(255),
    sample_id VARCHAR(255),
    feature_value FLOAT,
    FOREIGN KEY (feature_id) REFERENCES latent_features(feature_id) ON DELETE CASCADE
);

-- Table to store the results of the GSEApy enrichment analysis
CREATE TABLE enrichment_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    matrix_id INT,
    gene_set VARCHAR(255),
    overlap INT,
    enrichment_score FLOAT,
    p_value FLOAT,
    q_value FLOAT,
    genes_in_set TEXT,
    FOREIGN KEY (matrix_id) REFERENCES input_matrices(matrix_id) ON DELETE CASCADE
);

-- Table to store sample information
CREATE TABLE samples (
    sample_id INT AUTO_INCREMENT PRIMARY KEY,
    matrix_id INT,
    sample_name VARCHAR(255),
    FOREIGN KEY (matrix_id) REFERENCES input_matrices(matrix_id) ON DELETE CASCADE
);

-- Table to store gene information
CREATE TABLE genes (
    gene_id INT AUTO_INCREMENT PRIMARY KEY,
    gene_symbol VARCHAR(255),
    gene_name VARCHAR(255)
);

-- Link table to associate latent features with genes
CREATE TABLE feature_gene_association (
    association_id INT AUTO_INCREMENT PRIMARY KEY,
    feature_id INT,
    gene_id INT,
    FOREIGN KEY (feature_id) REFERENCES latent_features(feature_id) ON DELETE CASCADE,
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id) ON DELETE CASCADE
);

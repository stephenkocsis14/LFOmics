#!/usr/local/bin/python3

import cgi
import os
import cgitb
from model.autoencoder import run_autoencoder
from model.gseapy_analysis import run_gseapy_analysis
from utils.data_processing import process_input_data
from utils.file_management import save_uploaded_file

cgitb.enable()

# Constants
UPLOAD_DIR = '../data/uploads/'

# Get form data
form = cgi.FieldStorage()
uploaded_file = form['fileUpload']
email = form.getvalue('email')

# Save uploaded file
file_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)

# Process input data
processed_data = process_input_data(file_path)

# Run the autoencoder model
latent_features_path, gene_approximations_path, loss_plot_path = run_autoencoder(processed_data)

# Run GSEApy analysis
enrichment_results_path, plot_path = run_gseapy_analysis(gene_approximations_path)

# Output results (redirect to the results page)
print("Content-Type: text/html")
print()
print(f"""
<html>
<head>
    <meta http-equiv="refresh" content="0; url=../templates/results.html">
</head>
<body>
    <p>Data processed successfully. You will be redirected to the results page shortly.</p>
</body>
</html>
""")

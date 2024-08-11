#!/usr/bin/env python3

import cgi
import os
import cgitb
import shutil
from db_connection import get_db_connection
from utils.data_processing import process_input_data
from model.autoencoder import run_autoencoder
from model.gseapy_analysis import run_gseapy_analysis
from utils.file_management import save_uploaded_file, get_file_path

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
latent_features_path = run_autoencoder(processed_data)

# Run GSEApy analysis
decoded_features_path, enrichment_results_path = run_gseapy_analysis(latent_features_path)

# Connect to the database and save results
conn = get_db_connection()
cursor = conn.cursor()
# Example: Insert data into the database (implement schema-specific logic)
# cursor.execute("INSERT INTO table_name ...")

conn.commit()
cursor.close()
conn.close()

# Output results
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

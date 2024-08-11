#!/usr/local/bin/python3

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

if 'fileUpload' not in form:
    print("Content-Type: text/html")
    print()
    print("<p>Error: No file was uploaded</p>")
    exit()

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

try:
    # Insert the uploaded file information into the uploads table
    cursor.execute("""
        INSERT INTO uploads (file_name, email) 
        VALUES (%s, %s)
    """, (uploaded_file.filename, email))
    
    # Get the ID of the last inserted row (the uploaded file)
    upload_id = cursor.lastrowid
    
    # Insert the paths of the generated results into the results table
    cursor.execute("""
        INSERT INTO results (upload_id, latent_features_path, decoded_features_path, enrichment_results_path) 
        VALUES (%s, %s, %s, %s)
    """, (upload_id, latent_features_path, decoded_features_path, enrichment_results_path))
    
    # Commit the transaction
    conn.commit()

except Exception as e:
    conn.rollback()
    print(f"<p>Error: {str(e)}</p>")

finally:
    cursor.close()
    conn.close()

# Output results
print("Content-Type: text/html")
print()
print(f"""
<html>
<head>
    <meta http-equiv="refresh" content="0; url=http://bfx3.aap.jhu.edu/skocsis2/LFOmics/templates/results.html">
</head>
<body>
    <p>Data processed successfully. You will be redirected to the results page shortly.</p>
</body>
</html>
""")

#!/usr/local/bin/python3

import os
import shutil

def save_uploaded_file(uploaded_file, upload_dir):
    file_name = os.path.basename(uploaded_file.filename)
    file_path = os.path.join(upload_dir, file_name)

    with open(file_path, 'wb') as output_file:
        shutil.copyfileobj(uploaded_file.file, output_file)
    
    return file_path

def get_file_path(file_name, base_dir):
    return os.path.join(base_dir, file_name)

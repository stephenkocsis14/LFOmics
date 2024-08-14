#!/usr/local/bin/python3

import sys
import os

# Add the path to your local PyTorch installation to sys.path
sys.path.append('/export/home/skocsis2/.local/lib/python3.7/site-packages')

print("Content-Type: text/html")
print()

try:
    import torch
    print(f"<p>Successfully imported torch: {torch.__version__}</p>")
except ImportError as e:
    print(f"<p>Error importing torch: {str(e)}</p>")

#!/usr/local/bin/python3

# Print the HTTP header
print("Content-Type: text/html")
print()

try:
    # Attempt to import gseapy
    import gseapy
    print("<p>GSEApy is successfully installed and can be imported!</p>")
except ImportError as e:
    print(f"<p>Error: GSEApy is not installed or cannot be imported. {str(e)}</p>")

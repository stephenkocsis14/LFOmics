#!/usr/local/bin/python3

import cgi
import cgitb
cgitb.enable(display=1)

print("Content-Type: text/html")
print()
print("<html><body><h1>Testing CGI</h1></body></html>")

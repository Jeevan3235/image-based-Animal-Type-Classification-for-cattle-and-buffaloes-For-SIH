#!/usr/bin/env python3
"""
Simple HTTP Server for SIH Prototype
Serves static files on port 8000
"""

import http.server
import socketserver
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler with logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def log_message(self, format, *args):
        """Log messages with our custom logger"""
        logger.info("%s - - [%s] %s",
                    self.address_string(),
                    self.log_date_time_string(),
                    format % args)
    
    def do_GET(self):
        """Handle GET requests"""
        # If path is '/' or empty, serve index.html
        if self.path == '/' or not self.path:
            self.path = '/index.html'
        
        return super().do_GET()

def run_server():
    """Run the HTTP server"""
    handler = CustomHTTPRequestHandler
    
    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        logger.info(f"Serving at http://localhost:{PORT}")
        logger.info(f"Serving directory: {DIRECTORY}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            httpd.server_close()
            logger.info("Server closed")

if __name__ == "__main__":
    logger.info(f"Starting HTTP server on port {PORT}...")
    run_server()

import argparse
import argparse
from flask_ml.flask_ml_cli import MLCli
from server import ml_server   # Import the ml_server instance from server.py

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CLI for Financial Anomaly Detection Server")
    
    # Initialize Flask-ML CLI
    cli = MLCli(ml_server, parser)  # Enable verbose mode for debugging
    
    # Run the CLI
    cli.run_cli()

if __name__ == "__main__":
    main()

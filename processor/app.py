#!/usr/bin/env python3
"""
Audio Cleaner Processor Flask App Entry Point

This script runs the processor with a Flask web interface for health checks.
"""

import os
import sys
import logging
from pathlib import Path

# Add the processor directory to the Python path
processor_dir = Path(__file__).parent
sys.path.insert(0, str(processor_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Flask app version"""
    # Import here to avoid circular imports
    from src.app import app, start_background_processor
    
    try:
        # Start the background processor
        start_background_processor()
        
        # Start the Flask app for health checks
        app.run(host='0.0.0.0', port=8080, debug=False)
    except Exception as e:
        logger.error(f"Error starting Flask app: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

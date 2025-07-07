#!/usr/bin/env python3
"""
Audio Cleaner Processor Entry Point

This script serves as the main entry point for the audio processing service.
It can run in different modes based on environment configuration.
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
    """Main entry point for the processor service"""
    # Import here to avoid circular imports
    from src.main import AudioCleanerProcessor
    import asyncio
    
    try:
        processor = AudioCleanerProcessor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(processor.process_messages())
    except Exception as e:
        logger.error(f"Error starting processor: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

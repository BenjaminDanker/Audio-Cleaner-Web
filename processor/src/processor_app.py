import asyncio
import logging
from flask import Flask, jsonify
from processor_main import AudioCleanerProcessor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
processor = None

@app.route('/')
def root():
    """Root endpoint"""
    return jsonify({
        "service": "Audio Cleaner Processor", 
        "status": "running"
    })

def run_processor():
    """Run the message processor in a separate thread"""
    global processor
    try:
        processor = AudioCleanerProcessor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(processor.process_messages())
    except Exception as e:
        logger.error(f"Error in processor thread: {e}")

def start_background_processor():
    """Start the processor in a background thread"""
    processor_thread = threading.Thread(target=run_processor, daemon=True)
    processor_thread.start()
    logger.info("Background processor started")

if __name__ == '__main__':
    # Start the background processor
    start_background_processor()
    
    # Start the Flask app for health checks
    app.run(host='0.0.0.0', port=8080, debug=False)

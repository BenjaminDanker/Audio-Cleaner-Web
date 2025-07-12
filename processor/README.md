# Audio Processor Service

This directory contains the Python-based audio processing service that handles video denoising using AI models.

## Directory Structure

```
processor/
├── src/                        # Source code
│   ├── __init__.py            # Package initialization
│   ├── processor_app.py       # Flask app wrapper for health checks
│   ├── processor_main.py      # Main Service Bus processor
│   └── video_handler.py       # Core video processing logic
└── models/                    # AI Models
    └── DeepFilterNet3/        # DeepFilterNet3 model files
        ├── config.ini
        ├── df_dec.onnx
        ├── enc.onnx
        ├── erb_dec.onnx
        └── checkpoints/
```

## Entry Points

### For Azure Container Apps (Production)
```bash
python processor/src/processor_main.py
```
This starts the Service Bus message processor that listens for video processing jobs.

### For Health Check Web Interface
```bash
python processor/src/processor_app.py
```
This starts a Flask web server with health checks AND the background processor.

## Components

### `src/processor_main.py` - AudioCleanerProcessor
- Connects to Azure Service Bus for job queue
- Processes video files using DeepFilterNet3
- Updates job status in Cosmos DB
- Manages Azure Blob Storage for input/output files

### `src/video_handler.py` - VideoProcessor
- Core video processing logic
- Audio extraction and enhancement using DeepFilterNet3
- Video remuxing with enhanced audio
- Temporary file management

### `src/processor_app.py` - Flask Health Check App
- Provides HTTP endpoints for health monitoring
- Runs the processor in a background thread
- Used for container health checks

## Dependencies

The processor requires the following environment variables:
- `AZURE_SERVICE_BUS_CONNECTION_STRING` - Service Bus connection
- `AZURE_STORAGE_CONNECTION_STRING` - Blob storage connection  
- `COSMOS_CONNECTION_STRING` - Cosmos DB connection
- `USE_MANAGED_IDENTITY` - Set to 'true' for Azure deployment

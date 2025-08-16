# Processor Directory Structure

This directory has been reorganized to better separate the different processing pipelines:

## Directory Structure

```
processor/
├── batch/                    # File denoising/batch processing pipeline
│   ├── __init__.py
│   ├── processor_main.py     # Service Bus consumer and batch job processor
│   ├── media_processor.py    # Media processing pipeline orchestrator
│   └── media_extractor.py    # Media extraction utilities
├── shared/                   # Common components used by both pipelines
│   ├── __init__.py
│   ├── ai/                   # AI processing pipelines (ASR, audio enhancement)
│   ├── captions/             # Caption encoding utilities
│   ├── config.py             # Configuration and Application Insights setup
│   ├── job_models.py         # Data models for jobs
│   ├── job_store.py          # Cosmos DB job storage operations
│   ├── pricing.py            # Pricing calculation utilities
│   └── storage_service.py    # Azure Blob Storage operations
├── streaming/                # Real-time streaming pipeline
│   ├── __init__.py
│   ├── app.py               # FastAPI WebSocket streaming service
│   ├── main.py              # Streaming service entrypoint
│   ├── README.md
│   └── requirements.txt
├── models/                  # AI model files (DeepFilterNet3, etc.)
├── tests/                   # Test files
├── Dockerfile.batch         # Batch processor container build configuration
├── Dockerfile.streaming     # Streaming service container build configuration
├── docker-compose.dev.yml   # Development docker compose
├── build-containers.ps1     # Build and push script for both containers
└── requirements.txt         # Python dependencies
```

## How Streaming Works

### Architecture Overview

1. **Two Separate Container Apps**:
   - **Batch Processor**: Handles file uploads via Service Bus queue
   - **Streaming Service**: Handles real-time WebSocket connections

2. **Authentication Flow**:
   ```
   OBS Plugin/Client
   ↓ POST /api/create-stream-session (with API key)
   API Functions
   ↓ Returns signed token + WebSocket URL
   Client
   ↓ Connects to WebSocket with token
   Streaming Container App
   ```

3. **Container Scaling**:
   - **Batch**: Scales 0-10 based on Service Bus queue length (KEDA)
   - **Streaming**: Scales based on replica configuration (min/max replicas)

### Deployment Process

#### 1. Build and Push Container Images
```powershell
# Build both batch and streaming containers
./build-containers.ps1 -Registry "your-acr.azurecr.io" -Tag "latest"
```

#### 2. Configure Terraform Variables
```hcl
# In terraform.tfvars
stream_session_signing_key = "your-secure-random-key-here"
streaming_min_replicas     = 1
streaming_max_replicas     = 5
```

#### 3. Deploy Infrastructure
```bash
terraform apply
```

### Environment Variables

#### Required for API Functions
- `STREAM_SESSION_SIGNING_KEY` - HMAC key for session tokens
- `STREAMING_ENDPOINT` - URL of the streaming container app
- `STREAMING_API_KEYS` - Comma-separated API keys for OBS clients

#### Required for Streaming Container
- `STREAM_SESSION_SIGNING_KEY` - Same key as API (for token verification)
- `PROCESSOR_MODE=stream` - Ensures streaming-only mode
- `AZURE_OPENAI_*` - OpenAI configuration for ASR
- `COSMOS_CONNECTION_STRING` - For billing operations

#### Required for Batch Container
- `PROCESSOR_MODE=batch` - Ensures batch-only mode
- `AZURE_SERVICE_BUS_CONNECTION_STRING` - For job queue
- `AZURE_STORAGE_CONNECTION_STRING` - For file processing

### Usage

#### From OBS Plugin
1. Call `POST /api/create-stream-session` with API key header
2. Receive response with `wsUrl` and `token`
3. Connect WebSocket to `wsUrl`
4. Send audio chunks as binary data
5. Receive real-time subtitle deltas

#### Response Format
```json
{
  "sessionId": "uuid-here",
  "wsUrl": "wss://streaming-app.azurecontainerapps.io/stream/uuid?t=token",
  "token": "signed.token.here",
  "expiresInMinutes": 30,
  "languagesRequested": ["en"]
}
```

### Development

#### Local Development
```bash
# Terminal 1: Start streaming service
cd streaming
python main.py

# Terminal 2: Start batch processor
cd batch
python processor_main.py

# Use docker-compose for full integration testing
docker-compose -f docker-compose.dev.yml up
```

#### Container Testing
```bash
# Test streaming container
docker build -f Dockerfile.streaming -t streaming-test .
docker run -p 8000:8000 streaming-test

# Test batch container  
docker build -f Dockerfile -t batch-test .
docker run batch-test
```

## Import Structure

- **Batch files** (`batch/`) import shared components from `../shared/`
- **Streaming files** (`streaming/`) import shared components from `../shared/`
- **Shared components** (`shared/`) contain reusable code for both pipelines

## Security Notes

- Session tokens are HMAC-signed with expiration
- API key authentication required for session creation via create-stream-session
- WebSocket connections validate HMAC tokens only (API key validation happens at session creation)
- Streaming container only processes audio, no file persistence

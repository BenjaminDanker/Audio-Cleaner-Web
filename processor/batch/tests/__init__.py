"""Test configuration for batch processing tests."""
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
test_dir = Path(__file__).parent
batch_dir = test_dir.parent
processor_dir = batch_dir.parent
shared_dir = processor_dir / "shared"

for dir_path in [batch_dir, shared_dir]:
    if str(dir_path) not in sys.path:
        sys.path.insert(0, str(dir_path))

# Mock environment variables for testing
os.environ.update({
    # Service Bus and Cosmos (will be mocked)
    "AZURE_SERVICE_BUS_CONNECTION_STRING": "Endpoint=sb://test.servicebus.windows.net/;SharedAccessKeyName=test;SharedAccessKey=fake",
    "AZURE_COSMOS_CONNECTION_STRING": "AccountEndpoint=https://test.documents.azure.com:443/;AccountKey=fake",
    "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake;EndpointSuffix=core.windows.net",
    
    # Processing configuration
    "PROCESSOR_MODE": "batch",
    "PROCESSING_TEMP_DIR": str(test_dir / "temp"),
    "FFMPEG_TIMEOUT_S": "30",  # Shorter timeout for tests
    
    # Audio processing
    "DEEPFILTER_ENABLE": "true",
    "AUDIO_ENHANCEMENT_ENABLED": "true",
    
    # Billing/pricing (will be mocked)
    "BASE_CENTS_PER_MINUTE": "10",
    "EXTRA_LANG_CENTS_PER_MINUTE": "5",
    
    # Disable real Azure services for testing
    "AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING": "",
})

# Create temp directory for tests
temp_dir = Path(os.environ["PROCESSING_TEMP_DIR"])
temp_dir.mkdir(exist_ok=True)

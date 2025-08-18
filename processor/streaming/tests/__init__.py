"""Test configuration and shared fixtures."""
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
test_dir = Path(__file__).parent
streaming_dir = test_dir.parent
processor_dir = streaming_dir.parent
shared_dir = processor_dir / "shared"

for dir_path in [streaming_dir, shared_dir]:
    if str(dir_path) not in sys.path:
        sys.path.insert(0, str(dir_path))

# Mock environment variables for testing
os.environ.update({
    "STREAM_SESSION_SIGNING_KEY": "test-signing-key-for-unit-tests-only-not-production",
    "STREAM_ASR_BUFFER_SECONDS": "6",
    "STREAM_ASR_STRIDE_SECONDS": "2", 
    "STREAM_BASE_CENTS_PER_MINUTE": "10",
    "STREAM_EXTRA_LANG_CENTS_PER_MINUTE": "5",
    "STREAM_LOW_CREDITS_GRACE_SECONDS": "8",
    "STREAM_ACCUMULATE_CAPTIONS": "0",
    "UVICORN_HOST": "0.0.0.0",
    "UVICORN_PORT": "8000",
    "UVICORN_LOG_LEVEL": "info"
})

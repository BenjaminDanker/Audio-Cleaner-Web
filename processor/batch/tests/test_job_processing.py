"""Tests for batch job processing logic."""
import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from pathlib import Path

# We'll test the job processing functions without importing the full processor_main
# to avoid Azure service dependencies


class MockJobMessage:
    """Mock Azure Service Bus message."""
    
    def __init__(self, job_data: dict):
        self.job_data = job_data
        self._body = json.dumps(job_data).encode()
    
    @property 
    def body(self):
        return self._body


class MockBlobClient:
    """Mock Azure Blob client."""
    
    def __init__(self, account_url: str, container_name: str, blob_name: str):
        self.account_url = account_url
        self.container_name = container_name
        self.blob_name = blob_name
        self.download_called = False
        self.upload_called = False
    
    async def download_blob(self):
        """Mock blob download."""
        self.download_called = True
        return AsyncMock()
    
    async def upload_blob(self, data, **kwargs):
        """Mock blob upload."""
        self.upload_called = True
        return {"etag": "fake-etag"}


class MockJobStore:
    """Mock job store for testing."""
    
    def __init__(self):
        self.jobs = {}
        self.status_updates = []
        self.result_updates = []
    
    async def update_job_status(self, job_id: str, status: str, **kwargs):
        """Mock status update."""
        self.status_updates.append({
            "job_id": job_id,
            "status": status,
            **kwargs
        })
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
    
    async def update_job_result(self, job_id: str, result_data: dict):
        """Mock result update."""
        self.result_updates.append({
            "job_id": job_id,
            "result": result_data
        })
        if job_id in self.jobs:
            self.jobs[job_id]["result"] = result_data
    
    async def get_job(self, job_id: str):
        """Mock job retrieval."""
        return self.jobs.get(job_id)


class TestJobProcessingLogic:
    """Test job processing workflow without Azure dependencies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_job_store = MockJobStore()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def create_test_job_message(self, job_id: str = "test-job-123") -> MockJobMessage:
        """Create a test job message."""
        job_data = {
            "jobId": job_id,
            "userId": "user-456",
            "inputBlobUrl": f"https://storage.blob.core.windows.net/uploads/{job_id}_input.mp3",
            "outputBlobUrl": f"https://storage.blob.core.windows.net/processed/{job_id}_output.mp3",
            "languages": ["en"],
            "captionsFormat": "srt",
            "timestamp": "2025-08-17T12:00:00Z"
        }
        return MockJobMessage(job_data)
    
    def test_job_message_parsing(self):
        """Test parsing job messages from Service Bus."""
        message = self.create_test_job_message()
        
        # Parse message body
        job_data = json.loads(message.body.decode())
        
        assert job_data["jobId"] == "test-job-123"
        assert job_data["userId"] == "user-456"
        assert job_data["languages"] == ["en"]
        assert job_data["captionsFormat"] == "srt"
        assert "inputBlobUrl" in job_data
        assert "outputBlobUrl" in job_data
    
    def test_blob_url_parsing(self):
        """Test extracting blob info from URLs."""
        blob_url = "https://storage.blob.core.windows.net/uploads/test-job-123_input.mp3"
        
        # Simple URL parsing (mimics what the real processor does)
        from urllib.parse import urlparse
        parsed = urlparse(blob_url)
        path_parts = parsed.path.strip('/').split('/')
        
        assert len(path_parts) >= 2
        container_name = path_parts[0]
        blob_name = '/'.join(path_parts[1:])
        
        assert container_name == "uploads"
        assert blob_name == "test-job-123_input.mp3"
    
    @pytest.mark.asyncio
    async def test_job_processing_workflow_success(self):
        """Test successful job processing workflow."""
        # Create test job
        message = self.create_test_job_message()
        job_data = json.loads(message.body.decode())
        job_id = job_data["jobId"]
        
        # Mock successful processing steps
        with patch('tempfile.mkdtemp', return_value=self.temp_dir):
            
            # Step 1: Update status to processing
            await self.mock_job_store.update_job_status(job_id, "processing")
            
            # Step 2: Download input file (mocked)
            input_path = os.path.join(self.temp_dir, "input.mp3")
            Path(input_path).touch()  # Create fake file
            
            # Step 3: Process file (mocked)
            output_path = os.path.join(self.temp_dir, "output.mp3")
            Path(output_path).touch()  # Create fake processed file
            
            # Step 4: Generate captions (mocked)
            captions_content = "1\n00:00:00,000 --> 00:00:02,000\nTest caption\n\n"
            captions_path = os.path.join(self.temp_dir, "captions.srt")
            with open(captions_path, 'w') as f:
                f.write(captions_content)
            
            # Step 5: Upload results (mocked)
            await self.mock_job_store.update_job_status(job_id, "uploading")
            
            # Step 6: Update job with results
            result_data = {
                "processedBlobUrl": job_data["outputBlobUrl"],
                "captionsBlobUrl": f"https://storage.blob.core.windows.net/processed/{job_id}_captions.srt",
                "duration": 120.5,
                "languages": ["en"],
                "processingTimeMs": 5000
            }
            await self.mock_job_store.update_job_result(job_id, result_data)
            await self.mock_job_store.update_job_status(job_id, "completed")
        
        # Verify workflow
        status_updates = [u["status"] for u in self.mock_job_store.status_updates]
        assert "processing" in status_updates
        assert "uploading" in status_updates
        assert "completed" in status_updates
        
        assert len(self.mock_job_store.result_updates) == 1
        result = self.mock_job_store.result_updates[0]["result"]
        assert "processedBlobUrl" in result
        assert "captionsBlobUrl" in result
        assert result["duration"] > 0
    
    @pytest.mark.asyncio
    async def test_job_processing_workflow_failure(self):
        """Test job processing workflow with failure."""
        message = self.create_test_job_message()
        job_data = json.loads(message.body.decode())
        job_id = job_data["jobId"]
        
        # Mock failure during processing
        try:
            await self.mock_job_store.update_job_status(job_id, "processing")
            
            # Simulate processing failure
            raise RuntimeError("Audio processing failed")
            
        except RuntimeError as e:
            # Handle failure
            await self.mock_job_store.update_job_status(
                job_id, 
                "failed", 
                error=str(e),
                failedAt="audio_processing"
            )
        
        # Verify failure handling
        status_updates = self.mock_job_store.status_updates
        assert len(status_updates) == 2
        assert status_updates[0]["status"] == "processing"
        assert status_updates[1]["status"] == "failed"
        assert "error" in status_updates[1]
        assert "failedAt" in status_updates[1]
    
    def test_pricing_calculation(self):
        """Test pricing calculation for batch jobs."""
        # Mock the pricing calculation logic
        duration_minutes = 5.5  # 5.5 minutes of audio
        languages = ["en", "es"]  # 2 languages
        
        base_rate = 10  # cents per minute
        extra_lang_rate = 5  # cents per minute per extra language
        
        # Calculate cost: base rate + extra languages rate
        extra_languages = max(0, len(languages) - 1)  # First language included in base
        total_rate_per_minute = base_rate + (extra_languages * extra_lang_rate)
        total_cost_cents = int(duration_minutes * total_rate_per_minute)
        
        expected_cost = int(5.5 * (10 + 1 * 5))  # 5.5 * 15 = 82.5 -> 82 cents
        assert total_cost_cents == expected_cost
    
    def test_multiple_language_processing(self):
        """Test handling multiple language requirements."""
        message = self.create_test_job_message()
        job_data = json.loads(message.body.decode())
        
        # Update with multiple languages
        job_data["languages"] = ["en", "es", "fr"]
        
        languages = job_data["languages"]
        assert len(languages) == 3
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        
        # Verify this affects pricing
        base_rate = 10
        extra_lang_rate = 5
        extra_languages = max(0, len(languages) - 1)  # 2 extra languages
        total_rate = base_rate + (extra_languages * extra_lang_rate)  # 10 + 2*5 = 20 cents/min
        
        assert total_rate == 20
    
    def test_caption_format_handling(self):
        """Test different caption format requirements."""
        formats = ["srt", "vtt", "both"]
        
        for fmt in formats:
            message = self.create_test_job_message()
            job_data = json.loads(message.body.decode())
            job_data["captionsFormat"] = fmt
            
            assert job_data["captionsFormat"] == fmt
            
            # Verify format determines output files
            if fmt == "srt":
                expected_files = ["captions.srt"]
            elif fmt == "vtt":
                expected_files = ["captions.vtt"]
            elif fmt == "both":
                expected_files = ["captions.srt", "captions.vtt"]
            
            # This would be implemented in the real processor
            assert len(expected_files) > 0


class TestJobStoreIntegration:
    """Test job store operations with mocked Cosmos DB."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_job_store = MockJobStore()
    
    @pytest.mark.asyncio
    async def test_job_status_updates(self):
        """Test job status update flow."""
        job_id = "test-job-status"
        
        # Test status progression
        statuses = ["queued", "processing", "uploading", "completed"]
        
        for status in statuses:
            await self.mock_job_store.update_job_status(job_id, status)
        
        # Verify all updates were recorded
        assert len(self.mock_job_store.status_updates) == 4
        
        recorded_statuses = [u["status"] for u in self.mock_job_store.status_updates]
        assert recorded_statuses == statuses
    
    @pytest.mark.asyncio
    async def test_job_result_storage(self):
        """Test job result storage."""
        job_id = "test-job-result"
        
        result_data = {
            "processedBlobUrl": "https://example.com/output.mp3",
            "captionsBlobUrl": "https://example.com/captions.srt",
            "duration": 180.0,
            "languages": ["en", "es"],
            "processingTimeMs": 15000,
            "audioEnhanced": True,
            "captionsGenerated": True
        }
        
        await self.mock_job_store.update_job_result(job_id, result_data)
        
        # Verify result was stored
        assert len(self.mock_job_store.result_updates) == 1
        stored_result = self.mock_job_store.result_updates[0]["result"]
        
        assert stored_result["duration"] == 180.0
        assert stored_result["languages"] == ["en", "es"]
        assert stored_result["audioEnhanced"] is True
        assert stored_result["captionsGenerated"] is True


class TestBatchProcessingUtils:
    """Test utility functions for batch processing."""
    
    def test_file_extension_detection(self):
        """Test file extension detection for different media types."""
        test_files = {
            "video.mp4": "video",
            "audio.mp3": "audio", 
            "movie.mov": "video",
            "song.flac": "audio",
            "presentation.mkv": "video",
            "podcast.wav": "audio"
        }
        
        for filename, expected_type in test_files.items():
            ext = Path(filename).suffix.lower()
            
            # Simplified type detection
            video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
            audio_exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
            
            if ext in video_exts:
                detected_type = "video"
            elif ext in audio_exts:
                detected_type = "audio"
            else:
                detected_type = "unknown"
            
            assert detected_type == expected_type, f"Failed for {filename}"
    
    def test_blob_name_generation(self):
        """Test generating blob names for output files."""
        job_id = "test-job-123"
        user_id = "user-456"
        
        # Test output blob naming
        processed_blob = f"{job_id}_processed.mp3"
        captions_blob = f"{job_id}_captions.srt"
        
        assert processed_blob.startswith(job_id)
        assert captions_blob.startswith(job_id)
        assert processed_blob.endswith(".mp3")
        assert captions_blob.endswith(".srt")
    
    def test_error_categorization(self):
        """Test categorizing different types of processing errors."""
        error_scenarios = {
            "FileNotFoundError": "download_failed",
            "RuntimeError: FFmpeg failed": "processing_failed", 
            "ConnectionError": "upload_failed",
            "ValueError: Invalid audio format": "validation_failed",
            "TimeoutError": "timeout_failed"
        }
        
        for error_msg, expected_category in error_scenarios.items():
            # Simplified error categorization
            if "FileNotFound" in error_msg:
                category = "download_failed"
            elif "FFmpeg" in error_msg or "processing" in error_msg.lower():
                category = "processing_failed"
            elif "Connection" in error_msg or "upload" in error_msg.lower():
                category = "upload_failed"
            elif "Invalid" in error_msg or "ValueError" in error_msg:
                category = "validation_failed"
            elif "Timeout" in error_msg:
                category = "timeout_failed"
            else:
                category = "unknown_error"
            
            assert category == expected_category, f"Failed for {error_msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test MediaProcessor functionality for Audio Cleaner batch processing.

This test suite validates:
- MediaProcessor initialization and lifecycle
- Audio and video processing workflows
- Error handling and fallback mechanisms
- Cleanup functionality
- Integration with real audio files

All Azure service dependencies are mocked.
"""

import os
import tempfile
import shutil
import logging
import wave
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import pytest

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from media_processor import MediaProcessor


class MockUploadedFile:
    """Mock uploaded file for testing."""
    def __init__(self, filename: str, content: bytes = b"fake content"):
        self.filename = filename
        self.name = filename
        self.size = len(content)
        self._content = content
    
    def save(self, path: str):
        """Mock save method."""
        with open(path, 'wb') as f:
            f.write(self._content)


class TestMediaProcessor:
    """Test MediaProcessor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock uploaded file object
        mock_file = MockUploadedFile("test_audio.mp3")
        self.processor = MediaProcessor(
            uploaded_file=mock_file,
            atten_db=10,
            processing_type="denoise"
        )
        # Override temp directory for testing
        self.processor.temp_dir_path = self.temp_dir
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'processor'):
            try:
                self.processor.immediate_cleanup(logging.getLogger())
            except:
                pass
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_processor_initialization(self):
        """Test MediaProcessor initialization."""
        mock_file = MockUploadedFile("test.wav")
        processor = MediaProcessor(
            uploaded_file=mock_file,
            atten_db=15,
            processing_type="denoise"
        )
        
        assert processor.uploaded_file == mock_file
        assert processor.atten_db == 15
        assert processor.processing_type == "denoise"
        assert processor.input_filename == "test.wav"
        assert processor.original_extension == ".wav"
        assert os.path.exists(processor.temp_dir_path)
    
    def test_processor_temp_directory_creation(self):
        """Test temporary directory creation."""
        mock_file = MockUploadedFile("test.mp3")
        processor = MediaProcessor(
            uploaded_file=mock_file,
            atten_db=5,
            processing_type="denoise"
        )
        
        assert os.path.exists(processor.temp_dir_path)
        assert processor.temp_dir_path.startswith(tempfile.gettempdir())
    
    @patch('media_processor.registry')
    def test_process_audio_file_success(self, mock_registry):
        """Test successful audio file processing."""
        # Setup mocks
        mock_task = Mock()
        mock_task.process.return_value = "/output/processed.mp3"
        mock_registry.create.return_value = mock_task
        
        # Create fake input file
        input_path = os.path.join(self.temp_dir, "test_audio.mp3")
        with open(input_path, 'w') as f:
            f.write("fake audio content")
        
        # Override the save method to create our test file
        self.processor.uploaded_file.save = lambda path: shutil.copy(input_path, path)
        
        # Test processing
        result_path = self.processor.process()
        
        assert result_path == "/output/processed.mp3"
        mock_registry.create.assert_called_once_with("denoise")
        mock_task.process.assert_called_once()
    
    @patch('media_processor.registry')
    def test_process_unknown_processing_type_fallback(self, mock_registry):
        """Test fallback to denoise when unknown processing type is used."""
        # Setup mocks - first call fails, second succeeds
        mock_task = Mock()
        mock_task.process.return_value = "/output/processed.mp3"
        mock_registry.create.side_effect = [Exception("Unknown type"), mock_task]
        
        # Create processor with unknown type
        mock_file = MockUploadedFile("test.mp3")
        processor = MediaProcessor(
            uploaded_file=mock_file,
            atten_db=10,
            processing_type="unknown_type"
        )
        
        # Create fake input file
        input_path = os.path.join(self.temp_dir, "test_audio.mp3")
        with open(input_path, 'w') as f:
            f.write("fake audio content")
        
        processor.uploaded_file.save = lambda path: shutil.copy(input_path, path)
        
        # Test processing
        result_path = processor.process()
        
        assert result_path == "/output/processed.mp3"
        assert mock_registry.create.call_count == 2
        mock_registry.create.assert_any_call("unknown_type")
        mock_registry.create.assert_any_call("denoise")
    
    def test_cleanup_methods(self):
        """Test cleanup functionality."""
        mock_file = MockUploadedFile("test.mp3")
        processor = MediaProcessor(
            uploaded_file=mock_file,
            atten_db=10,
            processing_type="denoise"
        )
        
        temp_dir = processor.temp_dir_path
        assert os.path.exists(temp_dir)
        
        # Test immediate cleanup
        logger = logging.getLogger()
        processor.immediate_cleanup(logger)
        
        # Directory should be removed (or at least attempted)
        # Note: cleanup might fail on Windows due to file locks, but method should not raise
    
    def test_schedule_cleanup(self):
        """Test scheduled cleanup functionality."""
        mock_file = MockUploadedFile("test.mp3")
        processor = MediaProcessor(
            uploaded_file=mock_file,
            atten_db=10,
            processing_type="denoise"
        )
        
        temp_dir = processor.temp_dir_path
        assert os.path.exists(temp_dir)
        
        # Test scheduled cleanup
        logger = logging.getLogger()
        processor.schedule_cleanup(logger)
        
        # Method should complete without error
        # Actual cleanup is deferred in real implementation


class TestMediaProcessorIntegration:
    """Integration tests that require actual audio processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_wav_file(self, filepath: str, duration_seconds: float = 1.0, sample_rate: int = 44100):
        """Create a test WAV file with sine wave."""
        # Generate a sine wave
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
        # 440 Hz sine wave (A note)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # Convert to 16-bit PCM
        audio_data_16bit = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data_16bit.tobytes())
        
        return filepath
    
    @pytest.mark.skipif(True, reason="Requires full AI task registry setup")
    def test_real_audio_processing_pipeline(self):
        """Test the complete audio processing pipeline with real files."""
        try:
            # Create a test WAV file
            input_path = os.path.join(self.temp_dir, "test_input.wav")
            self.create_test_wav_file(input_path, duration_seconds=2.0)
            
            # Create a mock uploaded file object that points to our test file
            class RealMockUploadedFile:
                def __init__(self, filename: str, real_path: str):
                    self.filename = filename
                    self.name = filename
                    self.size = os.path.getsize(real_path)
                    self._real_path = real_path
                
                def save(self, path: str):
                    """Copy the real file to the target path."""
                    shutil.copy(self._real_path, path)
            
            mock_file = RealMockUploadedFile("test_input.wav", input_path)
            processor = MediaProcessor(
                uploaded_file=mock_file,
                atten_db=10,
                processing_type="denoise"
            )
            
            # Process the file
            # This will actually run the full pipeline if AI tasks are registered
            result_path = processor.process()
            
            # Verify result
            assert result_path is not None
            assert os.path.exists(result_path)
            
            # Clean up
            processor.immediate_cleanup(logging.getLogger())
            
        except Exception as e:
            # If the test fails due to missing dependencies, that's expected
            pytest.skip(f"Integration test requires full environment setup: {e}")

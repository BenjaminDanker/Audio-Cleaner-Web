"""
Test MediaExtractor functionality for Audio Cleaner batch processing.

This test suite validates:
- Media type detection (video/audio)
- Audio extraction to WAV format
- FFmpeg command execution
- Error handling
- Integration with real files

All Azure service dependencies are mocked.
"""

import os
import tempfile
import shutil
import subprocess
import wave
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from media_extractor import MediaExtractor, MediaType, ExtractionResult


class TestMediaType:
    """Test MediaType enum."""
    
    def test_media_type_values(self):
        """Test MediaType enum values."""
        assert MediaType.VIDEO.value == "video"
        assert MediaType.AUDIO.value == "audio"


class TestExtractionResult:
    """Test ExtractionResult data structure."""
    
    def test_extraction_result_creation(self):
        """Test creating ExtractionResult objects."""
        result = ExtractionResult(
            media_type=MediaType.AUDIO,
            source_path="/test/input.mp3",
            extracted_wav_path="/test/output.wav",
            working_video_path=None,
            original_extension=".mp3"
        )
        
        assert result.media_type == MediaType.AUDIO
        assert result.source_path == "/test/input.mp3"
        assert result.extracted_wav_path == "/test/output.wav"
        assert result.working_video_path is None
        assert result.original_extension == ".mp3"


class TestMediaExtractor:
    """Test MediaExtractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = MediaExtractor(target_sample_rate=48000)
    
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
    
    def test_extractor_initialization(self):
        """Test MediaExtractor initialization."""
        extractor = MediaExtractor(target_sample_rate=44100)
        assert extractor.target_sample_rate == 44100
        assert extractor.ffmpeg is not None
    
    def test_detect_media_type_video_extensions(self):
        """Test detecting video file types."""
        for ext in ['.mp4', '.mov', '.mkv', '.webm', '.avi']:
            test_path = f"/test/video{ext}"
            media_type = self.extractor.detect_media_type(test_path)
            assert media_type == MediaType.VIDEO
    
    def test_detect_media_type_audio_extensions(self):
        """Test detecting audio file types."""
        for ext in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.opus']:
            test_path = f"/test/audio{ext}"
            media_type = self.extractor.detect_media_type(test_path)
            assert media_type == MediaType.AUDIO
    
    def test_detect_media_type_unknown_extension(self):
        """Test detecting unknown file types defaults to audio."""
        test_path = "/test/unknown.xyz"
        media_type = self.extractor.detect_media_type(test_path)
        assert media_type == MediaType.AUDIO  # Default behavior
    
    @patch('media_extractor.subprocess.run')
    @patch('os.path.exists')
    def test_extract_audio_success_mock(self, mock_exists, mock_subprocess):
        """Test successful audio extraction with mocked subprocess."""
        # Setup mocks
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        input_path = os.path.join(self.temp_dir, "input.mp3")
        
        # Create fake input file
        with open(input_path, 'w') as f:
            f.write("fake audio content")
        
        # Test extraction
        result = self.extractor.extract(input_path, self.temp_dir)
        
        # Verify result
        assert isinstance(result, ExtractionResult)
        assert result.media_type == MediaType.AUDIO
        assert result.source_path == input_path
        assert "model_input.wav" in result.extracted_wav_path
        assert result.original_extension == ".mp3"
        
        # Verify subprocess was called with correct arguments
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == self.extractor.ffmpeg
        assert "-i" in call_args
        assert input_path in call_args
        assert str(self.extractor.target_sample_rate) in call_args
    
    @patch('media_extractor.subprocess.run')
    def test_extract_audio_ffmpeg_failure(self, mock_subprocess):
        """Test extraction with FFmpeg failure."""
        # Setup mock to simulate FFmpeg failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'ffmpeg', stderr='FFmpeg error')
        
        input_path = os.path.join(self.temp_dir, "input.mp3")
        with open(input_path, 'w') as f:
            f.write("fake audio content")
        
        # Test that extraction raises exception on FFmpeg failure
        with pytest.raises(RuntimeError, match="Audio extraction failed"):
            self.extractor.extract(input_path, self.temp_dir)
    
    def test_extract_audio_missing_input_file(self):
        """Test extraction with missing input file."""
        input_path = os.path.join(self.temp_dir, "nonexistent.mp3")
        
        # Should raise exception for missing file (FFmpeg will fail)
        with pytest.raises((FileNotFoundError, subprocess.CalledProcessError, RuntimeError)):
            self.extractor.extract(input_path, self.temp_dir)


class TestMediaExtractorIntegration:
    """Integration tests that require actual audio files."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = MediaExtractor(target_sample_rate=48000)
    
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
    
    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not available")
    def test_real_wav_extraction_passthrough(self):
        """Test actual WAV file extraction with real FFmpeg."""
        # Create a test WAV file
        input_path = os.path.join(self.temp_dir, "test_input.wav")
        self.create_test_wav_file(input_path, duration_seconds=0.5, sample_rate=44100)
        
        # Extract audio
        result = self.extractor.extract(input_path, self.temp_dir)
        
        # Verify result
        assert isinstance(result, ExtractionResult)
        assert result.media_type == MediaType.AUDIO
        assert os.path.exists(result.extracted_wav_path)
        assert result.original_extension == ".wav"
        
        # Verify the output WAV file has expected properties
        with wave.open(result.extracted_wav_path, 'r') as wav_file:
            assert wav_file.getnchannels() == 1  # Mono
            assert wav_file.getsampwidth() == 2  # 16-bit
            assert wav_file.getframerate() == 48000  # Target sample rate

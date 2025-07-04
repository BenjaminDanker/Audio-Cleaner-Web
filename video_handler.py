import os
import sys
import shutil
import subprocess
import tempfile
import time
import threading
from pathlib import Path

import imageio_ffmpeg
from df.enhance import enhance, init_df, load_audio, save_audio
from moviepy.video.io.VideoFileClip import VideoFileClip

# Constants moved from processing.py
AUDIO_BITRATE_AAC = "320k"
FFMPEG_TIMEOUT_S = 300
DEFAULT_ATTEN_DB = 30

class VideoProcessor:
    def __init__(self, uploaded_file, atten_db):
        self.uploaded_file = uploaded_file
        self.atten_db = atten_db
        self.temp_dir_path = tempfile.mkdtemp()
        
        self.input_filename = uploaded_file.filename if uploaded_file.filename else "uploaded_video"
        self.input_path = os.path.join(self.temp_dir_path, self.input_filename)
        self.output_path = os.path.join(self.temp_dir_path, "output.mp4")

        # Initialize DF model once per processor instance
        self.model, self.df_state = self._init_df_model()

    def _resource_path(self, relative_path: str) -> str:
        """Resolve resource path for bundled or development environments."""
        base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
        return os.path.join(base_path, relative_path)

    def _init_df_model(self):
        model_path = self._resource_path("models/DeepFilterNet3")
        model, df_state, _ = init_df(model_path, post_filter=True)
        return model, df_state

    def _remux(self, source_path: str, target_temp_dir: str) -> str:
        out_filename = "remux_" + Path(source_path).name
        out_path = Path(target_temp_dir) / out_filename
        ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-y", "-ignore_editlist", "1", "-i", source_path,
            "-map", "0:v", "-map", "0:a?", "-c", "copy",
            "-movflags", "+faststart", str(out_path),
        ]
        subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT_S, capture_output=True, text=True)
        return str(out_path)

    def _extract_audio(self, video_path: str, temp_dir: str) -> str:
        audio_path = os.path.join(temp_dir, "temp_original_audio.wav")
        clip = VideoFileClip(video_path)
        try:
            if clip.audio is None:
                raise ValueError("No audio track found.")
            clip.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio extraction failed.")
            return audio_path
        finally:
            clip.close()

    def _enhance_audio(self, audio_path: str, atten_lim_db: int | None):
        audio, _ = load_audio(audio_path, sr=self.df_state.sr())
        enhanced = enhance(self.model, self.df_state, audio, atten_lim_db=atten_lim_db)
        return enhanced, self.df_state.sr()

    def _save_enhanced_audio(self, enhanced_audio, sample_rate: int, temp_dir: str) -> str:
        path = os.path.join(temp_dir, "temp_enhanced_audio.wav")
        save_audio(path, enhanced_audio, sr=sample_rate)
        return path

    def _replace_audio(self, video_input: str, new_audio: str, output_video: str) -> None:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-ignore_editlist", "1", "-i", video_input,
            "-i", new_audio, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE_AAC,
            "-shortest", output_video,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)

    def process(self):
        """
        Saves the uploaded file and processes the video.
        Returns the path to the processed output file.
        Raises exceptions on failure.
        """
        self.uploaded_file.save(self.input_path)
        
        remuxed_path = self._remux(self.input_path, self.temp_dir_path)
        original_audio_path = self._extract_audio(remuxed_path, self.temp_dir_path)
        
        enhanced_audio_data, sr = self._enhance_audio(original_audio_path, self.atten_db)
        enhanced_audio_path = self._save_enhanced_audio(enhanced_audio_data, sr, self.temp_dir_path)
        
        self._replace_audio(remuxed_path, enhanced_audio_path, self.output_path)
        return self.output_path

    def schedule_cleanup(self, logger):
        """
        Schedules the cleanup of the temporary directory in a background thread.
        """
        # Capture self.temp_dir_path for the thread, as self might not be available
        # or could change if the instance is managed unexpectedly.
        temp_dir_to_clean = self.temp_dir_path
        
        def delayed_clean():
            time.sleep(3.0)  # Delay to allow file handles to be released
            try:
                if os.path.exists(temp_dir_to_clean):
                    shutil.rmtree(temp_dir_to_clean)
            except Exception as e:
                logger.error(f"Error cleaning up {temp_dir_to_clean} in background thread: {e}")
        
        threading.Thread(target=delayed_clean).start()

    def immediate_cleanup(self, logger):
        """
        Attempts to immediately clean up the temporary directory.
        """
        try:
            if os.path.exists(self.temp_dir_path):
                shutil.rmtree(self.temp_dir_path)
        except Exception as e:
            logger.error(f"Error during immediate cleanup of {self.temp_dir_path}: {e}")


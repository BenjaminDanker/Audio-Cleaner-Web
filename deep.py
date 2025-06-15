import sys
import os
import traceback
import time
import shutil
from pathlib import Path
import imageio_ffmpeg
import subprocess
import tempfile # Ensure tempfile is imported
from contextlib import contextmanager

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QSlider, QCheckBox, QMessageBox, QProgressBar
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt, QThread, pyqtSignal, QTimer # Added QTimer

# --- Constants ---
DEFAULT_ATTEN_DB = 30
MIN_ATTEN_DB = 1
MAX_ATTEN_DB = 60
AUDIO_BITRATE_AAC = "320k"
TEMP_FILE_CLEANUP_DELAY_S = 0.5
FFMPEG_TIMEOUT_S = 300 # 5 minutes for FFmpeg operations, adjust as needed

PROGRESS_STEPS = {
    'REMUX_DONE': 5,
    'AUDIO_EXTRACTED': 20,
    'DF_MODEL_INITIALIZED': 35,
    'AUDIO_ENHANCED': 70,
    'ENHANCED_AUDIO_SAVED': 85,
    'FINAL_VIDEO_READY': 100
}

# --- DeepFilterNet Imports ---
try:
    import soundfile as sf
    from df.enhance import enhance, init_df, load_audio, save_audio
    # Change moviepy import to use the editor module
    from moviepy.video.io.VideoFileClip import VideoFileClip
    DEEPFILTER_AVAILABLE = True
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure deepfilternet, soundfile, moviepy imageio_ffmpeg are installed: pip install deepfilternet soundfile moviepy imageio-ffmpeg")
    DEEPFILTER_AVAILABLE = False
    class QThread: pass
    pyqtSignal = lambda *args, **kwargs: None

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # _MEIPASS not set, running in development mode
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

@contextmanager
def managed_temp_directory():
    """Context manager for creating and cleaning up a temporary directory."""
    temp_dir_path = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir_path}")
    try:
        yield temp_dir_path
    finally:
        print(f"Initiating cleanup for temporary directory: {temp_dir_path}")
        time.sleep(TEMP_FILE_CLEANUP_DELAY_S)
        try:
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path) # shutil.rmtree handles non-empty directories
                print(f"Successfully removed temporary directory: {temp_dir_path}")
            else:
                print(f"Temporary directory {temp_dir_path} already removed or never existed.")
        except Exception as e:
            print(f"Error during cleanup of {temp_dir_path}: {e}")
            traceback.print_exc()

def remux(source_path: str, target_temp_dir: str) -> str:
    """
    Strip Sony 'rtmd' metadata, keep video+audio, rebuild a clean MP4
    into the target_temp_dir.
    """
    out_filename = "remux_" + Path(source_path).name
    out_path = Path(target_temp_dir) / out_filename
    ffmpeg  = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-ignore_editlist", "1",
        "-i", source_path,
        "-map", "0:v", "-map", "0:a?",
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path)
    ]
    try:
        # Added timeout to subprocess.run
        subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT_S, capture_output=True, text=True)
    except subprocess.TimeoutExpired as e:
        raise Exception(f"Remuxing timed out after {FFMPEG_TIMEOUT_S} seconds.") from e
    except subprocess.CalledProcessError as e:
        error_output = e.stderr or e.stdout or "Unknown remux error"
        raise Exception(f"Remuxing failed: {error_output[:500]}") from e
    return str(out_path)


# --- Worker Thread for Denoising ---
class DenoiseWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, input_video_abs_path, output_video_abs_path, atten_lim_db_val):
        super().__init__()
        self.original_input_video = input_video_abs_path
        self.output_video = output_video_abs_path
        self.atten_lim_db = atten_lim_db_val
        self._is_running = True
        self._error_already_emitted = False
        self.current_video_clip = None # For MoviePy VideoFileClip

    def _emit_error_once(self, message):
        if not self._error_already_emitted:
            self.error.emit(message)
            self._error_already_emitted = True

    def _check_if_cancelled(self):
        if not self._is_running:
            self.status_update.emit("Operation cancelled by user.")
            return True
        return False

    def _cleanup_resources(self):
        if self.current_video_clip:
            try:
                self.current_video_clip.close()
                self.current_video_clip = None
                print("Video clip closed in _cleanup_resources.")
            except Exception as e:
                print(f"Error closing video clip in _cleanup_resources: {e}")

    def run(self):
        if not DEEPFILTER_AVAILABLE:
            self._emit_error_once("Core dependencies (deepfilternet, etc.) not found.")
            return

        self._is_running = True # Ensure it's true at the start
        self._error_already_emitted = False # Reset flag

        try:
            with managed_temp_directory() as temp_dir:
                if self._check_if_cancelled(): return

                # Step 1: Remux Video
                self.status_update.emit("Step 1/6: Remuxing video...")
                remuxed_video_path = self._perform_remux_step(temp_dir)
                self.progress.emit(PROGRESS_STEPS['REMUX_DONE'])
                if self._check_if_cancelled(): return

                # Step 2: Extract Audio
                self.status_update.emit("Step 2/6: Extracting audio...")
                original_audio_path = self._perform_audio_extraction(remuxed_video_path, temp_dir)
                self.progress.emit(PROGRESS_STEPS['AUDIO_EXTRACTED'])
                if self._check_if_cancelled(): return

                # Step 3: Initialize DeepFilterNet Model
                self.status_update.emit("Step 3/6: Initializing Denoise Model...")
                model, df_state = self._initialize_df_model()
                self.progress.emit(PROGRESS_STEPS['DF_MODEL_INITIALIZED'])
                if self._check_if_cancelled(): return

                # Step 4: Enhance Audio
                self.status_update.emit("Step 4/6: Enhancing audio...")
                enhanced_audio, sr = self._enhance_audio_with_df(original_audio_path, model, df_state)
                self.progress.emit(PROGRESS_STEPS['AUDIO_ENHANCED'])
                if self._check_if_cancelled(): return

                # Step 5: Save Enhanced Audio
                self.status_update.emit("Step 5/6: Saving enhanced audio...")
                enhanced_audio_path = self._save_enhanced_audio_file(enhanced_audio, sr, temp_dir)
                self.progress.emit(PROGRESS_STEPS['ENHANCED_AUDIO_SAVED'])
                if self._check_if_cancelled(): return

                # Step 6: Replace Audio in Video (FFmpeg)
                self.status_update.emit("Step 6/6: Finalizing video with new audio...")
                self._replace_audio_in_final_video(remuxed_video_path, enhanced_audio_path)
                self.progress.emit(PROGRESS_STEPS['FINAL_VIDEO_READY'])

                if not self._check_if_cancelled():
                    self.finished.emit(self.output_video, f"Successfully denoised and saved to:\n{self.output_video}")

        except Exception as e:
            print(f"Error in DenoiseWorker run: {e}")
            traceback.print_exc()
            self._emit_error_once(f"An unexpected error occurred: {str(e)[:500]}")
        finally:
            self._cleanup_resources()
            # Temporary directory is cleaned by the 'with managed_temp_directory()' context manager

    def _perform_remux_step(self, temp_dir):
        return remux(self.original_input_video, temp_dir)

    def _perform_audio_extraction(self, video_path_for_extraction, temp_dir):
        original_audio_path = os.path.join(temp_dir, "temp_original_audio.wav")
        try:
            self.current_video_clip = VideoFileClip(video_path_for_extraction)
            if self.current_video_clip.audio is None:
                raise ValueError("Video has no audio track after remuxing.")
            self.current_video_clip.audio.write_audiofile(original_audio_path, codec='pcm_s16le', logger=None)
            if not os.path.exists(original_audio_path):
                raise FileNotFoundError("Audio extraction failed to produce an output file.")
            return original_audio_path
        except Exception as e:
            raise Exception(f"Audio extraction failed: {e}") from e
        finally:
            if self.current_video_clip:
                self.current_video_clip.close()
                self.current_video_clip = None

    def _initialize_df_model(self):
        try:
            model_path = resource_path("models/DeepFilterNet3")
            model, df_state, _ = init_df(model_path, post_filter=True)
            return model, df_state
        except Exception as e:
            raise Exception(f"DeepFilterNet model initialization failed: {e}") from e

    def _enhance_audio_with_df(self, audio_to_enhance_path, model, df_state):
        try:
            audio, sr = load_audio(audio_to_enhance_path, sr=df_state.sr())
            enhanced_audio = enhance(model, df_state, audio, atten_lim_db=self.atten_lim_db)
            if enhanced_audio is None:
                raise ValueError("Denoising (enhance function) produced no audio data.")
            return enhanced_audio, df_state.sr()
        except Exception as e:
            raise Exception(f"Audio enhancement failed: {e}") from e

    def _save_enhanced_audio_file(self, enhanced_audio_data, sample_rate, temp_dir):
        enhanced_audio_path = os.path.join(temp_dir, "temp_enhanced_audio.wav")
        try:
            save_audio(enhanced_audio_path, enhanced_audio_data, sr=sample_rate)
            return enhanced_audio_path
        except Exception as e:
            raise Exception(f"Saving enhanced audio failed: {e}") from e

    def _replace_audio_in_final_video(self, video_input_path, new_audio_path):
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y", "-ignore_editlist", "1",
                "-i", video_input_path,
                "-i", new_audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE_AAC,
                "-shortest", self.output_video
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
            print("FFmpeg stdout:", result.stdout)
            print("FFmpeg stderr:", result.stderr)
        except subprocess.TimeoutExpired as e:
            raise Exception(f"Final video processing (FFmpeg) timed out after {FFMPEG_TIMEOUT_S} seconds.") from e
        except subprocess.CalledProcessError as e:
            error_output = e.stderr or e.stdout or "Unknown FFmpeg error during final video processing"
            raise Exception(f"FFmpeg error during final video processing: {error_output[:500]}") from e
        except FileNotFoundError:
            raise FileNotFoundError("FFmpeg executable not found. Ensure imageio-ffmpeg is correctly installed.") from None
        except Exception as e:
            raise Exception(f"Unexpected error during final video processing: {e}") from e

    def stop(self):
        self.status_update.emit("Cancellation signal received by worker.")
        self._is_running = False
        print("DenoiseWorker stop requested.")
        # Note: Blocking calls within steps (like FFmpeg or long MoviePy ops)
        # won't be interrupted by this flag alone. Timeouts in subprocess calls help.


# --- Main Application Window ---
class VideoDenoiserApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Audio Denoiser")
        self.setGeometry(100, 100, 800, 650) # Increased height slightly for cancel button

        self.input_video_path = ""
        self.output_video_path = ""
        self.denoise_worker = None
        self.cancel_check_timer = QTimer(self) # Timer for checking cancellation status

        self._setup_ui()
        self._connect_signals()
        self.update_denoise_button_state()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.file_layout = QHBoxLayout()
        self.controls_layout = QHBoxLayout()
        self.denoise_controls_layout = QHBoxLayout() # Renamed for clarity

        # Input File
        self.input_label = QLabel("Input Video:")
        self.input_lineedit = QLineEdit()
        self.input_lineedit.setReadOnly(True)
        self.input_button = QPushButton("Browse...")

        # Output File
        self.output_label = QLabel("Output Video:")
        self.output_lineedit = QLineEdit()
        self.output_lineedit.setReadOnly(True)
        self.output_button = QPushButton("Select Save Location...")

        self.file_layout.addWidget(self.input_label)
        self.file_layout.addWidget(self.input_lineedit)
        self.file_layout.addWidget(self.input_button)
        self.file_layout.addWidget(self.output_label)
        self.file_layout.addWidget(self.output_lineedit)
        self.file_layout.addWidget(self.output_button)

        # Video Player
        self.video_widget = QVideoWidget()
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        # Playback Controls
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setEnabled(False)

        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.seek_slider)

        # Denoise Controls
        self.atten_checkbox = QCheckBox("Use Default Attenuation")
        self.atten_checkbox.setToolTip(f"Default: {DEFAULT_ATTEN_DB}dB. Lower = Better Sound Quality, Higher = More Noise Removal.")
        self.atten_checkbox.setChecked(True)

        self.atten_slider = QSlider(Qt.Orientation.Horizontal)
        self.atten_slider.setRange(MIN_ATTEN_DB, MAX_ATTEN_DB)
        self.atten_slider.setValue(DEFAULT_ATTEN_DB)
        self.atten_slider.setEnabled(False)

        self.atten_label = QLabel(f"Limit: {DEFAULT_ATTEN_DB} dB")
        self.atten_label.setMinimumWidth(90) # Adjusted width
        self.atten_label.setVisible(False)

        self.denoise_button = QPushButton("Denoise Video")
        self.denoise_button.setEnabled(False)
        
        self.cancel_button = QPushButton("Cancel Denoising")
        self.cancel_button.setVisible(False) # Initially hidden

        self.denoise_controls_layout.addWidget(self.atten_checkbox)
        self.denoise_controls_layout.addWidget(self.atten_slider)
        self.denoise_controls_layout.addWidget(self.atten_label)
        self.denoise_controls_layout.addStretch()
        self.denoise_controls_layout.addWidget(self.denoise_button)
        self.denoise_controls_layout.addWidget(self.cancel_button) # Placed next to denoise

        # Progress Bar and Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("Select input and output files.")

        self.main_layout.addLayout(self.file_layout)
        self.main_layout.addWidget(self.video_widget, stretch=1)
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.addLayout(self.denoise_controls_layout)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.status_label)

    def _connect_signals(self):
        self.input_button.clicked.connect(self.browse_input)
        self.output_button.clicked.connect(self.browse_output)
        
        self.media_player.errorOccurred.connect(self.handle_media_error)
        self.play_button.clicked.connect(self.toggle_play)
        self.seek_slider.sliderMoved.connect(self.set_position)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        
        self.atten_checkbox.stateChanged.connect(self.toggle_atten_slider)
        self.atten_slider.valueChanged.connect(self.update_atten_label)
        
        self.denoise_button.clicked.connect(self.start_denoising)
        self.cancel_button.clicked.connect(self.request_cancel_denoising)
        self.cancel_check_timer.timeout.connect(self._check_worker_status_after_cancel_request)

    def _set_ui_for_denoising_state(self, is_denoising_active):
        """Manages UI elements' enabled/disabled/visible state during denoising."""
        self.denoise_button.setVisible(not is_denoising_active)
        self.cancel_button.setVisible(is_denoising_active)
        if is_denoising_active: # Ensure cancel button is enabled when it becomes visible
            self.cancel_button.setEnabled(True)

        self.progress_bar.setVisible(is_denoising_active)
        if is_denoising_active:
            self.progress_bar.setValue(0)

        # Disable file selection and denoise option controls during processing
        self.input_button.setEnabled(not is_denoising_active)
        self.output_button.setEnabled(not is_denoising_active)
        self.atten_checkbox.setEnabled(not is_denoising_active)
        # Slider's state depends on checkbox, only if not denoising
        is_slider_enabled = not is_denoising_active and not self.atten_checkbox.isChecked()
        self.atten_slider.setEnabled(is_slider_enabled)
        self.atten_label.setVisible(is_slider_enabled) # Label visibility tied to slider

    def browse_input(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Video Files (*.mp4 *.avi *.mov *.mkv)')
        if fname:
            self.input_video_path = fname
            self.input_lineedit.setText(os.path.basename(fname))
            self.media_player.stop() # Stop previous playback
            self.media_player.setSource(QUrl.fromLocalFile(fname))
            self.play_button.setEnabled(True)
            self.seek_slider.setEnabled(True)
            self.play_button.setText("Play") # Set initial text
            self.status_label.setText("Input video loaded. Ready to play or denoise.")
            self.update_denoise_button_state()

            # Play and immediately pause to show the first frame
            self.media_player.play()
            self.media_player.pause()
            # Ensure slider is at the beginning
            self.seek_slider.setValue(0)

    def browse_output(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Denoised Video As...', '', 'MP4 Video Files (*.mp4)')
        if fname:
            # Ensure it has .mp4 extension
            if not fname.lower().endswith('.mp4'):
                fname += '.mp4'
            self.output_video_path = fname
            self.output_lineedit.setText(os.path.basename(fname))
            self.status_label.setText("Output path selected.")
            self.update_denoise_button_state()

    def update_denoise_button_state(self):
        enabled = bool(self.input_video_path and self.output_video_path and DEEPFILTER_AVAILABLE)
        self.denoise_button.setEnabled(enabled)
        if not DEEPFILTER_AVAILABLE:
             self.status_label.setText("Error: Core dependencies missing. Cannot denoise.")


    def toggle_play(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            # If paused at the beginning, ensure it plays from the start
            if self.media_player.position() == 0 and self.media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
                 self.media_player.setPosition(0) # Explicitly set position just in case
            self.media_player.play()
            self.play_button.setText("Pause")

    def set_position(self, position):
        self.media_player.setPosition(position)

    def position_changed(self, position):
        self.seek_slider.setValue(position)

    def duration_changed(self, duration):
        self.seek_slider.setRange(0, duration)

    def handle_media_error(self, error, error_string=""):
         # Handle cases where error might be an enum or an integer
        error_code = error if isinstance(error, int) else error.value
        print(f"Media Player Error ({error_code}): {self.media_player.errorString()}")
        QMessageBox.warning(self, "Media Player Error", f"Could not play the video:\n{self.media_player.errorString()}")
        self.play_button.setEnabled(False)
        self.seek_slider.setEnabled(False)


    def toggle_atten_slider(self, state):
        use_default = (state == Qt.CheckState.Checked.value)
        self.atten_slider.setEnabled(not use_default)
        self.atten_label.setVisible(not use_default)
        if use_default: # Reset to default if checkbox is re-checked
            self.atten_slider.setValue(DEFAULT_ATTEN_DB)
            self.update_atten_label(DEFAULT_ATTEN_DB)

    def update_atten_label(self, value):
        self.atten_label.setText(f"Limit: {value} dB")

    def start_denoising(self):
        if not self.input_video_path or not self.output_video_path:
            QMessageBox.warning(self, "Missing Information", "Please select both input and output video files.")
            return

        if self.denoise_worker and self.denoise_worker.isRunning():
            QMessageBox.information(self, "Busy", "Denoising process is already running.")
            return

        atten_limit_val = None
        if not self.atten_checkbox.isChecked():
            atten_limit_val = self.atten_slider.value()

        self._set_ui_for_denoising_state(True)
        self.status_label.setText("Starting denoising process...")

        self.denoise_worker = DenoiseWorker(self.input_video_path, self.output_video_path, atten_limit_val)
        self.denoise_worker.progress.connect(self.update_progress_bar)
        self.denoise_worker.status_update.connect(self.update_status_text_label) # New connection
        self.denoise_worker.finished.connect(self.handle_denoising_finished)
        self.denoise_worker.error.connect(self.handle_denoising_error)
        self.denoise_worker.start()

    def request_cancel_denoising(self):
        if self.denoise_worker and self.denoise_worker.isRunning():
            self.status_label.setText("Cancellation requested. Waiting for current step to finish...")
            self.cancel_button.setEnabled(False) # Prevent multiple clicks
            self.denoise_worker.stop()
            self.cancel_check_timer.start(500) # Check every 500ms
            self._cancel_checks_count = 0 # Reset counter for timeout

    def _check_worker_status_after_cancel_request(self):
        self._cancel_checks_count += 1
        worker_still_running = self.denoise_worker and self.denoise_worker.isRunning()

        if not worker_still_running:
            self.cancel_check_timer.stop()
            self.status_label.setText("Denoising process effectively cancelled.")
            self._set_ui_for_denoising_state(False) # Reset UI
            # Denoise button state will be updated by _set_ui_for_denoising_state
        elif self._cancel_checks_count > 40: # Timeout after 20 seconds (40 * 500ms)
            self.cancel_check_timer.stop()
            QMessageBox.warning(self, "Cancellation Timeout",
                                "Worker did not stop gracefully after cancel request. "
                                "It might be stuck in a long operation. "
                                "If issues persist, please restart the application.")
            self._set_ui_for_denoising_state(False) # Force UI reset
            self.update_denoise_button_state() # Ensure denoise button is correctly set

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        # self.status_label.setText(f"Denoising in progress... {value}%") # Status now handled by status_update

    def update_status_text_label(self, message): # Slot for worker's status_update signal
        self.status_label.setText(message)

    def handle_denoising_finished(self, output_path, message):
        self.cancel_check_timer.stop() # Stop timer if it was running for cancellation
        self._set_ui_for_denoising_state(False)
        self.status_label.setText(message)
        QMessageBox.information(self, "Success", message)
        self.update_denoise_button_state() # Re-evaluates denoise button state
        
        # Optionally load the denoised video
        self.media_player.stop()
        self.media_player.setSource(QUrl.fromLocalFile(output_path))
        self.play_button.setText("Play Denoised") # Indicate it's the new video
        self.play_button.setEnabled(True)
        self.seek_slider.setEnabled(True)
        self.seek_slider.setValue(0) # Reset slider for new video

    def handle_denoising_error(self, error_message):
        self.cancel_check_timer.stop() # Stop timer if it was running
        self._set_ui_for_denoising_state(False)
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Denoising Error", error_message)
        self.update_denoise_button_state()

    def closeEvent(self, event):
        if self.denoise_worker and self.denoise_worker.isRunning():
            self.status_label.setText("Attempting to stop worker before closing...")
            self.denoise_worker.stop()
            # Give the worker a chance to stop
            if not self.denoise_worker.wait(3000): # Wait up to 3 seconds
                QMessageBox.warning(self, "Worker Busy",
                                    "Denoising worker did not stop gracefully. "
                                    "It might be forcefully terminated.")
                self.denoise_worker.terminate() # Last resort
                self.denoise_worker.wait(1000) # Wait for termination to complete
        
        if self.media_player: # Ensure media player is stopped
            self.media_player.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoDenoiserApp()
    window.show()
    sys.exit(app.exec())
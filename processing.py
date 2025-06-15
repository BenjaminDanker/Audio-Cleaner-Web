import os
import sys
import shutil
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import imageio_ffmpeg
from df.enhance import enhance, init_df, load_audio, save_audio
from moviepy.video.io.VideoFileClip import VideoFileClip

DEFAULT_ATTEN_DB = 30
AUDIO_BITRATE_AAC = "320k"
TEMP_FILE_CLEANUP_DELAY_S = 0.5
FFMPEG_TIMEOUT_S = 300


def resource_path(relative_path: str) -> str:
    """Resolve resource path for bundled or development environments."""
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


@contextmanager
def managed_temp_directory():
    """Create and clean up a temporary directory."""
    temp_dir_path = tempfile.mkdtemp()
    try:
        yield temp_dir_path
    finally:
        time.sleep(TEMP_FILE_CLEANUP_DELAY_S)
        try:
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path)
        except Exception as e:
            print(f"Error cleaning temp directory {temp_dir_path}: {e}")
            traceback.print_exc()


def remux(source_path: str, target_temp_dir: str) -> str:
    """Strip metadata and rebuild a clean MP4."""
    out_filename = "remux_" + Path(source_path).name
    out_path = Path(target_temp_dir) / out_filename
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-ignore_editlist",
        "1",
        "-i",
        source_path,
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT_S, capture_output=True, text=True)
    return str(out_path)


def extract_audio(video_path: str, temp_dir: str) -> str:
    audio_path = os.path.join(temp_dir, "temp_original_audio.wav")
    clip = VideoFileClip(video_path)
    try:
        if clip.audio is None:
            raise ValueError("Video has no audio track after remuxing.")
        clip.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
        if not os.path.exists(audio_path):
            raise FileNotFoundError("Audio extraction failed to produce an output file.")
        return audio_path
    finally:
        clip.close()


def init_df_model():
    model_path = resource_path("models/DeepFilterNet3")
    model, df_state, _ = init_df(model_path, post_filter=True)
    return model, df_state


def enhance_audio(audio_path: str, model, df_state, atten_lim_db: int | None):
    audio, _ = load_audio(audio_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim_db)
    return enhanced, df_state.sr()


def save_enhanced_audio(enhanced_audio, sample_rate: int, temp_dir: str) -> str:
    path = os.path.join(temp_dir, "temp_enhanced_audio.wav")
    save_audio(path, enhanced_audio, sr=sample_rate)
    return path


def replace_audio(video_input: str, new_audio: str, output_video: str) -> None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ignore_editlist",
        "1",
        "-i",
        video_input,
        "-i",
        new_audio,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE_AAC,
        "-shortest",
        output_video,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)


def denoise_video(input_video: str, output_video: str, atten_lim_db: int | None = None) -> None:
    with managed_temp_directory() as temp_dir:
        remuxed = remux(input_video, temp_dir)
        audio = extract_audio(remuxed, temp_dir)
        model, df_state = init_df_model()
        enhanced_audio, sr = enhance_audio(audio, model, df_state, atten_lim_db)
        enhanced_path = save_enhanced_audio(enhanced_audio, sr, temp_dir)
        replace_audio(remuxed, enhanced_path, output_video)



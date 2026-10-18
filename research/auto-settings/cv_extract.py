"""Common Voice 17.0 English test shard (CC0), first N clips -> 48 kHz mono WAV (real-world recordings, no labels used)."""
import os, subprocess, sys, tarfile
import soundfile as sf
B = "/home/silver/audio-cleaner-train"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
FF = subprocess.run([sys.executable, "-c", "import imageio_ffmpeg as i;print(i.get_ffmpeg_exe())"], capture_output=True, text=True).stdout.strip()
out = f"{B}/clips/cv"; os.makedirs(out, exist_ok=True)
tmp = f"{B}/work/tmp/cv"; os.makedirs(tmp, exist_ok=True)
n = 0
with tarfile.open(f"{B}/data/cv/en_test_0.tar") as tf:
    for m in tf:
        if not m.name.endswith(".mp3"):
            continue
        if n >= N:
            break
        src = f"{tmp}/{n}.mp3"
        with open(src, "wb") as f:
            f.write(tf.extractfile(m).read())
        dst = f"{out}/cv_{n:04d}.wav"
        subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", src, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", dst], check=True)
        os.remove(src)
        n += 1
print("clips", n)

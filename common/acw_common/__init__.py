"""Code shared by the Audio Cleaner server and desktop app (desktop/).

    dfn       DeepFilterNet3 through libDF's C API (ctypes, no torch)
    clarity   post-denoise polish chain (NumPy/SciPy)
    ffmpeg    ffprobe/ffmpeg wrappers (probe, decode, encode, remux)
    pipeline  the "Studio clean" chain built from the three above

Pure Python + NumPy/SciPy/pyloudnorm; the native pieces (libDF, FFmpeg) are
passed in as paths so each product can ship or locate its own builds.
"""
__version__ = "1.0.0"

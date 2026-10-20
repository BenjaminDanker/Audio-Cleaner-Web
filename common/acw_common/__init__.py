"""Code shared by the Audio Cleaner server and desktop app (desktop/).

    dfn       DeepFilterNet3 through libDF's C API (ctypes, no torch)
    clarity   post-denoise leveling chain (NumPy/SciPy)
    ffmpeg    ffprobe/ffmpeg wrappers (probe, decode, encode, remux)
    pipeline  the "Studio clean" chain built from the three above
    blocks    the block grid and the block cache every model step runs on (whole file or part)
    timeline  the per-part editor's timeline: layers, parts, settings, JSON format and validation
    editor    the per-part editor: plan the blocks a timeline needs, billing inputs, render it
    suggest   automatic suggestions per part (change points, one decision per part, clipping spots)

Pure Python + NumPy/SciPy/pyloudnorm; the native pieces (libDF, FFmpeg) are
passed in as paths so each product can ship or locate its own builds.
"""
__version__ = "1.0.0"

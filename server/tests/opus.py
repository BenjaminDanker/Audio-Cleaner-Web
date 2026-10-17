"""A tiny libopus wrapper for tests that play an Opus client (the server side never decodes Opus in Python)."""
import ctypes
import ctypes.util

import numpy as np

_lib = ctypes.CDLL(ctypes.util.find_library("opus") or "libopus.so.0")
_lib.opus_encoder_create.restype = ctypes.c_void_p
_lib.opus_encoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.opus_encode_float.restype = ctypes.c_int32
_lib.opus_encode_float.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p, ctypes.c_int32]
_lib.opus_decoder_create.restype = ctypes.c_void_p
_lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.opus_decode_float.restype = ctypes.c_int
_lib.opus_decode_float.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]


class Encoder:
    def __init__(self, bitrate: int = 128_000):
        err = ctypes.c_int(0)
        self.st = _lib.opus_encoder_create(48_000, 1, 2049, ctypes.byref(err))
        _lib.opus_encoder_ctl(ctypes.c_void_p(self.st), 4002, ctypes.c_int32(bitrate))
        self.buf = ctypes.create_string_buffer(4000)

    def encode(self, x: np.ndarray) -> bytes:
        x = np.ascontiguousarray(np.clip(x, -1, 1), np.float32)
        n = _lib.opus_encode_float(self.st, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x), self.buf, 4000)
        return self.buf.raw[:n]


class Decoder:
    def __init__(self):
        err = ctypes.c_int(0)
        self.st = _lib.opus_decoder_create(48_000, 1, ctypes.byref(err))
        self.pcm = np.zeros(960, np.float32)

    def decode(self, pkt: bytes) -> np.ndarray:
        n = _lib.opus_decode_float(self.st, pkt, len(pkt), self.pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 960, 0)
        return self.pcm[:n].copy()

"""The live captioner's 48 -> 16 kHz conversion: batch by batch it gives what converting the whole stream at once
gives, whatever the batch sizes (network timing decides them)."""
import numpy as np
from scipy.signal import resample_poly

from acw.stream.engine import Decimate3


def test_batches_match_one_conversion_whatever_their_sizes():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(48000 * 2).astype(np.float32) * 0.1
    whole = resample_poly(x, 1, 3)
    for sizes in ([960], [960, 1920, 2880], [17, 961, 3], [48000 * 2]):
        d, out, i, k = Decimate3(), [], 0, 0
        while i < len(x):
            n = sizes[k % len(sizes)]
            out.append(d(x[i:i + n]))
            i += n
            k += 1
        got = np.concatenate(out)
        assert len(got) >= len(whole) - 11  # all but the last few samples (their window is not complete yet)
        np.testing.assert_allclose(got, whole[:len(got)], atol=1e-5)

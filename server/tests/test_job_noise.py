"""The Background noise choice for file jobs."""
from __future__ import annotations

import pytest

from acw import jobs


def _opts(**extra):
    return jobs.validate_options("denoise", "standard", None, extra)


def test_noise_defaults_to_auto_and_only_three_choices_are_taken():
    assert _opts()["noise"] == "auto"
    assert _opts()["audio_format"] == "flac"  # the default output: lossless like WAV, about half the size
    for value in ("auto", "light", "heavy"):
        assert _opts(noise=value)["noise"] == value
    for value in ("loud", "", None, 3):
        with pytest.raises(jobs.JobError, match="noise must be"):
            _opts(noise=value)

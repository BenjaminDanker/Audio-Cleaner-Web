"""The Background noise choice for file jobs."""
from __future__ import annotations

import pytest

from acw import jobs


def _opts(**extra):
    return jobs.validate_options("denoise", "standard", None, extra, False)


def test_noise_defaults_to_auto_and_accepts_the_three_choices():
    assert _opts()["noise"] == "auto"
    for value in ("auto", "light", "heavy"):
        assert _opts(noise=value)["noise"] == value


@pytest.mark.parametrize("value", ["loud", "", None, 3])
def test_other_values_are_refused(value):
    with pytest.raises(jobs.JobError, match="noise must be"):
        _opts(noise=value)

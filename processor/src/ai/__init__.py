"""AI inference plugin package.

Importing this package applies side-effect registrations for built-in tasks.
"""
from ai.base import registry  # noqa: F401
from ai.audio_denoise_dfnet import DenoiseDFNetTask, DeepFilterNetEnhancer  # noqa: F401

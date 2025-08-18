"""AI inference plugin package.

Conditional imports avoid pulling batch dependencies into streaming contexts.
"""
from ai.base import registry  # noqa: F401

# Try to import batch-dependent tasks (will fail in streaming-only containers)
try:
    from ai.audio_denoise_dfnet import DenoiseDFNetTask, DeepFilterNetEnhancer  # noqa: F401
except ImportError:
    # Running in streaming-only environment without batch dependencies
    pass

try:
    from ai.audio_clarity_task import ClarityTask  # noqa: F401 - registers 'clarity'
except ImportError:
    # audio_clarity_task depends on media_extractor
    pass

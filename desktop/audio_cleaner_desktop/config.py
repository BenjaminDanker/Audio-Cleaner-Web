APP_NAME = "Audio Cleaner"
ORG_NAME = "Audio Cleaner"
# Identifiers only (settings location, macOS bundle id); they are not addresses the app contacts.
ORG_DOMAIN = "audio-cleaner.local"
BUNDLE_ID = "local.audio-cleaner.desktop"

from . import __releases__

SITE_URL = "https://audiocleaner.live/"
CONTACT_URL = SITE_URL + "contact"  # where the source-code offers in the notices point

# The installers and the signed update.json come from the app's GitHub releases (one release per version, tag
# v<version>); the updater is the only thing that downloads from there.
RELEASES_URL = f"https://github.com/{__releases__}/releases/"
LATEST_URL = RELEASES_URL + "latest/download/"  # the newest release's files


def release_url(version: str) -> str:
    """The files of one version's release."""
    return RELEASES_URL + f"download/v{version}/"


# Our own converted models (AP-BWE, Sidon) and their licence texts are in one public Hugging Face model repository,
# every file pinned by this revision, its size and SHA-256 (resources.py). The revision MUST be a full 40-character
# commit id, never a branch name: a branch can change under the pins. "898d417c842c899b08e1dbd9510e1745f336af69" until the repository exists
# (packaging/check_model_urls.py skips its checks until then).
HF_MODELS_REPO = "SilverJunk08/audio-cleaner-models"
HF_MODELS_REVISION = "898d417c842c899b08e1dbd9510e1745f336af69"


def hf_models_url(path: str) -> str:
    """The pinned download address of `path` in our models repository."""
    return f"https://huggingface.co/{HF_MODELS_REPO}/resolve/{HF_MODELS_REVISION}/{path}"

APP_NAME = "Audio Cleaner"
ORG_NAME = "Audio Cleaner"
# Identifiers only (settings location, macOS bundle id); they are not addresses the app contacts.
ORG_DOMAIN = "audio-cleaner.local"
BUNDLE_ID = "local.audio-cleaner.desktop"

from . import __releases__, __released__, __version__

SITE_URL = "https://mc-multi-world.space/audio/"
CONTACT_URL = SITE_URL + "contact"  # where the source-code offers in the notices point

# Everything the app downloads comes from its GitHub releases (one release per version, tag v<version>): the installers
# and the signed update.json of the newest release, and the models of the app's own version.
RELEASES_URL = f"https://github.com/{__releases__}/releases/"
LATEST_URL = RELEASES_URL + "latest/download/"  # the newest release's files


def release_url(version: str) -> str:
    """The files of one version's release."""
    return RELEASES_URL + f"download/v{version}/"


# The models of this version: its own release (a source checkout, which has no release of its own, uses the newest).
MODELS_URL = release_url(__version__) if __released__ else LATEST_URL

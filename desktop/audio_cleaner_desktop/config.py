APP_NAME = "Audio Cleaner"
ORG_NAME = "Audio Cleaner"
# Identifiers only (settings location, macOS bundle id); they are not addresses the app contacts.
ORG_DOMAIN = "audio-cleaner.local"
BUNDLE_ID = "local.audio-cleaner.desktop"

# Where the app downloads from: our own server, in this ONE place (a domain move is one edit here). Release files are
# mirrored there; nothing the app, the installers or their texts say names any other host except Hugging Face (Whisper).
SITE_URL = "https://mc-multi-world.space/audio/"  # the website; everything below is under it
DOWNLOADS_BASE = SITE_URL + "api/downloads/"
CONTACT_URL = SITE_URL + "contact"  # where the source-code offers in the notices point
UPDATES_URL = DOWNLOADS_BASE + "desktop/"  # update.json, update.json.sig and the installers named in it
MODELS_URL = DOWNLOADS_BASE + "models/"  # the models archive

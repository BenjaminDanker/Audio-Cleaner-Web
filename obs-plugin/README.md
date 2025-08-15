# OBS Plugin (Placeholder)

This folder is reserved for an OBS plugin that:
- Captures mic audio and sends 2s PCM chunks to WS `/stream/{sessionId}`
- Receives processed audio and caption packets
- Replaces mic audio in OBS and injects 608/708 captions into RTMP (or overlay fallback)

Implementation TBD. Consider OBS WebSocket + custom audio filter or a native plugin.

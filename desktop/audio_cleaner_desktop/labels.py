"""The words the app uses for its choices, and how a job's settings are told in a sentence. No Qt in here: the job list
(its search, its saved summaries) and the settings panel both read it."""
from __future__ import annotations

from acw_common import reco_model

from . import blend, engine

STRENGTH_LABELS = [("gentle", "Gentle"), ("balanced", "Balanced"), ("strong", "Strong"), ("max", "Maximum")]
# The same wording as the website's Denoising form.
LOUDNESS_LABELS = {
    "podcast": "Podcast (−16 LUFS)", "youtube": "YouTube (−14 LUFS)", "broadcast": "Broadcast (−23 LUFS)",
    "keep": "Keep original loudness",
}
LOUDNESS_SHORT = {"podcast": "podcast", "youtube": "YouTube", "broadcast": "broadcast", "keep": "original loudness"}
NOISE_LABELS = {"auto": "Automatic", "light": "Light noise", "heavy": "Heavy noise"}
SUBTITLE_LABELS = {
    "off": "No subtitles", "srt": "SRT file", "vtt": "VTT file", "txt": "Text transcript", "embed": "Inside the video",
}
FORMAT_LABELS = {"same": "Same as the original", "wav": "WAV", "flac": "FLAC", "mp3": "MP3", "m4a": "M4A", "opus": "Opus"}
VIDEO_SIZE_LABELS = {
    "same": "Same as the original", "2160": "2160p (4K)", "1440": "1440p", "1080": "1080p (Full HD)", "720": "720p (HD)",
    "480": "480p",
}


def describe_settings(s: dict, has_video: bool) -> str:
    if s.get("parts"):
        parts = ["chosen parts, each with its own settings"]
    elif s.get("rebuild_voice"):
        parts = ["voice rebuilt"]
    elif s.get("picked_rebuild"):
        parts = [f"voice rebuilt (it sounded {reco_model.REBUILD_WORDS.get(s['picked_rebuild'], s['picked_rebuild'])}; keeps only speech)"]
    elif isinstance(s.get("strength_db"), (int, float)) and not isinstance(s.get("strength_db"), bool):
        parts = [f"{blend.strength_name(float(s['strength_db']))} noise reduction ({float(s['strength_db']):.0f} dB)"
                 if blend.strength_key(float(s["strength_db"])) is None
                 else dict(STRENGTH_LABELS)[blend.strength_key(float(s["strength_db"]))] + " noise reduction"]
    else:
        picked = s.get("picked_strength") if s.get("strength") == engine.AUTO_STRENGTH else None
        if s.get("strength") == engine.AUTO_STRENGTH:
            parts = [f"{dict(STRENGTH_LABELS)[picked]} noise reduction (picked for this file; keeps your voice as recorded)" if picked in engine.STRENGTHS
                     else "noise reduction picked for the file"]
        else:
            parts = [dict(STRENGTH_LABELS).get(s.get("strength"), "Maximum") + " noise reduction"]
    if s.get("noise", "auto") != "auto":
        parts.append(f"for {s['noise']} noise")
    if s.get("echo_track"):
        parts.append("speaker echo removed")
    if s.get("leveling", True):
        parts.append("leveled (" + LOUDNESS_SHORT.get(s.get("loudness"), "podcast") + ")")
    if not has_video and s.get("audio_format", "same") != "same":
        parts.append(FORMAT_LABELS[s["audio_format"]])
    if s.get("video_size", "same") != "same":
        parts.append("video " + VIDEO_SIZE_LABELS.get(s["video_size"], s["video_size"] + "p"))
    if s.get("subtitles", "off") != "off":
        parts.append("subtitles: " + SUBTITLE_LABELS[s["subtitles"]].lower())
    return ", ".join(parts)


def settings_dict(options: engine.JobOptions) -> dict:
    return {
        "strength": options.strength, "leveling": options.leveling, "loudness": options.loudness,
        "audio_format": options.audio_format, "subtitles": options.subtitles, "language": options.language,
        "noise": options.noise, "echo_track": options.echo_track,
        "rebuild_voice": options.rebuild_voice, "auto_rebuild": options.auto_rebuild,
        "video_size": options.video_size,
    }


# The choice between noise reduction and Rebuild the voice, in plain words (the settings, the job's part panel and the notes on a result
# all use these, so the same thing is said everywhere).
WHICH_TITLE = "Which one should I pick?"
WHICH_NOISE = (
    "Noise reduction keeps your voice exactly as recorded and never changes a word. Laughs, shouts, whispers and singing stay as they were. "
    "It works on any length, it is fast, and the gentler strengths keep some room sound. It removes most background sound, game sounds included."
)
WHICH_REBUILD = (
    "Rebuild the voice makes the voice again from scratch, so it also fixes echo, a muffled or phone-quality voice and distortion from "
    "recording too loud. The result is very clean and can sound slightly too clean. It keeps only speech: game sounds, music and room sound "
    "are all gone, and laughs, shouts and singing may come out oddly. On a very echoey recording it can change a word, so listen before you "
    "publish. It is slower, and it works in many languages."
)
WHICH_TEXT = f"Noise reduction: {WHICH_NOISE[len('Noise reduction '):]}\n\nRebuild the voice: {WHICH_REBUILD[len('Rebuild the voice '):]}"
WHICH_SHORT = {"noise": "Noise reduction keeps your voice exactly as recorded.",
               "rebuild": "Rebuild the voice fixes echo, a muffled or phone-quality voice and distortion, but keeps only speech."}


def rebuild_reason_text(reason: str | None) -> str:
    """Why the voice was rebuilt when the automatic settings chose it, in the terms of the choice above."""
    word = reco_model.REBUILD_WORDS.get(reason or "", reason or "damaged")
    return (f"the voice was rebuilt because it sounded {word}: rebuilding fixes that, but keeps only speech (game sounds, music and "
            "room sound are gone)")


def noise_pick_text() -> str:
    return "noise reduction picked: it keeps your voice exactly as recorded"

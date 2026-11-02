"""The words of the editor screen. Where the web's Cleanup page says the same thing, it says it in the same words (web/src/studio/copy.js,
web/src/components/studio/InfoModal.jsx); what only this app has (everything runs on this computer, nothing is billed) is told in its own."""
from __future__ import annotations

from .. import labels

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "ru": "Russian", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
}


def loudness_name(lufs: float | None) -> str:
    """The short name of a loudness target: Podcast, YouTube, Broadcast, Keep (or the number)."""
    if lufs is None:
        return "Keep"
    for name, value in (("Podcast", -16.0), ("YouTube", -14.0), ("Broadcast", -23.0)):
        if abs(value - lufs) < 0.05:
            return name
    return f"{lufs:g} LUFS"


def len_text(n: float) -> str:
    n = round(n)
    if n < 60:
        return f"{n} s"
    if n < 3600:
        return f"{n // 60} min" + (f" {n % 60} s" if n % 60 else "")
    m = n % 3600 // 60
    return f"{n // 3600} h" + (f" {m} min" if m else "")


def say_clock(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def selection_title(sel: tuple[float, float] | None, seconds: float) -> str:
    """"Whole file · 3:00" or "0:15–0:25 · 10 s" (the heading over the settings)."""
    if sel is None or (sel[0] <= 0 and sel[1] >= seconds):
        return f"Whole file · {say_clock(seconds)}"
    return f"{say_clock(sel[0])}–{say_clock(sel[1])} · {len_text(sel[1] - sel[0])}"


TIPS = {
    "v.off": "Leave the voice as it is in this part.",
    "v.noise": "Takes the background sound out and keeps your voice exactly as recorded; never changes words. Removes most background sound, including "
               "game sounds; the gentler strengths keep some room sound. Works on any length of recording.",
    "v.rebuild": "For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud: the whole voice is rebuilt, "
                 "which also brings back the high sounds that are missing. Keeps only speech, so game sounds, music and room sound are gone. Slower. "
                 "On very echoey recordings it can change a word, so listen before you publish.",
    "s.1": "Gentle: takes the edge off. Keeps some room sound and music, so it sounds the most natural.",
    "s.2": "Balanced: most of the noise goes; the room still sounds like a room.",
    "s.3": "Strong: nearly all of the noise goes.",
    "s.4": "Max: removes as much as it can. Best for noisy places, and good for most voices.",
    "r.0": "Auto: we measure how noisy the recording is and pick the best way to clean it.",
    "r.1": "Light noise: hiss, hum, a quiet room or a fan. Keeps the voice the most natural.",
    "r.2": "Heavy noise: a café, traffic, a crowd, wind: noise almost as loud as the voice.",
    "echo": "Removes your speaker echo using the file’s second audio track (your desktop sound). OBS can record your mic and your desktop audio on "
            "separate tracks: put the mic on track 1 and the desktop sound on track 2.",
    "level": "Evens out the volume and sets the loudness, with or without noise removal. Turn it off to keep the voice as it was recorded. It is one "
             "loudness for the whole file.",
    "l.0": "Leave the volume as it is.",
    "l.1": "Podcast: evens out the volume at −16 LUFS, the usual loudness for podcasts.",
    "l.2": "YouTube: evens out the volume at −14 LUFS, the loudness YouTube plays videos at.",
    "l.3": "Broadcast: evens out the volume at −23 LUFS, the TV and radio standard.",
    "l.4": "Keep loudness: evens out the volume but keeps the overall loudness as recorded.",
    "subs": "Writes down what is said, in the language you choose: subtitle files (SRT, VTT) and a transcript (TXT). Made on this computer from the original "
            "recording: nothing is uploaded.",
    "q.0": "No subtitles or transcript.",
    "q.1": "Made on this computer: nothing is uploaded. Fine for clear speech. The first time, it needs the speech model (a one-time download), and "
           "it is slower on long files.",
    "lang": "The language spoken. Detect automatically works it out from the recording.",
    "sf.srt": "SRT: subtitles that most video players and editors take.",
    "sf.vtt": "VTT: subtitles for web players.",
    "sf.txt": "TXT: the transcript as plain text, without times.",
    "sf.embed": "Put into the video: players can turn them on and off. They go into the video file you save (an MKV, MP4 or MOV). Video editors usually "
                "take the .srt file instead.",
    "sf.embed.no": "Only a video file can hold the subtitles: pick a video type under Output.",
    "hear": "Hear the file with {name} or without it. The waveform shows the difference.",
    "info": "Compare all the options",
}

CARD_LINES = {
    "voice": "Noise removal or Voice rebuild",
    "level": "Podcast, YouTube or Broadcast loudness",
    "subs": "Subtitles and a transcript",
}

VOICE_MODES = (
    ("off", "Off", ""),
    ("noise", "Noise removal", "keeps your voice as recorded · takes out most other sounds, game sounds too"),
    ("rebuild", "Voice rebuild", "fixes echo, muffled or distorted voices · keeps only speech"),
)

EMPTY_TITLE = "Choose a file"
EMPTY_LINE = "Take the noise out of a recording, and get subtitles and a transcript."
DROP_TITLE = "Drop an audio or video file here"
DROP_HINT = "or click to pick one. Almost any audio or video file: MP3, WAV, M4A, FLAC, WMA, AMR, MP4, MOV, MKV, AVI, WMV, MPEG and more."
OUTPUT_EMPTY = "The file type, name and video size go here."
NAME_TIP = "Tip: type another ending (.mp3, .flac, .mkv…) or pick a file type from the list. Each follows the other."

# "All the options": what the web's (i) window says, for what this app has; where it works differently, in its own words.
VOICE_POINTS = [
    "Keeps your voice exactly as recorded and never changes words.",
    "Laughs, shouts, whispers and singing stay as they were.",
    "Works on any length of recording.",
    "Fast.",
    "Removes most background sound, including game sounds. The gentler strengths keep some room sound.",
]
REBUILD_POINTS = [
    "Makes the voice again from scratch, so it also fixes echo, a muffled or phone-quality voice and distortion from recording too loud.",
    "Very clean, and can sound slightly too clean.",
    "Keeps only speech, so game sounds, music and room sound are all gone.",
    "Laughs, shouts and singing may come out oddly.",
    "On very echoey recordings it can change a word, so listen before you publish.",
    "Slower. Needs its model (a one-time download of 0.5 GB). Works in many languages.",
]
STRENGTH_TEXT = ("Gentle takes the edge off and keeps some room sound and music. Balanced: most of the noise goes; the room still sounds like a room. "
                 "Strong: nearly all of the noise goes. Max removes as much as it can; best for noisy places.")
REMOVER_TEXT = ("Auto measures how noisy the recording is and picks the best way. Light is for hiss, hum, a quiet room or a fan. Heavy is for a café, "
                "traffic, a crowd or wind: noise almost as loud as the voice.")
ECHO_TEXT = ("Removes your speaker echo using the file’s second audio track (your desktop sound). Only for files that have one: in OBS, put the mic on "
             "track 1 and the desktop sound on track 2.")
LEVELING_TEXT = "Evens out the volume and sets how loud the result is, with or without noise removal."
LEVELING_ROWS = [
    ("Podcast", "−16 LUFS, the usual loudness for podcasts"),
    ("YouTube", "−14 LUFS, the loudness YouTube plays videos at"),
    ("Broadcast", "−23 LUFS, the TV and radio standard"),
    ("Keep", "Evens out the volume but keeps the overall loudness as recorded"),
]
SUBTITLES_TEXT = ("Made on this computer from the original recording, in about 100 languages: nothing is uploaded. Pick the language yourself if the speech "
                  "is noisy or mixes accents. Files: SRT and VTT subtitles and a TXT transcript, and the subtitles can go into the video (the picture "
                  "is kept). It adds time to each file, and needs the speech model (a one-time download of 1.6 GB).")
GOOD_TO_KNOW = [
    "Everything happens on this computer. Nothing is uploaded, and nothing is billed.",
    "Your original is never changed. The result is saved next to it (or in the folder you chose in Settings), with “-cleaned” in its name.",
    "Choose a part of your file on the timeline to change just that part: a few seconds with an echo, a louder stretch, a muffled answer.",
    "Every finished file keeps its work in the job (its saved copies), so you can change a part or the strength and save again; only what changed is "
    "made again, so a new strength often takes seconds.",
    "A file that is stopped or fails leaves no half-made result.",
]
GOOD_TO_KNOW_ALTERNATIVES = [
    "Just a little hum? Your editor’s or OBS’s own noise filter may be enough.",
    "Only a bit noisy? Noise removal is enough; Voice rebuild is for echoey, muffled, phone-quality or distorted voices.",
    "English video for YouTube? Its free automatic captions are fine for a first pass.",
]

WHICH_NOISE = labels.WHICH_NOISE
WHICH_REBUILD = labels.WHICH_REBUILD

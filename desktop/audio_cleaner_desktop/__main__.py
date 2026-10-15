from __future__ import annotations

import argparse
import sys

from . import __version__, config
from .engine import AUDIO_CHOICES, DEFAULT_LOUDNESS, LANGUAGES, LOUDNESS_TARGETS, STRENGTHS, SUBTITLE_MODES, JobOptions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-cleaner",
        description=f"{config.APP_NAME}: remove background noise from video and audio files. "
        "Without --cli the app opens and queues any files given.",
    )
    p.add_argument("files", nargs="*", metavar="FILE", help="files to clean (--cli) or to queue in the window")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    p.add_argument("--cli", action="store_true", help="clean FILE... without opening a window")
    p.add_argument("--selftest", action="store_true", help="check the bundled native components and exit")
    p.add_argument(
        "--download-speech-model", action="store_true",
        help="download the speech model for subtitles (about 1.6 GB, once) and exit",
    )
    p.add_argument("-o", "--output-dir", metavar="OUTDIR", help="write results here (default: next to each original)")
    p.add_argument("--strength", choices=list(STRENGTHS), default="max", help="noise reduction strength (default: max)")
    p.add_argument("--no-polish", action="store_true", help="skip level matching, compression and loudness normalisation")
    p.add_argument(
        "--audio-format", choices=list(AUDIO_CHOICES), default="same",
        help="format for audio files (video keeps its container); default: same as input",
    )
    p.add_argument(
        "--loudness", choices=list(LOUDNESS_TARGETS), default=DEFAULT_LOUDNESS,
        help="loudness target of the polish step: podcast -16 LUFS, youtube -14, broadcast -23, keep (default: podcast)",
    )
    p.add_argument(
        "--subtitles", choices=list(SUBTITLE_MODES), default="off",
        help="also write subtitles: srt, vtt, txt (next to the result) or embed (into a video; audio files get an srt)",
    )
    p.add_argument(
        "--language", choices=["auto", *LANGUAGES], default="auto",
        help="language spoken for the subtitles (default: auto-detect)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)

    if args.version or args.selftest or args.cli or args.download_speech_model:
        from . import console

        console.ensure_output()
    if args.version:
        print(f"{config.APP_NAME} {__version__}")
        return 0
    if args.selftest:
        from . import cli

        return cli.selftest()
    if args.download_speech_model:
        from . import cli

        return cli.download_speech_model()
    if args.cli:
        if not args.files:
            parser.error("--cli needs at least one input file")
        from . import cli

        options = JobOptions(
            strength=args.strength, polish=not args.no_polish, audio_format=args.audio_format,
            loudness=args.loudness, subtitles=args.subtitles, language=args.language,
        )
        return cli.run(args.files, args.output_dir, options)

    from .gui.app import run

    return run(args.files)


if __name__ == "__main__":
    sys.exit(main())

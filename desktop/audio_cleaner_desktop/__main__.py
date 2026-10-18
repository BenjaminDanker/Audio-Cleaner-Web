from __future__ import annotations

import argparse
import os
import sys

# numpy's own thread pool (OpenBLAS/OpenMP/MKL) would only waste CPU next to the app's explicitly threaded work
# (onnxruntime, CTranslate2 and libDF get their thread counts passed in): one thread, set before numpy is imported
# (the engine import below pulls it in). An explicit setting by the user still wins.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from . import quiet_popen  # noqa: E402

quiet_popen.install()  # first, before the heavier imports below (the frozen app does this in packaging/launcher.py)

from . import __version__, config  # noqa: E402
from .engine import (  # noqa: E402
    AUDIO_CHOICES, AUDIO_ENDINGS, AUTO_STRENGTH, DEFAULT_LOUDNESS, LANGUAGES, LOUDNESS_TARGETS, STRENGTH_CHOICES, SUBTITLE_MODES,
    VIDEO_ENDINGS,
    VIDEO_SIZES, JobOptions,
)


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
    p.add_argument("--gui", action="store_true", help="with --selftest: also build the window, the preview and the About box")
    p.add_argument(
        "--download-models", "--download-speech-model", action="store_true", dest="download_models",
        help="download the models for noise removal, subtitles, restoring highs and rebuilding the voice (about 2.6 GB, once) and exit",
    )
    p.add_argument("-o", "--output-dir", metavar="OUTDIR", help="write results here (default: next to each original)")
    p.add_argument(
        "--strength", choices=list(STRENGTH_CHOICES), default=AUTO_STRENGTH,
        help="noise reduction strength; auto (default) picks the one that fits each file from how noisy it is",
    )
    p.add_argument("--no-leveling", action="store_true", help="skip level matching, compression and loudness normalisation")
    p.add_argument(
        "--audio-format", choices=list(AUDIO_CHOICES), default="same",
        help="format for audio files (video keeps its container); default: same as input",
    )
    p.add_argument(
        "--output-name", metavar="NAME",
        help="the file name of the result (one file only); its ending says the type, e.g. talk.mp3 saves just the sound "
        "(default: <name>-cleaned plus the usual ending)",
    )
    p.add_argument(
        "--format", choices=[e.lstrip(".") for e in (*AUDIO_ENDINGS, *VIDEO_ENDINGS)], dest="output_format",
        help="the type of the results: a sound type (just the sound, also from a video), or a video type (a sound "
        "becomes a video with a black picture); default: the type of the original (--audio-format for audio files)",
    )
    p.add_argument(
        "--video-size", choices=list(VIDEO_SIZES), default="same",
        help="picture size of video results: the pixels of the shorter side, or same as the original (default); "
        "bigger than the original only stretches the picture",
    )
    p.add_argument(
        "--loudness", choices=list(LOUDNESS_TARGETS), default=DEFAULT_LOUDNESS,
        help="loudness target of the leveling step: podcast -16 LUFS, youtube -14, broadcast -23, keep (default: podcast)",
    )
    p.add_argument(
        "--subtitles", choices=list(SUBTITLE_MODES), default="off",
        help="also write subtitles: srt, vtt, txt (next to the result) or embed (into a video; audio files get an srt)",
    )
    p.add_argument(
        "--language", choices=["auto", *LANGUAGES], default="auto",
        help="language spoken for the subtitles (default: auto-detect)",
    )
    p.add_argument(
        "--noise", choices=["auto", "light", "heavy"], default="auto",
        help="kind of background noise: auto measures each file (default), light is hiss or hum, heavy is a cafe or traffic",
    )
    p.add_argument(
        "--echo-track", action="store_true",
        help="remove speaker echo using the file's second audio track (files with one track are cleaned as usual)",
    )
    p.add_argument("--no-restore-highs", action="store_true", help="do not restore the missing highs of phone-quality recordings")
    p.add_argument(
        "--rebuild-voice", action="store_true",
        help="rebuild the voice instead of filtering noise: also removes room echo; slow, needs its model",
    )
    p.add_argument(
        "--no-auto-rebuild", action="store_true",
        help="do not rebuild the voice of files that sound echoey or muffled (by default they are, when its model is installed)",
    )
    p.add_argument(
        "--parallel", type=int, choices=range(0, 5), default=1, metavar="N",
        help="files cleaned at once, 1-4, or 0 for Automatic (default: 1); never more than the memory allows",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # The app goes online only to download a model when asked to and (setting on) to look for an update, with its own
    # code (updater.py, models.py): keep the libraries it uses from doing so.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if sys.platform == "win32":
        quiet_popen.warm_platform_info()  # `platform` is asked at import time by onnxruntime: answer it off the main thread
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)

    if args.version or args.selftest or args.cli or args.download_models:
        from . import console

        console.ensure_output()
    if args.version:
        print(f"{config.APP_NAME} {__version__}")
        return 0
    if args.selftest:
        from . import cli

        return cli.selftest(gui=args.gui)
    if args.download_models:
        from . import cli

        return cli.download_models()
    if args.cli:
        if not args.files:
            parser.error("--cli needs at least one input file")
        from . import cli

        options = JobOptions(
            strength=args.strength, leveling=not args.no_leveling, audio_format=args.audio_format,
            loudness=args.loudness, subtitles=args.subtitles, language=args.language, noise=args.noise,
            echo_track=args.echo_track, restore_highs=not args.no_restore_highs, rebuild_voice=args.rebuild_voice,
            auto_rebuild=not args.no_auto_rebuild,
            video_size=args.video_size,
        )
        if args.output_name is not None and len(args.files) != 1:
            parser.error("--output-name works for one file at a time; use --format to choose the type of several")
        return cli.run(args.files, args.output_dir, options, args.parallel, args.output_name, args.output_format)

    from .gui.app import run

    return run(args.files)


if __name__ == "__main__":
    sys.exit(main())

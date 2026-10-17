# PyInstaller spec for Audio Cleaner (onedir, PyInstaller 6.x). Run from anywhere:
#
#   pyinstaller desktop/packaging/audio_cleaner.spec --distpath DIST --workpath BUILD --noconfirm
#
# Environment:
#   ACW_DESKTOP_VENDOR  directory made by prepare_vendor.py   (default desktop/vendor)
#   ACW_ICON_DIR        directory made by make_icons.py        (icon.ico on Windows, icon.icns on macOS)
#
# Result: DIST/AudioCleaner/ (Windows, Linux) or DIST/Audio Cleaner.app (macOS), with Qt, FFmpeg and libDF as
# separate files so they stay replaceable (LGPL), and the licence notices in a "licenses" folder.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

SPEC_DIR = Path(SPECPATH).resolve()  # noqa: F821 - provided by PyInstaller
DESKTOP = SPEC_DIR.parent
ROOT = DESKTOP.parent
VENDOR = Path(os.environ.get("ACW_DESKTOP_VENDOR") or DESKTOP / "vendor").resolve()
ICON_DIR = Path(os.environ["ACW_ICON_DIR"]).resolve() if os.environ.get("ACW_ICON_DIR") else None

sys.path.insert(0, str(DESKTOP))
from audio_cleaner_desktop import __version__  # noqa: E402
from audio_cleaner_desktop.config import APP_NAME, BUNDLE_ID  # noqa: E402

EXE_NAME = "AudioCleaner"


def need(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"audio_cleaner.spec: {path} is missing; run desktop/packaging/prepare_vendor.py first")
    return path


# ---- what goes in -------------------------------------------------------------------------------------
binaries = []
for f in sorted(need(VENDOR / "native").iterdir()):
    binaries.append((str(f), "native"))  # libdf.*, ffmpeg[.exe], ffprobe[.exe] -> <app>/native/

# Only the two small noise models ship (DeepFilterNet3 8 MB, DPDFNet-8 15 MB); the other models (subtitles,
# restoring highs, rebuilding the voice: 2.5 GB) are downloaded by the installer's setup (installer_models.py)
# or by the app (models.py), so the installer stays small. Refuse to bundle anything else.
_shipped = sorted(p.name for p in need(VENDOR / "models").iterdir())
if _shipped != ["DeepFilterNet3_onnx.tar.gz", "dpdfnet8_48khz_hr_batch.onnx"]:
    raise SystemExit(f"audio_cleaner.spec: vendor/models must only hold the two noise models, found {_shipped}")

datas = [
    (str(need(VENDOR / "models")), "models"),
    (str(need(VENDOR / "licenses")), "licenses"),
    # loaded at run time through the package path (About dialog, window icon)
    (str(DESKTOP / "audio_cleaner_desktop" / "assets"), "audio_cleaner_desktop/assets"),
]
hiddenimports = [
    # the updater checks the signature of update.json with cryptography's Ed25519 (a compiled Rust module + cffi)
    "cryptography.hazmat.bindings._rust", "cryptography.hazmat.primitives.asymmetric.ed25519", "_cffi_backend",
]
if sys.platform == "win32":
    # `platform` imports _wmi inside a try: block; keep it, or platform.uname() (onnxruntime and truststore ask for it
    # at import time) falls back to running `cmd /c ver`, which flashes a console window.
    hiddenimports.append("_wmi")

# faster-whisper: the Silero VAD model (assets/silero_vad_v6.onnx); CTranslate2: its shared libraries.
datas += collect_data_files("faster_whisper")
ct2_datas, ct2_binaries, ct2_hidden = collect_all("ctranslate2")
datas += ct2_datas
binaries += ct2_binaries
hiddenimports += ct2_hidden
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("livekit")  # its native library carries WebRTC's echo canceller (acw_common.aec)

# Qt modules the app does not use. Nothing imports them; excluding keeps the hooks from pulling them
# (and their GPL-only third-party parts) in if the build environment has PySide6_Addons installed.
UNUSED_QT = [
    "QtWebEngineCore", "QtWebEngineWidgets", "QtWebEngineQuick", "QtWebChannel", "QtWebSockets", "QtWebView",
    "Qt3DCore", "Qt3DRender", "Qt3DInput", "Qt3DLogic", "Qt3DAnimation", "Qt3DExtras", "QtQuick3D",
    "QtQuick", "QtQuickWidgets", "QtQuickControls2", "QtQml", "QtQmlCore", "QtQuickTemplates2",
    "QtPdf", "QtPdfWidgets", "QtCharts", "QtDataVisualization", "QtGraphs", "QtGraphsWidgets",
    "QtSql", "QtTest", "QtBluetooth", "QtPositioning", "QtLocation", "QtSensors", "QtSerialPort", "QtSerialBus",
    "QtDesigner", "QtUiTools", "QtHelp", "QtNfc", "QtRemoteObjects", "QtScxml", "QtStateMachine",
    "QtTextToSpeech", "QtSpatialAudio", "QtHttpServer", "QtNetworkAuth", "QtCanvasPainter", "QtAxContainer",
    "QtOpenGL", "QtOpenGLWidgets", "QtSvgWidgets", "QtXml", "QtPrintSupport", "QtConcurrent",
]
excludes = [f"PySide6.{m}" for m in UNUSED_QT] + [
    "tkinter", "_tkinter", "turtle",
    "av",  # PyAV bundles GPL x264/x265; replaced by rthook_av_stub.py
    "pytest", "setuptools", "pkg_resources", "distutils", "IPython", "matplotlib", "PIL", "sympy",
    # cffi's helper for building C extensions (imports distutils); the app only loads cffi's compiled modules
    # (through cryptography), never builds any.
    "cffi._shimmed_dist_utils",
]

a = Analysis(  # noqa: F821
    [str(SPEC_DIR / "launcher.py")],
    pathex=[str(DESKTOP), str(ROOT / "common")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(SPEC_DIR / "rthook_av_stub.py")],
    excludes=excludes,
    noarchive=False,
)

# Qt pieces that must not ship: the Qt Virtual Keyboard input plugin is GPL-only (it is collected when
# PySide6_Addons is installed), and Quick/QML/PDF/WebEngine/3D are not used by this widgets application.
DROP = ("virtualkeyboard", "qt6quick", "qtquick", "qt6qml", "qtqml", "qmltooling", "qt6pdf", "qtpdf", "webengine", "qt63d", "qt3d")
a.binaries = [b for b in a.binaries if not any(d in b[0].lower().replace("-", "") for d in DROP)]
a.datas = [d for d in a.datas if not any(x in d[0].lower() for x in DROP)]
left = [b[0] for b in [*a.binaries, *a.datas] if "virtualkeyboard" in b[0].lower()]
if left:
    raise SystemExit(f"audio_cleaner.spec: GPL-only Qt Virtual Keyboard files are still in the bundle: {left}")

pyz = PYZ(a.pure)  # noqa: F821


# ---- Windows file properties (Explorer > Properties > Details) ------------------------------------------
version_file = None
if sys.platform == "win32":
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
    )

    nums = tuple((int(p) for p in (__version__.split(".") + ["0", "0", "0"])[:3])) + (0,)
    version_file = VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums, mask=0x3F, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("CompanyName", "Audio Cleaner"),
                StringStruct("FileDescription", APP_NAME),
                StringStruct("FileVersion", __version__),
                StringStruct("InternalName", EXE_NAME),
                StringStruct("LegalCopyright", "Copyright (c) 2026 Audio Cleaner"),
                StringStruct("OriginalFilename", f"{EXE_NAME}.exe"),
                StringStruct("ProductName", APP_NAME),
                StringStruct("ProductVersion", __version__),
            ])]),
            VarFileInfo([VarStruct("Translation", [0x409, 1200])]),
        ],
    )


def icon_for_platform():
    if ICON_DIR is None:
        return None
    name = {"win32": "icon.ico", "darwin": "icon.icns"}.get(sys.platform)
    return str(need(ICON_DIR / name)) if name else None


exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXE_NAME,
    console=False,
    upx=False,
    strip=False,
    icon=icon_for_platform(),
    version=version_file,
    bootloader_ignore_signals=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(  # noqa: F821
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name=EXE_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{APP_NAME}.app",
        icon=icon_for_platform(),
        bundle_identifier=BUNDLE_ID,
        version=__version__,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleExecutable": EXE_NAME,
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "14.0",  # PySide6 6.12 and onnxruntime 1.31 wheels need macOS 14
            "LSApplicationCategoryType": "public.app-category.video",
            "NSHumanReadableCopyright": "Copyright (c) 2026 Audio Cleaner",
            # "Open With" and dropping files on the Dock icon: Finder only offers the app for these types.
            # Alternate rank: Audio Cleaner is offered but never becomes the default player.
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Audio and video files",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": ["public.audio", "public.movie"],
            }],
        },
    )

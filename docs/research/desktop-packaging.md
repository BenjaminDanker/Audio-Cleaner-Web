# Desktop packaging research: PySide6 + LGPL FFmpeg + code signing

Researched 2026-10-14 from the sources linked inline. Not legal advice; confirm the Qt/LGPL points with counsel. Nothing was downloaded, installed or run. Items marked UNVERIFIED were not confirmed in a primary source.

## 1. Qt / PySide6 licensing

**Licence options.** PySide6 6.12.0 (released Oct 8, 2026) is licensed "LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only" ([PyPI PySide6-Essentials](https://pypi.org/project/PySide6-Essentials/)). A commercial Qt licence is available: "For commercial licensees, all modules are available under a single, commercial Qt license" ([Qt Licensing](https://doc.qt.io/qt-6/licensing.html)). Qt for Python "contains some code that is not provided under the GNU Lesser General Public License", and its third-party components must be acknowledged ([Licenses Used in Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html)).

**LGPLv3 obligations for a closed-source app** (Qt's summary, [Qt LGPL obligations](https://www.qt.io/licensing/open-source-lgpl-obligations)):
- Dynamic linking: "it is possible, but not mandatory, to keep application source code proprietary."
- "The user is allowed to change and re-link the library", and must be able to run the relinked binary.
- Ship the LGPL licence text and "a prominent notice about using the LGPL library".
- Deliver the complete corresponding source of Qt/PySide, or "a written offer with instructions on how to get the source code".
- Provide "sufficient installation information" for relinked binaries.
- Static linking can take the app out of "work that uses the library".
- Application stores may have rules that conflict with the LGPL.

The statutory basis is [GNU LGPL v3 section 4(d)(1)](https://www.gnu.org/licenses/lgpl-3.0.html): "Use a suitable shared library mechanism for linking with the Library."

*Inference (confirm with counsel):* PyInstaller onefile packs Qt into one executable that is extracted to a temp folder on each start ([PyInstaller operating mode](https://pyinstaller.org/en/stable/operating-mode.html)), which makes replacing Qt awkward. Ship onedir, with Qt DLLs/.so/.dylib as separate files, inside an installer, and include the notices and a written source offer.

**Modules.**
- GPLv3 only in the open-source route (avoid unless buying commercial): Graphs, Quick 3D, Quick 3D Physics, Quick Timeline, Canvas Painter, CoAP, GRPC, HTTP Server, Lottie Animation, MQTT, Network Authorization, QML Compiler, Virtual Keyboard, Wayland Compositor ([Qt Licensing](https://doc.qt.io/qt-6/licensing.html)).
- Qt Charts: "commercial licenses and GPLv3", no LGPL option ([Qt Charts 6.12](https://doc.qt.io/qt-6/qtcharts-index.html)).
- Qt Data Visualization: not listed on Qt's licence page; forum replies say GPL/commercial only (UNVERIFIED; avoid).
- GPL-only third-party code inside some modules: BlueZ in Qt Bluetooth (GPLv2-only), glslang in Qt Shader Tools (GPLv3+ with exception; Qt Quick pulls this in, so check), Linux perf in Qt Test ([Licenses used in Qt](https://doc.qt.io/qt-6/licenses-used-in-qt.html)).
- **Qt Multimedia is available under LGPL.** "Since Qt 5.6, these free software licenses are GNU Lesser General Public License, version 3", alongside GPL v2 and commercial, on [Qt Multimedia 6.8.9](https://doc.qt.io/qt-6.8/qtmultimedia-index.html) and [6.12.0](https://doc.qt.io/qt-6/qtmultimedia-index.html). Its FFmpeg backend is LGPL 2.1+ ([Licenses used in Qt](https://doc.qt.io/qt-6/licenses-used-in-qt.html)), so the app then carries a second FFmpeg with its own notices. Qt also warns that H.264 "may be covered by patents" and "The Qt licenses do not cover such fees" ([Qt Multimedia 6.12](https://doc.qt.io/qt-6/qtmultimedia-index.html)).

## 2. FFmpeg

**Build types.** BtbN: "lgpl: Omits GPL-only libraries, most notably libx264 and libx265"; "nonfree: Adds fdk-aac to the gpl set." Targets are win64, win32, linux64 and linuxarm64, with no macOS target. Auto-builds run only for win(arm)64 and linux(arm)64 ([BtbN FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)).

**Windows x64, LGPL, prebuilt.** The release "Latest Auto-Build (2026-10-13 13:05)" includes `ffmpeg-n9.0-latest-win64-lgpl-9.0.zip`, `ffmpeg-n8.1-latest-win64-lgpl-8.1.zip` and their `-shared` variants, plus `checksums.sha256` ([BtbN releases](https://github.com/BtbN/FFmpeg-Builds/releases)). The "latest" in these names is a rolling build, so pin an autobuild tag and verify against `checksums.sha256`. The hashes were not downloaded or checked here.

**macOS arm64 and x64: no verified LGPL prebuilt.**
- evermeet.cx: Intel only; configured with `--enable-gpl`; GPG-signed; no SHA256 mentioned. GPL, so not usable for LGPL ([evermeet.cx](https://evermeet.cx/ffmpeg/)).
- martin-riedl.de: arm64 and amd64 builds with per-file SHA256 files; "macOS binaries are signed (zip and installer) and notarized (installer)". Its page lists x264, x265, LAME and opus, so the builds are probably GPL, and the licence per build is not stated. Intel builds end January 2027 ([martin-riedl](https://ffmpeg.martin-riedl.de/)). Check `ffmpeg -buildconf` before use.
- Homebrew: licence "GPL-3.0-or-later", depends on x264 and x265 ([Homebrew formula](https://formulae.brew.sh/formula/ffmpeg)). GPL.
- **Recommendation:** build an LGPL FFmpeg in CI for macOS. Configure without `--enable-gpl` and `--enable-nonfree`, and add `--enable-libmp3lame --enable-libopus --enable-libvorbis`. BtbN's build scripts show the library recipes (`50-libmp3lame.sh`, `50-libopus.sh`, `45-libvorbis.sh`; [BtbN scripts.d](https://github.com/BtbN/FFmpeg-Builds/tree/master/scripts.d)). Confirm the result with `ffmpeg -buildconf`.

**Licence rule.** "FFmpeg is licensed under the GNU Lesser General Public License (LGPL) version 2.1", and "If those parts get used the GPL applies to all of FFmpeg" ([FFmpeg legal](https://ffmpeg.org/legal.html)).

**Encoders.**
- Lost in an LGPL build: libx264 and libx265 (GPL) ([BtbN](https://github.com/BtbN/FFmpeg-Builds)). libfdk_aac needs `--enable-nonfree`, and the FFmpeg checklist says to compile without it for LGPL ([FFmpeg legal](https://ffmpeg.org/legal.html)). The codecs page says it "is also incompatible with GPL" ([FFmpeg codecs](https://ffmpeg.org/ffmpeg-codecs.html)).
- Still available: native `aac`, "the default AAC encoder, natively implemented into FFmpeg" ([FFmpeg codecs](https://ffmpeg.org/ffmpeg-codecs.html)); libopus (three-clause BSD, [Opus licence](https://opus-codec.org/license/)); libvorbis (BSD, [Xiph](https://xiph.org/vorbis/)); libmp3lame, which the FFmpeg checklist names as an LGPL external library ([FFmpeg legal](https://ffmpeg.org/legal.html)).

**Obligations for shipping `ffmpeg.exe` alongside the app** ([FFmpeg legal checklist](https://ffmpeg.org/legal.html)):
- Distribute the source of FFmpeg, "no matter if you modified it or not", matching the binary exactly, and host it next to the binary or make a written offer.
- Put the configure line in a text file in the source root.
- Show LGPL notices on the download page and in the About box; mention FFmpeg in the EULA.
- Don't misspell FFmpeg, don't obfuscate DLL names, and remove any reverse-engineering ban from the EULA.
- If you ever link the libav* libraries via ctypes, link dynamically. *Inference:* running ffmpeg.exe as a child process keeps it a separate program, but the source and notice duties still apply.

## 3. Packaging

- **PyInstaller.** Dual-licensed GPL 2.0 with an exception. "The executable bundles generated by PyInstaller from your source code can be shipped with whatever license you want" ([PyInstaller licence](https://pyinstaller.org/en/stable/license.html)). Onefile starts more slowly, and "Do not give administrator privileges to a one-file executable on Windows" ([operating mode](https://pyinstaller.org/en/stable/operating-mode.html)). On macOS, `--codesign-identity` signs the collected binaries, ad-hoc otherwise. UPX fails on dylibs, and onefile does not work with sandboxing ([usage](https://pyinstaller.org/en/stable/usage.html)). Qt's deployment page lists PyInstaller among the tools that meet its cross-platform requirement ([Qt for Python deployment](https://doc.qt.io/qtforpython-6/deployment/index.html)).
- **Nuitka.** The README says AGPLv3 with a runtime exception in `LICENSE-RUNTIME` ([Nuitka](https://github.com/Nuitka/Nuitka)). Qt's table lists Nuitka as MIT, which conflicts; read the LICENSE file. Supports Python 3.4 to 3.14. One community comparison: Nuitka output is smaller, PyInstaller builds are faster ([pyside6-nuitka-test](https://github.com/Erriez/pyside6-nuitka-test)). Reported PySide6 deploy issues include `.webp` data files and a macOS QtSql deploy block ([pyside-setup deploy tool](https://git.omada.cafe/Nova/pyside-setup/src/tag/v6.11.1/sources/pyside-tools/deploy.py)).
- **Briefcase.** Targets: macOS `.app`, Windows MSI, Linux native package ([Briefcase](https://briefcase.beeware.org/en/stable/)). Notarization resumes after interruption and supports `--no-wait` ([Briefcase 0.4.3](https://newreleases.io/project/github/beeware/briefcase/release/v0.4.3)). PySide6 on macOS needs `min_os_version = "12.0"` ([Briefcase release notes](https://briefcase.beeware.org/en/v0.3.24/about/releases.html)). A Qt/PySide6 backend was not confirmed (UNVERIFIED).
- **Size.** PySide6_Essentials 6.12.0 wheels: 82.2 MB (Windows amd64), 116.0 MB (macOS universal2) ([PyPI](https://pypi.org/project/PySide6-Essentials/)). The shipped app will be larger once FFmpeg and Python are added. Not measured.
- **Startup.** Not measured. The PyInstaller docs say onefile is slower to start than onedir ([operating mode](https://pyinstaller.org/en/stable/operating-mode.html)).
- **PySide6 6.8+ issues.** No confirmed frozen-app crash found. One forum report: a frameless-window border under PySide 6.11 with PyInstaller 6.19, which went away after reverting to 6.10.2 ([forum](https://forum.pythonguis.com/t/possible-regression-in-pyside-6-11-with-windowflags-frameless-and-pyinstaller/2144)).

**Recommendation:** PyInstaller 6.x in onedir mode, wrapped in an installer (Inno Setup or WiX; not researched). Use Nuitka only if size matters. Briefcase has MSI and notarization, but no PySide6 precedent was confirmed.

## 4. Code signing

**macOS.**
- Apple Developer Program: "$99 annual membership". It includes Developer ID distribution and notarization ([Apple Developer Program](https://developer.apple.com/programs/)).
- The Developer ID certificate must be generated by the Account Holder ([Developer ID](https://developer.apple.com/developer-id/)).
- Workflow: `codesign --options runtime --timestamp`, then `xcrun notarytool submit --wait`, then `xcrun stapler staple` ([Flavio Copes](https://flaviocopes.com/apple-developer-free-vs-paid/); [Apple notarization](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution), whose body did not render here). Sign every nested `.dylib` and `.so` in the PyInstaller bundle.

**Windows.**
- "The simplest way to avoid SmartScreen warnings is to publish through the Microsoft Store" ([Microsoft SmartScreen](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation)).
- "EV certificates no longer bypass SmartScreen", so paying for EV to avoid warnings "is no longer justified" (same page).
- Reputation builds per file or publisher and "can take several weeks and hundreds of clean installs" (same page). Unsigned files start from zero on every version. Windows 11 Smart App Control can block unsigned files without positive reputation (same page).
- **Artifact Signing** (formerly Trusted Signing): "Starts at $9.99/month", and it works in GitHub Actions (same page). Basic is $9.99 per account per month with 5,000 signatures ([Artifact Signing SKU](https://learn.microsoft.com/en-us/azure/artifact-signing/how-to-change-sku)). Public Trust individual developers "must be located in the United States or Canada" and need an Individual billing account ([Artifact Signing quickstart](https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart)). A paid Azure subscription is required; free and trial subscriptions are not supported ([Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/6020935/artifact-signing)).
- **OV certificates.** Since mid-2023 the key must be on hardware or a cloud HSM. Resellers list Sectigo OV at about $219/yr ([SSL Dragon](https://www.ssldragon.com/ssl-certificates/code-signing/)) and $220 to $399 elsewhere ([sslinsights](https://sslinsights.com/sectigo-code-signing-certificate-guide/)). SSL.com's cloud signing entry tier is listed at $180/yr ([SSL.com eSigner](https://www.ssl.com/guide/esigner-pricing-for-code-signing/)). These are reseller figures that disagree; verify before buying.
- **EV** is not worth it for SmartScreen.

**Cheapest legitimate path for a solo developer:**
- US or Canada resident: Artifact Signing Basic ($119.88/yr) plus Apple ($99/yr), about $219/yr, plus a paid Azure subscription.
- Anywhere else: OV via cloud HSM (about $180 to $400/yr) plus Apple ($99/yr), about $280 to $500/yr. Expect SmartScreen warnings until reputation builds.
- Do not ship unsigned Windows builds.

## 5. Auto-update

- **Sparkle (macOS).** "open source software update framework for macOS". It recommends EdDSA signatures and to "Notarize and code sign the application via Apple's Developer ID program (if possible)". Programmatic setup is documented for non-Cocoa toolkits ([Sparkle docs](https://sparkle-project.org/documentation/)). Use from Python would need a PyObjC bridge (UNVERIFIED).
- **WinSparkle (Windows).** MIT-licensed, C API, "bind it from another language" ([WinSparkle](https://winsparkle.org/)). No Python example, and update signing is not described there (UNVERIFIED).
- **Velopack.** MIT; supports "Windows, OSX, and Linux" ([Velopack GitHub](https://github.com/velopack/velopack)). Python is not among its named languages. `vpk pack` takes `--packId`, `--packVersion`, `--packDir` and `--mainExe`, and `--mainExe` is "primarily a Windows concept" ([Velopack packaging](https://docs.velopack.io/packaging/overview)). Updates can be hosted "anywhere static files can be served" (Velopack GitHub page). Plausible for a PyInstaller onedir folder, untested.
- **PyUpdater.** Archived by its owner on Sep 26, 2022, read-only ([PyUpdater](https://github.com/JMSwag/PyUpdater)). Do not use.
- **DIY.** A version JSON on HTTPS, with the installer's SHA256 checked before the user runs it, and a download page. Simplest for v1.

## Recommendations

1. **Qt:** PySide6 under LGPLv3. Ship onedir inside an installer, with Qt as separate files, the LGPL text, an About-box notice and a written source offer. Avoid the GPL-only modules listed above. Qt Multimedia is usable. Get counsel's review.
2. **FFmpeg, Windows:** BtbN `lgpl` build, pinned to an autobuild tag and verified with `checksums.sha256`.
3. **FFmpeg, macOS:** build an LGPL FFmpeg in CI and verify with `-buildconf`. Use native `aac`, libopus, libmp3lame and libvorbis. Drop x264, x265 and fdk-aac.
4. **Packaging:** PyInstaller 6.x onedir plus installer.
5. **Signing:** Apple Developer ID ($99/yr), notarize and staple. Windows: Artifact Signing if US or Canada, otherwise OV via cloud HSM. No EV.
6. **Updates:** DIY version JSON with checksums and a download page for v1.

## Open items (not verified)

- Briefcase's Qt/PySide6 backend; Inno Setup and WiX; measured startup time and bundle size.
- Actual SHA256 values for the BtbN and martin-riedl files.
- Per-build licence for martin-riedl macOS builds.
- The Nuitka licence conflict between Qt's table and the repository.
- Qt Data Visualization's licence status on official Qt pages.

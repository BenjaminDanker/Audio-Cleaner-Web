<#
Builds the Audio Cleaner installer on Windows 10/11 x64, the same steps as the release build.

    powershell -ExecutionPolicy Bypass -File desktop\packaging\build_windows.ps1 [-WorkDir DIR] [-WhisperModelDir DIR]
                                                                               [-FfmpegDir DIR] [-SkipTests]

Needs: Python 3.12 (python.org or `winget install Python.Python.3.12`), Git for Windows (provides Git Bash),
Rust with the MSVC toolchain (https://rustup.rs, plus Visual Studio Build Tools "Desktop development with C++"),
Inno Setup 6 (`winget install JRSoftware.InnoSetup`) and MSYS2 (`winget install MSYS2.MSYS2`, used once to build FFmpeg
natively with mingw-w64; skip it with -FfmpegDir pointing at an FFmpeg built by build_ffmpeg.sh on Linux).

Result: <WorkDir>\out\AudioCleaner-Setup-x64.exe
Code signing (a certificate in the Windows store, signtool) is not scripted here: the installer is built unsigned.
#>
param(
    [string]$WorkDir,
    [string]$WhisperModelDir,
    [string]$FfmpegDir,
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repo
if (-not $WorkDir) { $WorkDir = Join-Path $repo 'build\windows' }
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$work = (Resolve-Path $WorkDir).Path

function Step($t) { Write-Host "== $t" -ForegroundColor Cyan }
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe failed with exit code $LASTEXITCODE" }
}
function Find-Tool($name, [string[]]$candidates) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    return $null
}

Step 'prerequisites'
$py = Find-Tool 'py' @()
$pyArgs = @('-3.12')
if (-not $py) { $py = Find-Tool 'python' @(); $pyArgs = @() }
if (-not $py) { throw 'Python 3.12 is missing (winget install Python.Python.3.12)' }
if ((& $py @pyArgs -c 'import sys; print("%d.%d" % sys.version_info[:2])') -ne '3.12') { throw 'Python 3.12 is required' }
$gitBash = Find-Tool 'bash' @("$env:ProgramFiles\Git\bin\bash.exe")
if (-not $gitBash -or $gitBash -like '*\System32\*') { $gitBash = "$env:ProgramFiles\Git\bin\bash.exe" }
if (-not (Test-Path $gitBash)) { throw 'Git for Windows (Git Bash) is missing (winget install Git.Git)' }
if (-not (Find-Tool 'cargo' @("$env:USERPROFILE\.cargo\bin\cargo.exe"))) { throw 'Rust is missing (https://rustup.rs)' }
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$iscc = Find-Tool 'iscc' @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")
if (-not $iscc) { throw 'Inno Setup 6 is missing (winget install JRSoftware.InnoSetup)' }
$msysBash = 'C:\msys64\usr\bin\bash.exe'
if (-not $FfmpegDir -and -not (Test-Path $msysBash)) { throw 'MSYS2 is missing (winget install MSYS2.MSYS2) - or pass -FfmpegDir' }

# Git Bash gets the repository path in its own notation.
function ToUnix($p) { (& $gitBash -lc "cygpath -u '$($p -replace '\\','/')'").Trim() }

Step 'libDF (DeepFilterNet C library)'
$env:BUILD_DIR = ToUnix (Join-Path $work 'deepfilter')
$env:OUT = ToUnix (Join-Path $work 'libdf')
Run $gitBash @('-lc', "cd '$(ToUnix $repo)' && bash native/deepfilter/build.sh native")
Remove-Item Env:\BUILD_DIR, Env:\OUT

Step 'FFmpeg (LGPL, built from source with mingw-w64 in MSYS2)'
if ($FfmpegDir) {
    $ffmpeg = (Resolve-Path $FfmpegDir).Path
} else {
    $ffmpeg = Join-Path $work 'ffmpeg'
    $stamp = (Get-FileHash desktop\packaging\build_ffmpeg.sh -Algorithm SHA256).Hash.Substring(0, 16)
    if (-not (Test-Path (Join-Path $ffmpeg ".stamp-$stamp"))) {
        if (Test-Path $ffmpeg) { Remove-Item -Recurse -Force $ffmpeg }
        $env:MSYSTEM = 'MINGW64'; $env:CHERE_INVOKING = '1'
        $env:DL_DIR_WIN = Join-Path $work 'dl'
        $cmd = "pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-pkgconf mingw-w64-x86_64-nasm make diffutils perl tar xz curl && " +
               "cd `"`$(cygpath -u '$repo')`" && REQUIRE_ASM=1 DL_DIR=`"`$(cygpath -u '$($env:DL_DIR_WIN)')`" " +
               "bash desktop/packaging/build_ffmpeg.sh windows-x86_64 `"`$(cygpath -u '$ffmpeg')`""
        Run $msysBash @('-lc', $cmd)
        New-Item -ItemType File -Path (Join-Path $ffmpeg ".stamp-$stamp") | Out-Null
    }
}

Step 'Python environment'
$venv = Join-Path $work 'venv'
if (-not (Test-Path "$venv\Scripts\python.exe")) { Run $py ($pyArgs + @('-m', 'venv', $venv)) }
$vpy = "$venv\Scripts\python.exe"
Run $vpy @('-m', 'pip', 'install', '-q', '--upgrade', 'pip')
Run $vpy @('-m', 'pip', 'install', '-q', '--require-hashes', '-r', 'desktop\requirements.lock', '-r', 'desktop\requirements-build.lock')

Step 'vendor directory'
$vendor = Join-Path $work 'vendor'
$icons = Join-Path $work 'icons'
$env:PYTHONPATH = "$repo\desktop;$repo\common"
$env:ACW_DESKTOP_VENDOR = $vendor
$env:ACW_ICON_DIR = $icons
$whisper = if ($WhisperModelDir) { @('--whisper-model', $WhisperModelDir) } else { @('--download-whisper') }
Run $vpy (@('desktop\packaging\prepare_vendor.py', '--libdf', "$work\libdf\df.dll", '--ffmpeg-dir', $ffmpeg,
    '--deepfilter-src', "$work\deepfilter", '--download-noise-models',
    '--rust-target', 'x86_64-pc-windows-msvc', '--out', $vendor) + $whisper)

if (-not $SkipTests) {
    Step 'tests'
    $env:QT_QPA_PLATFORM = 'offscreen'
    Run $vpy @('-m', 'pytest', '-q', 'common\tests', 'desktop\tests')
}

Step 'app folder'
Run $vpy @('desktop\packaging\make_icons.py', $icons)
$dist = Join-Path $work 'dist'
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
Run "$venv\Scripts\pyinstaller.exe" @('desktop\packaging\audio_cleaner.spec', '--distpath', $dist, '--workpath', (Join-Path $work 'pyinstaller'), '--noconfirm')

Step 'models for the frozen selftest'
# The app downloads its models like any user (the noise removal models are required); only the frozen selftest needs them here.
$env:ACW_DESKTOP_MODELS = Join-Path $work 'models'
Run $vpy @('-c', 'from audio_cleaner_desktop import models, resources; models.download(resources.NOISE_MODEL)')

Step 'frozen selftest'
# The exe is windowed: without a console it writes audio-cleaner-cli.log in the temp folder.
$log = Join-Path $env:TEMP 'audio-cleaner-cli.log'
Remove-Item $log -ErrorAction SilentlyContinue
$p = Start-Process -FilePath "$dist\AudioCleaner\AudioCleaner.exe" -ArgumentList '--selftest', '--gui' -Wait -PassThru
if (Test-Path $log) { Get-Content $log }
if ($p.ExitCode -ne 0) { throw "selftest exited with code $($p.ExitCode)" }
if ((Test-Path $log) -and -not (Select-String -Path $log -Pattern 'selftest ok' -Quiet)) { throw 'selftest did not report ok' }

Step 'installer'
$version = (& $vpy -c "from audio_cleaner_desktop import __version__ as v; print(v, end='')")
$out = Join-Path $work 'out'
New-Item -ItemType Directory -Force -Path $out | Out-Null
# Setup downloads the models (2.6 GB, file by file from Hugging Face): the list is generated from the app's catalog (needs no numpy).
Run $vpy @('desktop\packaging\installer_models.py', "$work\models.iss")
Run $iscc @("/DAppVersion=$version", "/DSourceDir=$dist\AudioCleaner", "/DOutputDir=$out", "/DIconFile=$icons\icon.ico", "/DModelsInclude=$work\models.iss", 'desktop\packaging\windows\AudioCleaner.iss')
Copy-Item "$out\AudioCleaner-Setup-$version-x64.exe" "$out\AudioCleaner-Setup-x64.exe" -Force
Step "done: $out\AudioCleaner-Setup-x64.exe"

#!/usr/bin/env python3
"""Make the installers download the models during setup.

    python installer_models.py OUT.iss                Windows: Inno Setup include (iscc /DModelsInclude=OUT.iss)
    python installer_models.py --mac SCRIPTS_DIR      macOS: the .pkg's preinstall and postinstall scripts (make_pkg.sh)
    python installer_models.py --total-gb             size of all the downloads in GB (for the texts)

Generated from the app's own catalog (resources.SPEECH_MODEL, HIGHS_MODEL, VOICE_MODEL; read without importing
numpy, so a bare python3 is enough), so the installer and the app always agree on URLs, sizes and SHA-256.

The speech model comes as individual files: the installer checks every file's SHA-256 (Inno Setup's download page
does) and installs it into {app}\\models\\<id>\\, where the app looks (models.location). The two ONNX models (restoring
highs, rebuilding the voice) come in ONE zip archive (resources.MODELS_ARCHIVE): setup downloads it once (SHA-256
checked), unpacks it with Windows' own tar.exe (Windows 10 1803 and later; without it setup leaves the models to
the app's Settings), moves each file to its model's folder after checking its size, puts the archive's LICENSES
folder next to the models, and deletes the archive. Files that a previous install already put there with the right
size are not fetched again, so updating the app does not download 2.5 GB each time. A model whose files are all
there gets a .complete marker with the hashes of its files, which the app compares with its catalog
(models.location). On a Mac the .pkg's postinstall script does the same into
/Library/Application Support/Audio Cleaner/models/<id>/ (models.installer_dir), unpacking with ditto. If a download
fails (offline), setup still finishes; the app then offers the missing download in Settings.

Running setup again over an installed app (same or newer version) downloads nothing that is already complete: a model
whose files are all there with the right sizes is left alone (and its marker rewritten); a marker that names other
hashes than this catalog (a model of another version) makes setup forget that model's files and fetch it again; a
partial earlier install is completed (individual files one by one, the archive again if any of its files is missing
or has the wrong size). Nothing in the models folder is ever deleted otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "common")]

from audio_cleaner_desktop import models, resources  # noqa: E402

CATALOG = (resources.SPEECH_MODEL, resources.HIGHS_MODEL, resources.VOICE_MODEL)
BS = chr(92)
ARCHIVE_TMP = "acw-models.zip"  # the archive's name in setup's temporary folder


def _pascal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _app_path(m, f) -> str:
    """Where setup puts file `f` of model `m`, as an Inno path."""
    return "{app}" + BS + "models" + BS + m.id + BS + f.path.replace("/", BS)


def _archive(catalog):
    """The one archive the catalog's models share (None when there is none)."""
    found = {m.archive for m in catalog if m.archive is not None}
    if len(found) > 1:
        raise ValueError("the installers handle one models archive")
    return next(iter(found), None)


def _plain(catalog):
    """(model, file) of everything that is downloaded as an individual file."""
    return [(m, f) for m in catalog if m.archive is None for f in m.files]


def _archived(catalog):
    """(model, file) of everything that comes out of the archive."""
    return [(m, f) for m in catalog if m.archive is not None for f in m.files]


def total_bytes(catalog=CATALOG) -> int:
    """Everything setup downloads: the individual files and the archive (once)."""
    return models.download_bytes(catalog)


def model_names() -> str:
    """The models in words, for the texts the installers show."""
    return "subtitles, phone-quality recordings and rebuilding voices"


def render(catalog=CATALOG) -> str:
    archive = _archive(catalog)
    plain = _plain(catalog)
    archived = _archived(catalog)
    total = total_bytes(catalog)
    files = []
    fetches = []
    for n, (m, f) in enumerate(plain):
        tmp = f"acw-model-{n}"
        rel = Path(f.path)
        dest = "{app}\\models\\" + m.id + ("\\" + str(rel.parent).replace("/", "\\") if str(rel.parent) != "." else "")
        full = dest + "\\" + rel.name
        files.append(
            f'Source: "{{tmp}}\\{tmp}"; DestDir: "{dest}"; DestName: "{rel.name}"; ExternalSize: {f.size}; '
            "Flags: external skipifsourcedoesntexist ignoreversion deleteafterinstall"
        )
        fetches.append(f"    FetchModel({_pascal(f.url)}, {_pascal(tmp)}, {_pascal(f.sha256)}, {_pascal(full)}, {f.size});")
    markers = []
    stale = []
    for m in catalog:
        marker_path = _pascal("{app}" + BS + "models" + BS + m.id + BS + ".complete")
        expected = "#10".join(_pascal(f.sha256) for f in m.files)
        drops = "\n".join(
            f"    DeleteFile(ExpandConstant({_pascal(_app_path(m, f))}));" for f in m.files
        )
        stale.append(
            f"  if not MarkerAgrees(ExpandConstant({marker_path}), {expected}) then\n  begin\n{drops}\n"
            f"    DeleteFile(ExpandConstant({marker_path}));\n  end;"
        )
    for m in catalog:
        ready = " and ".join(f"ModelFileReady(ExpandConstant({_pascal(_app_path(m, f))}), {f.size})" for f in m.files)
        hashes = "#10".join(_pascal(f.sha256) for f in m.files)
        marker = _pascal("{app}" + BS + "models" + BS + m.id + BS + ".complete")
        markers.append(f"  if {ready} then\n    SaveStringToFile(ExpandConstant({marker}), {hashes}, False);")
    needed = chr(10).join(
        f"  if not ModelFileReady(ExpandConstant({_pascal(_app_path(m, f))}), {f.size}) then Result := Result + {f.size};"
        for m, f in plain
    )
    setup_extra = ""
    archive_code = ""
    archive_fetch = ""
    archive_step = ""
    need_lines = ["  Result := 0;", needed]
    if archive is not None:
        unpacked = sum(f.size for _, f in archived) + archive.unpacked_size
        setup_extra = (
            "[Setup]\n; The models unpacked from the archive. The archive itself waits in the temporary folder.\n"
            f"ExtraDiskSpaceRequired={unpacked}\n\n"
        )
        missing = chr(10).join(
            f"  if not ModelFileReady(ExpandConstant({_pascal(_app_path(m, f))}), {f.size}) then Result := True;"
            for m, f in archived
        )
        places = [
            f"    PlaceModelFile(Work + {_pascal(BS + f.member.replace('/', BS))}, ExpandConstant({_pascal(_app_path(m, f))}), {f.size});"
            for m, f in archived
        ] + [
            "    PlaceModelFile(Work + " + _pascal(BS + f.member.replace("/", BS)) + ", ExpandConstant("
            + _pascal("{app}" + BS + "models" + BS + f.path.replace("/", BS)) + f"), {f.size});"
            for f in archive.shared
        ]
        need_lines.append(f"  if ArchiveNeeded then Result := Result + {archive.size};")
        archive_code = f"""
{{ True when a file that comes out of the models archive is not installed yet. }}
function ArchiveNeeded: Boolean;
begin
  Result := False;
{missing}
end;

{{ Windows' own tar.exe (Windows 10 1803 and later) unpacks zip files. Not the first tar on the PATH: that may be a GNU tar
  that cannot read a drive letter. }}
function TarPath: String;
begin
  Result := ExpandConstant('{{win}}\\System32\\tar.exe');
  if not FileExists(Result) then
    Result := ExpandConstant('{{win}}\\Sysnative\\tar.exe');
  if not FileExists(Result) then
    Result := '';
end;

procedure FetchArchive;
begin
  if ModelsCancelled or not ArchiveNeeded then
    Exit;
  if TarPath = '' then
  begin
    ModelsFailed := ModelsFailed + 1;
    Log('tar.exe is not available: the models archive was not downloaded');
    Exit;
  end;
  ModelDownloads.Clear;
  ModelDownloads.Add({_pascal(archive.url)}, {_pascal(ARCHIVE_TMP)}, {_pascal(archive.sha256)});
  try
    ModelDownloads.Download;
    ArchiveDownloaded := True;
  except
    if ModelDownloads.AbortedByUser then
      ModelsCancelled := True
    else
    begin
      ModelsFailed := ModelsFailed + 1;
      Log('Could not download {archive.url}: ' + GetExceptionMessage);
    end;
  end;
end;

{{ Move one unpacked file to where the app looks for it, after checking its size. }}
procedure PlaceModelFile(const Src, Dest: String; Size: Int64);
begin
  if ModelFileReady(Dest, Size) then
    Exit;
  if not ModelFileReady(Src, Size) then
  begin
    Log('Not unpacked correctly: ' + Src);
    ModelsFailed := ModelsFailed + 1;
    Exit;
  end;
  ForceDirectories(ExtractFilePath(Dest));
  DeleteFile(Dest);
  if not RenameFile(Src, Dest) then
  begin
    Log('Could not move ' + Src + ' to ' + Dest);
    ModelsFailed := ModelsFailed + 1;
  end;
end;

procedure UnpackModels;
var
  Zip, Work: String;
  Code, FailedBefore: Integer;
begin
  if not ArchiveDownloaded then
    Exit;
  FailedBefore := ModelsFailed;
  Zip := ExpandConstant({_pascal("{tmp}" + BS + ARCHIVE_TMP)});
  Work := ExpandConstant('{{app}}\\models\\.unpack');
  DelTree(Work, True, True, True);
  ForceDirectories(Work);
  WizardForm.StatusLabel.Caption := 'Unpacking the models...';
  if Exec(TarPath, '-xf "' + Zip + '" -C "' + Work + '"', '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0) then
  begin
{chr(10).join(places)}
  end
  else
  begin
    ModelsFailed := ModelsFailed + 1;
    Log('tar.exe could not unpack the models archive (exit code ' + IntToStr(Code) + ')');
  end;
  DelTree(Work, True, True, True);
  DeleteFile(Zip);
  if ModelsFailed > FailedBefore then
    SuppressibleMsgBox('Not every model could be unpacked. Setup will finish without them. '
      + 'Open Audio Cleaner > Settings to finish the download.', mbInformation, MB_OK, IDOK);
end;
"""
        archive_fetch = "    FetchArchive;\n"
        archive_step = "  UnpackModels;\n"
    return f"""; Generated by desktop/packaging/installer_models.py: do not edit by hand.

{setup_extra}[Files]
{chr(10).join(files)}

[UninstallDelete]
Type: filesandordirs; Name: "{{app}}\\models"

[Code]
var
  ModelDownloads: TDownloadWizardPage;
  ModelsFailed: Integer;
  ModelsCancelled: Boolean;
  ArchiveDownloaded: Boolean;

function ModelFileReady(const Path: String; Size: Int64): Boolean;
var
  Got: Int64;
begin
  Result := FileSize64(Path, Got) and (Got = Size);
end;

{{ A model's .complete marker lists the hashes of the files it was installed from. No marker (an install from before
  the marker existed, or nothing installed): the sizes decide. A marker that lists other hashes belongs to another
  version of the model. }}
function MarkerAgrees(const Marker, Expected: String): Boolean;
var
  Got: AnsiString;
  Content: String;
begin
  Result := True;
  if not FileExists(Marker) then
    Exit;
  Result := False;
  if not LoadStringFromFile(Marker, Got) then
    Exit;
  Content := Trim(String(Got));
  StringChangeEx(Content, #13, '', True);
  Result := (Content = Expected);
end;

{{ Models of another version are removed first, so that everything below fetches them again. }}
procedure ForgetStaleModels;
begin
{chr(10).join(stale)}
end;

procedure InitializeWizard;
begin
  ModelDownloads := CreateDownloadPage('Downloading Audio Cleaner''s models',
    'The models for {model_names()} ({total / 1e9:.1f} GB, once). '
    + 'After this, everything works offline.', nil);
end;

{{ Each file on its own: one that fails does not stop the others. }}
procedure FetchModel(const Url, TempName, Sha256, Dest: String; Size: Int64);
begin
  if ModelsCancelled or ModelFileReady(ExpandConstant(Dest), Size) then
    Exit;
  ModelDownloads.Clear;
  ModelDownloads.Add(Url, TempName, Sha256);
  try
    ModelDownloads.Download;
  except
    if ModelDownloads.AbortedByUser then
      ModelsCancelled := True
    else
    begin
      ModelsFailed := ModelsFailed + 1;
      Log('Could not download ' + Url + ': ' + GetExceptionMessage);
    end;
  end;
end;
{archive_code}
function ModelsStillNeeded: Int64;
begin
{chr(10).join(need_lines)}
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Need, Free, Total: Int64;
begin
  Result := True;
  if CurPageID <> wpReady then
    Exit;
  ForgetStaleModels;
  Need := ModelsStillNeeded;
  if Need = 0 then
    Exit;  {{ everything is installed already (an update): nothing to download }}
  {{ The downloads wait in the temporary folder until the files are installed. }}
  if GetSpaceOnDisk64(ExpandConstant('{{tmp}}'), Free, Total) and (Free < Need + 200000000) then
  begin
    SuppressibleMsgBox('There is not enough free space on the drive for temporary files to download the models ('
      + Format('%.1f', [Need / 1.0e9]) + ' GB needed, ' + Format('%.1f', [Free / 1.0e9]) + ' GB free). '
      + 'Setup will finish without them. Free up some space, then open Audio Cleaner > Settings to download them.',
      mbInformation, MB_OK, IDOK);
    Exit;
  end;
  ModelsFailed := 0;
  ModelsCancelled := False;
  ArchiveDownloaded := False;
  ModelDownloads.Show;
  try
{archive_fetch}{chr(10).join(fetches)}
  finally
    ModelDownloads.Hide;
  end;
  if ModelsCancelled then
    SuppressibleMsgBox('The download of the models was cancelled. Setup will finish without them. '
      + 'Open Audio Cleaner > Settings to download them later.', mbInformation, MB_OK, IDOK)
  else if ModelsFailed > 0 then
    SuppressibleMsgBox('Not every model could be downloaded. Setup will finish without them. '
      + 'Open Audio Cleaner > Settings to finish the download.', mbInformation, MB_OK, IDOK);
end;

{{ A model counts as installed only when all its files are there; this marker says which version of it. }}
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;
{archive_step}{chr(10).join(markers)}
end;
"""


MAC_MODELS = "/Library/Application Support/Audio Cleaner/models"
APP_BUNDLE = "Audio Cleaner.app"


def render_mac(catalog=CATALOG) -> str:
    archive = _archive(catalog)
    fetches = [f'fetch "{f.url}" "{MAC_MODELS}/{m.id}/{f.path}" {f.size} {f.sha256}' for m, f in _plain(catalog)]
    finishes = []
    forgets = []
    for m in catalog:
        specs = " ".join(f'"{f.sha256}:{f.size}:{f.path}"' for f in m.files)
        finishes.append(f'finish "{MAC_MODELS}/{m.id}" {specs}')
        forgets.append(f'forget_stale "{MAC_MODELS}/{m.id}" {specs}')
    archive_block = ""
    if archive is not None:
        archived = _archived(catalog)
        checks = "\n".join(f'have "{MAC_MODELS}/{m.id}/{f.path}" {f.size} || need_archive=1' for m, f in archived)
        places = "\n".join(
            f'      place "$WORK/{f.member}" "{MAC_MODELS}/{m.id}/{f.path}" {f.size}' for m, f in archived
        ) + "\n" + "\n".join(f'      place "$WORK/{f.member}" "{MAC_MODELS}/{f.path}" {f.size}' for f in archive.shared)
        archive_block = f"""
# The ONNX models and their licence texts come in one archive: download it once, check its SHA-256, unpack it
# with ditto into a work folder, move every file to its model's folder after checking its size, delete the archive.
ARCHIVE="{MAC_MODELS}/.{archive.name}.download"
WORK="{MAC_MODELS}/.unpack"
need_archive=0
{checks}
if [ "$need_archive" = 1 ]; then
  mkdir -p "{MAC_MODELS}"
  rm -f "$ARCHIVE"
  if curl -fL --proto '=https' --retry 3 --connect-timeout 30 --speed-limit 1000 --speed-time 60 --max-time 3600 -C - \\
       -o "$ARCHIVE" "{archive.url}" >>"$LOG" 2>&1 \\
     && [ "$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)" = "{archive.sha256}" ]; then
    rm -rf "$WORK"; mkdir -p "$WORK"
    if ditto -x -k "$ARCHIVE" "$WORK" >>"$LOG" 2>&1 || /usr/bin/unzip -q -o "$ARCHIVE" -d "$WORK" >>"$LOG" 2>&1; then
{places}
    else
      echo "could not unpack {archive.name}" >>"$LOG"; failed=1
    fi
  else
    echo "could not download {archive.url}" >>"$LOG"; failed=1
  fi
  rm -rf "$WORK" "$ARCHIVE"
fi
"""
    total = total_bytes(catalog)
    return f"""#!/bin/bash
# postinstall of the Audio Cleaner .pkg: download the models ({total / 1e9:.1f} GB, once).
# Generated by desktop/packaging/installer_models.py: do not edit by hand. Runs as root during setup.
# Every download is checked against its SHA-256 before it is used; files already there with the right
# size are kept (updates do not download again). A failed download does not fail the installation: the app
# offers it again in Settings. A model that is complete gets a .complete marker (the hashes of its files).
LOG=/var/log/audio-cleaner-install.log
failed=0
have() {{  # have FILE SIZE: the file is there with that size
  [ -f "$1" ] && [ "$(stat -f%z "$1")" = "$2" ]
}}
fetch() {{
  local url="$1" dest="$2" size="$3" sha="$4"
  if have "$dest" "$size"; then return 0; fi
  mkdir -p "$(dirname "$dest")"
  local tmp="$dest.download"
  if curl -fL --proto '=https' --retry 3 --connect-timeout 30 --speed-limit 1000 --speed-time 60 --max-time 3600 -C - \\
       -o "$tmp" "$url" >>"$LOG" 2>&1 \\
     && [ "$(shasum -a 256 "$tmp" | cut -d' ' -f1)" = "$sha" ]; then
    mv -f "$tmp" "$dest"; chmod 644 "$dest"
  else
    rm -f "$tmp"; echo "could not download $url" >>"$LOG"; failed=1
  fi
}}
place() {{  # place SRC DEST SIZE: move an unpacked file to its place when it has the right size
  local src="$1" dest="$2" size="$3"
  if have "$dest" "$size"; then return 0; fi
  if have "$src" "$size"; then
    mkdir -p "$(dirname "$dest")"; mv -f "$src" "$dest"; chmod 644 "$dest"
  else
    echo "not unpacked correctly: $src" >>"$LOG"; failed=1
  fi
}}
forget_stale() {{  # forget_stale DIR SHA:SIZE:PATH...  a .complete marker naming other hashes: another version's files go
  local dir="$1" spec rel want got
  shift
  [ -f "$dir/.complete" ] || return 0   # no marker (an older install, or none yet): the sizes decide
  want="$(for spec in "$@"; do echo "${{spec%%:*}}"; done)"
  got="$(cat "$dir/.complete")"
  [ "$got" = "$want" ] && return 0
  for spec in "$@"; do rel="${{spec#*:*:}}"; rm -f "$dir/$rel"; done
  rm -f "$dir/.complete"
}}
finish() {{  # finish DIR SHA:SIZE:PATH...  writes DIR/.complete when every file is there with its size
  local dir="$1" ok=1 spec rel size marker=""
  shift
  for spec in "$@"; do
    rel="${{spec#*:*:}}"; size="${{spec#*:}}"; size="${{size%%:*}}"
    if have "$dir/$rel" "$size"; then marker="$marker${{spec%%:*}}"$'\\n'; else ok=0; fi
  done
  if [ "$ok" = 1 ]; then printf '%s' "$marker" > "$dir/.complete"; chmod 644 "$dir/.complete"; fi
}}
{chr(10).join(forgets)}
{chr(10).join(fetches)}
{archive_block}{chr(10).join(finishes)}
chmod -R a+rX "{MAC_MODELS}" 2>/dev/null
[ "$failed" = 0 ] || echo "some models are missing; Audio Cleaner offers them in Settings" >>"$LOG"
exit 0
"""


def render_mac_preinstall() -> str:
    """The .pkg's preinstall script: the installer replaces the app bundle completely, so an update cannot leave
    files of the old version inside it. Removes that one folder (never the models or anything else)."""
    return f"""#!/bin/bash
# preinstall of the Audio Cleaner .pkg: remove the installed app so that the new one replaces it completely
# (nothing of an older version stays inside the bundle). Only this one folder is removed: the models in
# /Library/Application Support/Audio Cleaner, the settings and the history are not touched.
# Generated by desktop/packaging/installer_models.py: do not edit by hand. Runs as root during setup.
# $3 is the volume the package is installed on ("/" for the startup disk).
LOG=/var/log/audio-cleaner-install.log
VOLUME="${{3:-/}}"
[ -d "$VOLUME" ] || VOLUME=/
APP="${{VOLUME%/}}/Applications/{APP_BUNDLE}"
if [ -d "$APP" ] && [ ! -L "$APP" ]; then
  echo "removing the installed $APP before the update" >>"$LOG"
  rm -rf "$APP" >>"$LOG" 2>&1 || echo "could not remove $APP" >>"$LOG"
fi
exit 0
"""


def main() -> int:
    args = sys.argv[1:]
    if args == ["--total-gb"]:
        print(f"{total_bytes() / 1e9:.1f}")
        return 0
    if len(args) == 2 and args[0] == "--mac":
        out = Path(args[1])
        out.mkdir(parents=True, exist_ok=True)
        script = out / "postinstall"
        script.write_text(render_mac(), encoding="utf-8")
        script.chmod(0o755)
        pre = out / "preinstall"
        pre.write_text(render_mac_preinstall(), encoding="utf-8")
        pre.chmod(0o755)
        return 0
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    Path(args[0]).write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

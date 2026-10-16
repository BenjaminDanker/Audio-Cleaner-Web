; Inno Setup 6 script for the Audio Cleaner installer.
;
;   iscc /DAppVersion=2.0.0 /DSourceDir=C:\path\to\dist\AudioCleaner [/DOutputDir=C:\out] [/DIconFile=C:\path\icon.ico]
;        [/DModelsInclude=C:\path\models.iss] AudioCleaner.iss
;   (models.iss from installer_models.py: setup then downloads the models, 2.5 GB: the speech model's files and one zip
;   archive with the two ONNX models, which setup unpacks with Windows' own tar.exe. Without it nothing is downloaded
;   during setup; the app then offers the downloads in Settings.)
;
; SourceDir is the PyInstaller output folder (the one that contains AudioCleaner.exe and _internal\).
; Installs per user by default (no administrator prompt); the "Install for all users" choice on the first
; page switches to Program Files. Settings, history and models the app downloaded itself live in the user profile; the
; uninstaller asks whether to remove them too (and leaves them when the uninstall is silent).
;
; Running setup again (same or newer version) updates in place: it reuses the previous install's folder and mode (per
; user or all users), closes the running app, deletes the old program files (_internal) before copying the new ones,
; and downloads no model that is already complete in {app}\models (installer_models.py). {app}\models, the settings
; (registry), the history and the models the app downloaded itself are never touched by an update.
;
; The app updates itself by running this setup silently over the install (audio_cleaner_desktop/updater.py):
;   AudioCleaner-x.y.z-Windows.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART [/CURRENTUSER or /ALLUSERS] [/RELAUNCH=1]
; A silent setup shows no model page and downloads no model. /RELAUNCH=1 ("Restart now" in the app) makes a silent
; setup start the new version when it is done, as the user who started setup, not elevated; without it nothing starts.

#ifndef AppVersion
  #error Pass /DAppVersion=x.y.z
#endif
#ifndef SourceDir
  #error Pass /DSourceDir=<PyInstaller dist\AudioCleaner folder>
#endif
#ifndef OutputDir
  #define OutputDir "Output"
#endif

#define AppName "Audio Cleaner"
#define AppExe "AudioCleaner.exe"

[Setup]
; Never change AppId: it is how upgrades and the uninstaller recognise this product.
AppId={{6F0D1A52-3C41-4B7E-9A68-2D5E8B7C41F3}
; An update installs where the previous setup installed (folder, per user or all users) without asking again.
; These are Inno's defaults; they are written out so that nobody switches them off by accident.
UsePreviousAppDir=yes
UsePreviousPrivileges=yes
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Benjamin Danker
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile=..\EULA.txt
OutputDir={#OutputDir}
OutputBaseFilename=AudioCleaner-Setup-{#AppVersion}-x64
#ifdef IconFile
SetupIconFile={#IconFile}
#endif
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/fast
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Everything the build puts in the program folder besides the exe is in _internal (PyInstaller 6): delete it before the new
; files are copied, so files an older version had and this one does not cannot linger (and cannot be loaded by mistake).
; NEVER add {app}\models here: that folder holds the downloaded models (it is not {app}\_internal\models, the two
; small noise models that ship inside the program).
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

; The models (subtitles, highs restoration, rebuilding the voice): downloaded during setup, see installer_models.py.
#ifdef ModelsInclude
  #include ModelsInclude
#endif

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
; The app's own update ("Restart now"): the silent setup starts the new version.
Filename: "{app}\{#AppExe}"; Flags: nowait runasoriginaluser; Check: RelaunchAfterSilentUpdate

[Code]
{ Only the app's own silent update asks for this (/RELAUNCH=1): a silent install by hand starts nothing. }
function RelaunchAfterSilentUpdate: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');
end;

{ The models and history the app made in the user's profile are not part of the install log: ask before removing them. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  Data := ExpandConstant('{localappdata}\Audio Cleaner\Audio Cleaner');
  if UninstallSilent or not DirExists(Data) then
    Exit;
  if MsgBox('Also remove the models Audio Cleaner downloaded (up to 2.6 GB), your history and your settings?',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
  begin
    DelTree(Data, True, True, True);
    RegDeleteKeyIncludingSubkeys(HKCU, 'Software\Audio Cleaner');
  end;
end;

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

[Code]
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

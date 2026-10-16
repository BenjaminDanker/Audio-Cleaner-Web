; Inno Setup 6 script for the Audio Cleaner OBS plugin installer (Windows).
;
;   iscc /DAppVersion=1.0.42 /DSourceDir=C:\path\to\release\RelWithDebInfo\audio-cleaner [/DOutputDir=C:\out] audio-cleaner-obs.iss
;
; SourceDir is the plugin folder from `cmake --install` (the one holding bin\64bit\audio-cleaner.dll and data\).
; It installs that folder as C:\ProgramData\obs-studio\plugins\audio-cleaner, where OBS 28 and newer look for
; plugins of every user, creating the plugins folder when it isn't there. No administrator prompt: Windows lets any
; user create folders there, and the person who created it owns it, so the plugin's own updater can replace it later.
; The uninstaller lives in the user's AppData (not in the plugin folder, which the updater replaces as a whole).

#ifndef AppVersion
  #error Pass /DAppVersion=x.y.z
#endif
#ifndef SourceDir
  #error Pass /DSourceDir=<the installed audio-cleaner plugin folder>
#endif
#ifndef OutputDir
  #define OutputDir "Output"
#endif

#define AppName "Audio Cleaner OBS plugin"

[Setup]
; Never change AppId: it is how reinstalls and the uninstaller recognise this product.
AppId={{B2E4C7A1-5D3F-4E9B-8C26-7A1F0D93E5B4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Audio Cleaner
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
DefaultDirName={commonappdata}\obs-studio\plugins\audio-cleaner
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
UninstallFilesDir={localappdata}\Audio Cleaner OBS plugin
UninstallDisplayName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename=audio-cleaner-obs-plugin-{#AppVersion}-windows-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Messages]
FinishedLabel=The Audio Cleaner filter is ready. Start OBS, right-click your microphone source, choose Filters, and add Audio Cleaner.

[InstallDelete]
; a clean copy every time: nothing left over from an older version
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{app}\data"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
function ObsRunning(): Boolean;
var
  Code: Integer;
begin
  // find.exe ends with 0 when tasklist listed obs64.exe
  Result := Exec(ExpandConstant('{cmd}'), '/C tasklist /FI "IMAGENAME eq obs64.exe" /NH | find /I "obs64.exe" >NUL',
    '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

function CloseObsFirst(): Boolean;
begin
  Result := True;
  while ObsRunning() do
    if MsgBox('OBS is open. Please close OBS, then click OK.', mbInformation, MB_OKCANCEL) = IDCANCEL then begin
      Result := False;
      Exit;
    end;
end;

function InitializeSetup(): Boolean;
begin
  Result := WizardSilent() or CloseObsFirst();
end;

function InitializeUninstall(): Boolean;
begin
  Result := UninstallSilent() or CloseObsFirst();
end;

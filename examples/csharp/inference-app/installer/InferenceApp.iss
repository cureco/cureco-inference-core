; Build after publishing the self-contained application. No automatic startup registration.
#ifndef PublishDir
  #error PublishDir must point to the reviewed publish directory
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
[Setup]
AppId={{780F17DA-8300-458F-B2B3-50ADC2755431}
AppName=Cureco Inference
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\Cureco Inference
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=CurecoInference-win-x64-setup
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\CurecoInference.exe
[Files]
Source: "{#PublishDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\Cureco Inference"; Filename: "{app}\CurecoInference.exe"

param(
  [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$runtimePython = "C:\Users\Rasul\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if ($Python) {
  $py = $Python
} elseif (Test-Path $runtimePython) {
  $py = $runtimePython
} else {
  $py = "python"
}
& $py -m PyInstaller --noconfirm --clean --onefile --windowed --name UZYRO `
  --icon "uzyro/assets/branding/uzyro.ico" `
  --exclude-module scipy --exclude-module skimage --exclude-module aggdraw `
  --add-binary "uzyro/assets/native/fribidi-0.dll;." `
  --add-data "uzyro/assets/native/LICENSE.fribidi.txt;licenses" `
  --add-data "uzyro/assets/tool_demos;uzyro/assets/tool_demos" `
  --add-data "uzyro/assets/branding;uzyro/assets/branding" uzyro_launcher.py
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller завершился с кодом $LASTEXITCODE"
}
if (-not (Test-Path ".\dist\UZYRO.exe")) {
  throw "UZYRO.exe не был создан"
}

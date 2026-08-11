$ErrorActionPreference = "Stop"
$runtimePython = "C:\Users\Rasul\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $runtimePython) {
  $py = $runtimePython
} else {
  $py = "python"
}
& $py -m PyInstaller --noconfirm --clean --onefile --windowed --name PhotoRedactor `
  --exclude-module scipy --exclude-module skimage --exclude-module aggdraw `
  --add-data "photoredactor/assets/tool_demos;photoredactor/assets/tool_demos" launcher.py
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller завершился с кодом $LASTEXITCODE"
}
if (-not (Test-Path ".\dist\PhotoRedactor.exe")) {
  throw "PhotoRedactor.exe не был создан"
}

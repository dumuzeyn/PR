$ErrorActionPreference = "Stop"
$runtimePython = "C:\Users\Rasul\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $runtimePython) {
  $py = $runtimePython
} else {
  $py = "python"
}
& $py -m PyInstaller --noconfirm --clean --onefile --windowed --name PhotoRedactor `
  --add-data "photoredactor/assets/tool_demos;photoredactor/assets/tool_demos" launcher.py

$ErrorActionPreference = "Stop"
$runtimePython = "C:\Users\Rasul\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $runtimePython) {
  $py = $runtimePython
} else {
  $py = "python"
}
& $py -m PyInstaller --noconfirm --onefile --windowed --name PhotoRedactor --collect-all cv2 --collect-all PIL --collect-all numpy launcher.py

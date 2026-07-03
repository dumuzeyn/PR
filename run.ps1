$ErrorActionPreference = "Stop"
$runtimePython = "C:\Users\Rasul\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $runtimePython) {
  & $runtimePython -m photoredactor
} else {
  python -m photoredactor
}

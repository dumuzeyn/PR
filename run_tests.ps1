param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$testFiles = Get-ChildItem (Join-Path $root "tests") -Filter "test_*.py" | Sort-Object Name
$uiFiles = @(
  $testFiles | Where-Object {
    Select-String -LiteralPath $_.FullName -Pattern 'PhotoRedactorApp\(\)|tk\.Tk\(\)' -Quiet
  }
)
$uiPaths = @($uiFiles | ForEach-Object { $_.FullName })
$corePaths = @(
  $testFiles | Where-Object { $_.FullName -notin $uiPaths } | ForEach-Object { $_.FullName }
)

Write-Host "Core test files: $($corePaths.Count)"
& $Python -m pytest -q @corePaths
if ($LASTEXITCODE -ne 0) {
  throw "Core tests failed with exit code $LASTEXITCODE"
}

Write-Host "UI test files: $($uiPaths.Count)"
$collected = & $Python -m pytest --collect-only -q @uiPaths
if ($LASTEXITCODE -ne 0) {
  throw "UI test collection failed with exit code $LASTEXITCODE"
}
$uiTests = @($collected | Where-Object { $_ -match '::' -and $_ -notmatch '^\s' })
foreach ($test in $uiTests) {
  & $Python -m pytest -q $test
  if ($LASTEXITCODE -ne 0) {
    throw "UI test failed: $test"
  }
}

Write-Host "All tests passed: core files $($corePaths.Count), isolated UI tests $($uiTests.Count)"

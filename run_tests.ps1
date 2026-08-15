param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$testFiles = Get-ChildItem (Join-Path $root "tests") -Filter "test_*.py" | Sort-Object Name
$uiFiles = @(
  $testFiles | Where-Object {
    Select-String -LiteralPath $_.FullName -Pattern 'UZYROApp\(\)|tk\.Tk\(\)' -Quiet
  }
)
$uiPaths = @($uiFiles | ForEach-Object { $_.FullName })
$corePaths = @(
  $testFiles | Where-Object { $_.FullName -notin $uiPaths } | ForEach-Object { $_.FullName }
)

Write-Host "Core test files: $($corePaths.Count)"
$coreOutput = & $Python -m pytest -q @corePaths 2>&1
$coreExit = $LASTEXITCODE
$coreOutput | Write-Output
if ($coreExit -ne 0) {
  if ($env:GITHUB_ACTIONS -eq "true") {
    $detail = ($coreOutput -join "`n").Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
    Write-Output "::error title=Ошибка основных тестов::$detail"
    "## Ошибка основных тестов" | Add-Content $env:GITHUB_STEP_SUMMARY
    $coreOutput | Add-Content $env:GITHUB_STEP_SUMMARY
  }
  throw "Core tests failed with exit code $coreExit"
}

Write-Host "UI test files: $($uiPaths.Count)"
$collected = & $Python -m pytest --collect-only -q @uiPaths
if ($LASTEXITCODE -ne 0) {
  throw "UI test collection failed with exit code $LASTEXITCODE"
}
$uiTests = @($collected | Where-Object { $_ -match '::' -and $_ -notmatch '^\s' })
foreach ($test in $uiTests) {
  $uiOutput = & $Python -m pytest -q $test 2>&1
  $uiExit = $LASTEXITCODE
  $uiOutput | Write-Output
  if ($uiExit -ne 0) {
    if ($env:GITHUB_ACTIONS -eq "true") {
      $detail = ($uiOutput -join "`n").Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
      Write-Output "::error title=Ошибка UI-теста ${test}::$detail"
      "## Ошибка UI-теста: $test" | Add-Content $env:GITHUB_STEP_SUMMARY
      $uiOutput | Add-Content $env:GITHUB_STEP_SUMMARY
    }
    throw "UI test failed: $test"
  }
}

Write-Host "All tests passed: core files $($corePaths.Count), isolated UI tests $($uiTests.Count)"

param(
  [string]$Python = "python",
  [ValidateSet("Full", "Fast", "Core", "Ui", "Security", "Smoke", "Performance")]
  [string]$Mode = "Full",
  [switch]$Coverage
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Invoke-Pytest {
  param(
    [string]$Label,
    [string]$Marker,
    [switch]$AppendCoverage,
    [string[]]$Targets = @()
  )

  $arguments = @("-m", "pytest", "-q", "-m", $Marker)
  $arguments += $Targets
  if ($Coverage) {
    $arguments += "--cov=uzyro", "--cov-branch", "--cov-report=term-missing"
    if ($AppendCoverage) {
      $arguments += "--cov-append"
    }
  }
  Write-Host "`n[$Label] pytest -m `"$Marker`""
  $output = & $Python @arguments 2>&1
  $exitCode = $LASTEXITCODE
  $output | Write-Output
  if ($exitCode -ne 0) {
    if ($env:GITHUB_ACTIONS -eq "true") {
      $detail = ($output -join "`n").Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
      Write-Output "::error title=$Label::$detail"
      if ($env:GITHUB_STEP_SUMMARY) {
        "## $Label" | Add-Content $env:GITHUB_STEP_SUMMARY
        $output | Add-Content $env:GITHUB_STEP_SUMMARY
      }
    }
    throw "$Label failed with exit code $exitCode"
  }
}

function Invoke-UiTests {
  param([switch]$AppendCoverage)

  $collected = & $Python -m pytest --collect-only -q -m $uiMarker 2>&1
  if ($LASTEXITCODE -ne 0) {
    $collected | Write-Output
    throw "UI test collection failed with exit code $LASTEXITCODE"
  }
  $nodes = @(
    $collected |
      Where-Object { $_ -match '::' -and $_ -notmatch '^\s' }
  )
  if ($Coverage) {
    $append = $AppendCoverage.IsPresent
    foreach ($node in $nodes) {
      Invoke-Pytest "UI test: $node" $uiMarker -AppendCoverage:$append -Targets @($node)
      $append = $true
    }
    return
  }

  Write-Host "`n[UI tests] $($nodes.Count) isolated tests, up to 4 in parallel"
  $results = @(
    $nodes | ForEach-Object -Parallel {
      Set-Location $using:root
      $node = $_
      $output = (& $using:Python -m pytest -q $node 2>&1 | Out-String)
      [PSCustomObject]@{
        Node = $node
        ExitCode = $LASTEXITCODE
        Output = $output.TrimEnd()
      }
    } -ThrottleLimit 4
  )
  $failures = @($results | Where-Object { $_.ExitCode -ne 0 })
  if ($failures.Count -gt 0) {
    foreach ($failure in $failures) {
      Write-Output "`n[FAILED] $($failure.Node)"
      Write-Output $failure.Output
    }
    throw "$($failures.Count) isolated UI test(s) failed"
  }
  Write-Host "$($results.Count) UI tests passed."
}

function Invoke-IsolatedUiTests {
  $collected = & $Python -m pytest --collect-only -q -m $isolatedUiMarker 2>&1
  if ($LASTEXITCODE -ne 0) {
    $collected | Write-Output
    throw "Isolated UI collection failed with exit code $LASTEXITCODE"
  }
  $nodes = @($collected | Where-Object { $_ -match '::' -and $_ -notmatch '^\s' })
  foreach ($node in $nodes) {
    $arguments = @("-m", "pytest", "-q", $node)
    if ($Coverage) {
      $arguments += "--cov=uzyro", "--cov-branch", "--cov-report=term-missing", "--cov-append"
    }
    Write-Host "`n[Isolated native UI test] $node"
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
      throw "Isolated UI test failed: $node"
    }
  }
}

try {
  & $Python --version | Out-Null
} catch {
  throw "Python executable was not found: $Python"
}
if ($LASTEXITCODE -ne 0) {
  throw "Python executable was not found: $Python"
}

$coreMarker = "not ui and not security and not performance and not slow and not hardware"
$fastMarker = "smoke or ($coreMarker)"
$uiMarker = "ui and not ui_isolated and not performance"
$isolatedUiMarker = "ui_isolated and not performance"

switch ($Mode) {
  "Smoke" { Invoke-Pytest "Smoke tests" "smoke" }
  "Fast" { Invoke-Pytest "Fast correctness tests" $fastMarker }
  "Core" { Invoke-Pytest "Core tests" $coreMarker }
  "Ui" {
    Invoke-UiTests
    Invoke-IsolatedUiTests
  }
  "Security" { Invoke-Pytest "Security tests" "security" }
  "Performance" { Invoke-Pytest "Performance tests" "performance" }
  "Full" {
    Invoke-Pytest "Core tests" $coreMarker
    Invoke-Pytest "Security tests" "security" -AppendCoverage:$Coverage
    Invoke-UiTests -AppendCoverage:$Coverage
    Invoke-IsolatedUiTests
  }
}

Write-Host "`nUZYRO test mode '$Mode' completed successfully."

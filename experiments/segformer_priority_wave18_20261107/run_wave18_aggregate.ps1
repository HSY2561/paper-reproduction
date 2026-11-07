[CmdletBinding()]
param(
    [string]$Root = "",
    [string]$ResultsDir = "",
    [string]$OutputDir = "",
    [string]$Python = "python",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $Root) { $Root = $PSScriptRoot }

$scriptPath = Join-Path $PSScriptRoot "aggregate_wave18.py"
$arguments = @($scriptPath, "--root", $Root)
if ($ResultsDir) { $arguments += @("--results-dir", $ResultsDir) }
if ($OutputDir) { $arguments += @("--output-dir", $OutputDir) }
if ($Strict) { $arguments += "--strict" }

# This wrapper only invokes the read-only aggregator; it does not call a
# training/evaluation entry point.
& $Python @arguments
exit $LASTEXITCODE

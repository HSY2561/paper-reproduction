$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding
$waveRoot = $PSScriptRoot
Set-Location -LiteralPath $waveRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONPATH = Join-Path $waveRoot 'src\PaddleSeg-2.10.0'
$logPath = Join-Path $waveRoot 'logs\scheduled_queue_20261017.log'
$returnCode = 1
$failureMessage = $null
try {
    & 'C:\venv_bisenet\Scripts\python.exe' (Join-Path $waveRoot 'experiment_queue.py') --run >> $logPath 2>&1
    $returnCode = $LASTEXITCODE
} catch {
    $failureMessage = $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath $logPath -Encoding UTF8
}
[ordered]@{
    ended_at = (Get-Date -Format o)
    exit_code = $returnCode
    error = $failureMessage
    test_used = $false
    log = $logPath
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $waveRoot 'results\scheduled_queue_exit_20261017.json') -Encoding UTF8
exit $returnCode

$ErrorActionPreference = 'Stop'
$lossRoot = $PSScriptRoot
$env:PYTHONPATH = Join-Path $lossRoot 'src\PaddleSeg-2.10.0'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TEMP = Join-Path $lossRoot 'tmp'
$env:TMP = $env:TEMP
(Get-Process -Id $PID).PriorityClass = 'Normal'
$lossCode = $null
$lossError = $null
try {
    $lossArgs = '"' + (Join-Path $lossRoot 'loss_queue.py') + '" --run'
    $lossProcess = Start-Process -FilePath 'C:\venv_bisenet\Scripts\python.exe' -ArgumentList $lossArgs -WorkingDirectory $lossRoot -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $lossRoot 'logs\runner.log') -RedirectStandardError (Join-Path $lossRoot 'logs\runner.stderr.log')
    $lossCode = $lossProcess.ExitCode
    if ($null -eq $lossCode) { $lossError = 'Process exited but ExitCode was unavailable; do not assume success' }
} catch {
    $lossError = $_.Exception.Message
} finally {
    [ordered]@{exit_code=$lossCode;error=$lossError;success=($null -ne $lossCode -and $lossCode -eq 0 -and $null -eq $lossError);ended_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $lossRoot 'results\scheduled_loss_exit.json') -Encoding UTF8
}
if ($null -eq $lossCode -or $null -ne $lossError) { exit 1 }
exit $lossCode

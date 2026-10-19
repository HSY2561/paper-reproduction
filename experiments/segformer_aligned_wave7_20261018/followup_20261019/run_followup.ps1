$ErrorActionPreference = 'Stop'
$followupRoot = $PSScriptRoot
$mainRoot = Split-Path -Parent $followupRoot
$env:PYTHONPATH = Join-Path $mainRoot 'src\PaddleSeg-2.10.0'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TEMP = Join-Path $followupRoot 'tmp'
$env:TMP = $env:TEMP
(Get-Process -Id $PID).PriorityClass = 'Normal'
$followupCode = $null
$followupError = $null
try {
    $followupArgs = '"' + (Join-Path $followupRoot 'followup_queue.py') + '" --run'
    $followupProcess = Start-Process -FilePath 'C:\venv_bisenet\Scripts\python.exe' -ArgumentList $followupArgs -WorkingDirectory $followupRoot -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $followupRoot 'logs\runner.log') -RedirectStandardError (Join-Path $followupRoot 'logs\runner.stderr.log')
    $followupCode = $followupProcess.ExitCode
} catch {
    $followupError = $_.Exception.Message
    $followupCode = 1
} finally {
    [ordered]@{exit_code=$followupCode;error=$followupError;ended_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $followupRoot 'results\scheduled_followup_exit.json') -Encoding UTF8
}
exit $followupCode

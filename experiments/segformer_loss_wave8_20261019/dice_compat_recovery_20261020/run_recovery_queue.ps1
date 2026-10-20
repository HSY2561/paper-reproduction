$ErrorActionPreference = 'Stop'
$recoveryRoot = $PSScriptRoot
$env:PYTHONPATH = Join-Path $recoveryRoot 'src\PaddleSeg-2.10.0'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:LOSS_PREFLIGHT_DEVICE = 'gpu:0'
$env:LOSS_PREFLIGHT_OUTPUT = 'gradient_preflight.json'
$env:TEMP = Join-Path $recoveryRoot 'tmp'
$env:TMP = $env:TEMP
(Get-Process -Id $PID).PriorityClass = 'Normal'
$recoveryCode = $null
$recoveryError = $null
try {
    $recoveryArgs = '"' + (Join-Path $recoveryRoot 'recovery_queue.py') + '" --run'
    $recoveryProcess = Start-Process -FilePath 'C:\venv_bisenet\Scripts\python.exe' -ArgumentList $recoveryArgs -WorkingDirectory $recoveryRoot -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $recoveryRoot 'logs\runner.log') -RedirectStandardError (Join-Path $recoveryRoot 'logs\runner.stderr.log')
    $recoveryCode = $recoveryProcess.ExitCode
    if ($null -eq $recoveryCode) { $recoveryError = 'ExitCode unavailable; not assumed successful' }
} catch {
    $recoveryError = $_.Exception.Message
} finally {
    [ordered]@{exit_code=$recoveryCode;error=$recoveryError;success=($null -ne $recoveryCode -and $recoveryCode -eq 0 -and $null -eq $recoveryError);ended_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $recoveryRoot 'results\scheduled_recovery_exit.json') -Encoding UTF8
}
if ($null -eq $recoveryCode -or $null -ne $recoveryError) { exit 1 }
exit $recoveryCode

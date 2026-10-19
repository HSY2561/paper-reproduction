$ErrorActionPreference = 'Stop'
$waveRoot = $PSScriptRoot
$env:PYTHONPATH = Join-Path $waveRoot 'src\PaddleSeg-2.10.0'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TMP = Join-Path $waveRoot 'tmp'
$env:TEMP = $env:TMP
(Get-Process -Id $PID).PriorityClass = 'Normal'
Set-Location -LiteralPath $waveRoot
$waveCode = $null
$waveError = $null
try {
    $workflowArgs = '"' + (Join-Path $waveRoot 'workflow.py') + '"'
    $workflowProcess = Start-Process -FilePath 'C:\venv_bisenet\Scripts\python.exe' -ArgumentList $workflowArgs -WorkingDirectory $waveRoot -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $waveRoot 'logs\workflow.log') -RedirectStandardError (Join-Path $waveRoot 'logs\workflow.stderr.log')
    $waveCode = $workflowProcess.ExitCode
} catch {
    $waveError = $_.Exception.Message
    $waveCode = 1
} finally {
    [ordered]@{exit_code=$waveCode;error=$waveError;ended_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $waveRoot 'results\scheduled_workflow_exit.json') -Encoding UTF8
}
exit $waveCode

param()
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$pythonPath = 'C:\venv_bisenet\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $experimentRoot 'src\PaddleSeg-2.10.0'
$env:PYTHONIOENCODING = 'utf-8'
(Get-Process -Id $PID).PriorityClass = 'Normal'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $experimentRoot "logs\postprocess_waiter_${stamp}.stdout.log"
$stderrPath = Join-Path $experimentRoot "logs\postprocess_waiter_${stamp}.stderr.log"
$receiptPath = Join-Path $experimentRoot "results\postprocess_waiter_${stamp}_exit.json"
$started = Get-Date
$exitCode = $null
try {
    & $pythonPath (Join-Path $experimentRoot 'postprocess_w11.py') '--wait-and-profile' 1> $stdoutPath 2> $stderrPath
    $exitCode = $LASTEXITCODE
} finally {
    [ordered]@{
        started_at = $started.ToString('o')
        ended_at = (Get-Date).ToString('o')
        exit_code = $exitCode
        runner_pid = $PID
        stdout = $stdoutPath
        stderr = $stderrPath
        training_restarted = $false
        mode = 'wait-and-profile'
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding utf8
}
if ($null -eq $exitCode) { exit 1 }
exit $exitCode

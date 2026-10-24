$ErrorActionPreference = 'Stop'
$root = 'D:\论文复现\experiments\segformer_loss_wave8_20261019\matched_seeds_20261022'
$python = 'C:\venv_bisenet\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $root '..\dice_compat_recovery_20261020\src\PaddleSeg-2.10.0'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $root
& $python (Join-Path $root 'seed_queue.py') --run *> (Join-Path $root 'logs\runner.log')
$code = $LASTEXITCODE
$record = [ordered]@{ exit_code = $code; ended_at = (Get-Date).ToString('o'); test_used = $false }
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'results\runner_exit.json') -Encoding UTF8
exit $code

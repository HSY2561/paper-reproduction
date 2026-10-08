$ErrorActionPreference = 'Continue'
$env:PYTHONPATH = 'D:\论文复现\experiments\retrain_wave4_20261007\src\PaddleSeg-2.10.0'
$py = 'C:\venv_bisenet\Scripts\python.exe'
$src = 'D:\论文复现\experiments\retrain_wave4_20261007\src\PaddleSeg-2.10.0'
$root = 'D:\论文复现\experiments\retrain_wave4_20261007'
$cfg = Join-Path $root 'configs\segformer_uafm_crack500.yml'
$save = Join-Path $root 'training\segformer_uafm_crack500'
$logDir = Join-Path $root 'logs'
$stdout = Join-Path $logDir 'segformer_uafm_crack500_train.stdout.log'
$stderr = Join-Path $logDir 'segformer_uafm_crack500_train.stderr.log'
$exitFile = Join-Path $root 'results\segformer_uafm_crack500_exit.json'
New-Item -ItemType Directory -Force -Path $save,$logDir,(Join-Path $root 'results') | Out-Null
& $py -u (Join-Path $src 'tools\train.py') --config $cfg --save_dir $save --iters 18000 --batch_size 4 --do_eval --use_ema --save_interval 1000 --seed 42 --log_iters 100 --num_workers 0 --keep_checkpoint_max 5 1> $stdout 2> $stderr
$ec = $LASTEXITCODE
[ordered]@{model='segformer_uafm_crack500';exit_code=$ec;ended_at=(Get-Date).ToString('o');iters=18000;seed=42;test_used=$false}|ConvertTo-Json|Set-Content -Encoding UTF8 $exitFile
exit $ec

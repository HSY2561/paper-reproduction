$ErrorActionPreference='Stop'
$env:PYTHONPATH='C:\local_wave1src'
$py='C:\venv_bisenet\Scripts\python.exe'
$cfg='C:\segformer_wave1\configs\S0_source_disjoint.yml'
$save='C:\segformer_wave1\training\S0_seed42'
$log='C:\segformer_wave1\logs\S0_seed42_train.log'
$exit='C:\segformer_wave1\results\S0_seed42_exit.json'
New-Item -ItemType Directory -Force -Path (Split-Path $save),(Split-Path $log),(Split-Path $exit) | Out-Null
& $py -u C:\local_wave1src\tools\train.py --config $cfg --save_dir $save --do_eval --use_ema --save_interval 1000 --seed 42 --log_iters 10 --num_workers 0 --keep_checkpoint_max 2 2>&1 | Tee-Object -FilePath $log
$ec=$LASTEXITCODE
[ordered]@{exit_code=$ec; ended_at=(Get-Date).ToString('o'); seed=42; train_pairs=1400; val_pairs=177; batch_size=4; iters=18000; source='C:\local_wave1src'; test_used=$false} | ConvertTo-Json | Set-Content -Encoding UTF8 $exit

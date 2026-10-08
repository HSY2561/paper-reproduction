$ErrorActionPreference='Continue'
$src='C:\local_wave2src'
$py='C:\venv_bisenet\Scripts\python.exe'
$cfg='C:\local_wave3\configs\public_segformer_b0.yml'
$save='C:\local_wave3\training\public_segformer_b0'
$log='C:\local_wave3\logs\public_segformer_b0_train.log'
$exit='C:\local_wave3\results\public_segformer_b0_exit.json'
$env:PYTHONPATH=$src
& $py "$src\tools\train.py" --config $cfg --save_dir $save --do_eval --use_ema --save_interval 1000 --seed 42 2>&1 | Tee-Object -FilePath $log
$ec=$LASTEXITCODE
@{test_used=$false;exit_code=$ec;seed=42;ended_at=(Get-Date -Format o)} | ConvertTo-Json | Set-Content $exit

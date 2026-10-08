$ErrorActionPreference='Continue'
$src='C:\local_wave2src'
$py='C:\\venv_bisenet\Scripts\python.exe'
$cfg='C:\local_wave3\configs\C4_inputcrop.yml'
$save='C:\local_wave3\training\C4_seed44'
$log='C:\local_wave3\logs\C4_seed44_train.log'
$exit='C:\local_wave3\results\C4_seed44_exit.json'
$env:PYTHONPATH=$src
& $py "$src\tools\train.py" --config $cfg --save_dir $save --do_eval --use_ema --save_interval 1000 --seed 44 2>&1 | Tee-Object -FilePath $log
$ec=$LASTEXITCODE
@{test_used=$false;exit_code=$ec;seed=44;ended_at=(Get-Date -Format o)} | ConvertTo-Json | Set-Content $exit

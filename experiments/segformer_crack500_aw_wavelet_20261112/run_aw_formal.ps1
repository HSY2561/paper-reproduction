$ErrorActionPreference='Continue'
$root='D:\论文复现\experiments\segformer_crack500_aw_wavelet_20261112'
$src='D:\论文复现\experiments\segformer_crack500_aligned_wtconv_ablation_wave17_20261030\src\PaddleSeg-2.10.0'
$py='C:\venv_bisenet\Scripts\python.exe'
$env:PYTHONPATH=$src
$id='AW_wavelet_s42'
$cfg=Join-Path $root 'configs\AW_wavelet.yml'
$save=Join-Path $root 'training\AW_wavelet_s42'
$log=Join-Path $root 'logs\AW_wavelet_s42_train.log'
New-Item -ItemType Directory -Force -Path (Join-Path $root 'results'),(Join-Path $root 'logs'),$save | Out-Null
@{id=$id;status='running';started=(Get-Date).ToString('o');config=$cfg;source=$src;seed=42;iters=18000} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'results\AW_wavelet_s42_train_status.json')
& $py "$src\tools\train.py" --config $cfg --device gpu --save_dir $save --do_eval --use_ema --iters 18000 --batch_size 4 --seed 42 --precision fp32 --save_interval 1000 --keep_checkpoint_max 3 --log_iters 100 2>&1 | Tee-Object -FilePath $log
$ec=$LASTEXITCODE
@{id=$id;exit_code=$ec;completed=(Get-Date).ToString('o');expected_iters=18000;log=$log;save_dir=$save;config=$cfg;source=$src;seed=42} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'results\AW_wavelet_s42_train_exit.json')
@{id=$id;status=$(if($ec -eq 0){'completed'}else{'failed'});exit_code=$ec;completed=(Get-Date).ToString('o');config=$cfg} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'results\AW_wavelet_s42_train_status.json')
exit $ec

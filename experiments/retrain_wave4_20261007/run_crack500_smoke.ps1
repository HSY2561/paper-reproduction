$ErrorActionPreference='Continue'
$env:PYTHONPATH='C:\local_wave1src'
$py='C:\venv_bisenet\Scripts\python.exe'
$root='C:\retrain_wave4'
$items=@(
 @('segformer','C:\retrain_wave4\configs\public_segformer_b0.yml'),
 @('ocrnet','C:\retrain_wave4\configs\public_ocrnet_hrnetw18.yml'),
 @('ppliteseg','C:\retrain_wave4\configs\ppliteseg_stdc1_crack500_18k.yml'),
 @('pidnet','C:\retrain_wave4\configs\pidnet_s_crack500.yml'),
 @('deeplabv3p','C:\retrain_wave4\configs\deeplabv3p_r50_crack500_18k.yml')
)
foreach($it in $items){$n=$it[0];$cfg=$it[1];$save="$root\training\smoke_$n";$log="$root\logs\smoke_$n.log";$exit="$root\results\smoke_${n}_exit.json";New-Item -ItemType Directory -Force -Path (Split-Path $save),(Split-Path $log),(Split-Path $exit)|Out-Null;& $py -u C:\local_wave1src\tools\train.py --config $cfg --save_dir $save --iters 20 --batch_size 2 --do_eval --use_ema --save_interval 20 --seed 42 --log_iters 10 --num_workers 0 --keep_checkpoint_max 1 2>&1 | Tee-Object -FilePath $log;$ec=$LASTEXITCODE;[ordered]@{model=$n;exit_code=$ec;ended_at=(Get-Date).ToString('o');test_used=$false}|ConvertTo-Json|Set-Content -Encoding UTF8 $exit;if($ec -ne 0){break}}

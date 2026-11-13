$ErrorActionPreference='Continue'
$root=(Resolve-Path (Join-Path (Get-Location) 'experiments\segformer_crack500_wtconv_bgce_20261112')).Path
$src=(Resolve-Path (Join-Path (Get-Location) 'experiments\segformer_crack500_aw_wavelet_20261112\src\PaddleSeg-2.10.0')).Path
$py='C:\venv_bisenet\Scripts\python.exe'; $env:PYTHONPATH=$src
New-Item -ItemType Directory -Force -Path "$root\training","$root\checkpoints","$root\logs","$root\results","$root\provenance" | Out-Null
$cfg=Join-Path $root 'configs\WTConv_bgCE.yml'; $save=Join-Path $root 'training\WTConv_bgCE_s42'; $log=Join-Path $root 'logs\WTConv_bgCE_s42_train.log'
@{id='WTConv_bgCE_s42';status='running';started=(Get-Date).ToString('o');config=$cfg;source=$src;seed=42;iters=18000} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'results\WTConv_bgCE_s42_train_status.json')
& $py "$src\tools\train.py" --config $cfg --device gpu --save_dir $save --do_eval --use_ema --iters 18000 --batch_size 4 --seed 42 --precision fp32 --save_interval 1000 --keep_checkpoint_max 3 --log_iters 100 2>&1 | Tee-Object -FilePath $log
$ec=$LASTEXITCODE
@{id='WTConv_bgCE_s42';exit_code=$ec;completed=(Get-Date).ToString('o');expected_iters=18000;log=$log;save_dir=$save;config=$cfg;source=$src;seed=42} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'results\WTConv_bgCE_s42_train_exit.json')
@{id='WTConv_bgCE_s42';status=$(if($ec -eq 0){'completed'}else{'failed'});exit_code=$ec;completed=(Get-Date).ToString('o');config=$cfg} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'results\WTConv_bgCE_s42_train_status.json')
if($ec -ne 0){exit $ec}
$ck=Join-Path $root 'checkpoints\WTConv_bgCE_s42'; New-Item -ItemType Directory -Force $ck | Out-Null
Copy-Item (Join-Path $save 'best_model\model.pdparams') (Join-Path $ck 'model.pdparams') -Force
Copy-Item (Join-Path $save 'ema_best_model\ema_model.pdparams') (Join-Path $ck 'ema_model.pdparams') -Force
foreach($kind in 'ordinary','ema'){
 $mp=if($kind -eq 'ordinary'){Join-Path $ck 'model.pdparams'}else{Join-Path $ck 'ema_model.pdparams'}
 $out=Join-Path $root ("results\WTConv_bgCE_s42_val_${kind}_exact")
 & $py "$root\evaluate_val_boundary.py" --config $cfg --model_path $mp --output $out --device gpu
}
& $py -c "import json,pathlib,hashlib; r=pathlib.Path(r'$root'); a=json.loads((r/'results/WTConv_bgCE_s42_val_ordinary_exact.json').read_text()); e=json.loads((r/'results/WTConv_bgCE_s42_val_ema_exact.json').read_text()); s='ordinary' if a['mIoU']>=e['mIoU'] else 'ema'; p=r/'checkpoints/WTConv_bgCE_s42'/('model.pdparams' if s=='ordinary' else 'ema_model.pdparams'); d={'id':'WTConv_bgCE_s42','selected':s,'ordinary_val_mIoU':a['mIoU'],'ema_val_mIoU':e['mIoU'],'selected_checkpoint':str(p),'selected_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'test_not_used_for_selection':True}; (r/'results/WTConv_bgCE_s42_selection.json').write_text(json.dumps(d,indent=2)); print(json.dumps(d))"
$testCfg=Join-Path $root 'configs\WTConv_bgCE_test_eval.yml'; (Get-Content $cfg -Raw).Replace('pretrained: C:/Users/user/Desktop/segformer_aw_mit_b0.pdparams','pretrained: null').Replace('val_path: C:/Users/user/algolab_data/val.txt','val_path: C:/Users/user/algolab_data/test.txt') | Set-Content $testCfg -Encoding utf8
$sel=Get-Content (Join-Path $root 'results\WTConv_bgCE_s42_selection.json') -Raw | ConvertFrom-Json
& $py "$root\evaluate_test_boundary.py" --config $testCfg --model_path $sel.selected_checkpoint --output (Join-Path $root 'results\WTConv_bgCE_s42_test_exact') --device gpu
@{completed=(Get-Date).ToString('o');dataset='Crack500';train=1896;val=348;test=1124;seed=42;iters=18000} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'results\queue_done.json')

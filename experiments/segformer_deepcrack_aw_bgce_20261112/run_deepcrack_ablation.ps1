$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path (Get-Location) 'experiments\segformer_deepcrack_aw_bgce_20261112')).Path
$src = (Resolve-Path (Join-Path (Get-Location) 'experiments\segformer_crack500_aw_wavelet_20261112\src\PaddleSeg-2.10.0')).Path
$py = 'C:\venv_bisenet\Scripts\python.exe'
$eval = Join-Path $root 'evaluate_split_boundary.py'
$dataset = 'C:\Users\user\Desktop\segformer_aw_cross_dataset_wave17_20261106\normalized\DeepCrack'
$env:PYTHONPATH = $src
$groups = @(
    @{ id = 'C0_B0'; cfg = 'C0_B0.yml' },
    @{ id = 'C1_Aligned'; cfg = 'C1_Aligned.yml' },
    @{ id = 'C2_WTConvWavelet'; cfg = 'C2_WTConvWavelet.yml' },
    @{ id = 'C3_Aligned_WTConvWavelet_bgce'; cfg = 'C3_Aligned_WTConvWavelet_bgce.yml' }
)
foreach ($g in $groups) {
    $id = $g.id
    $cfg = Join-Path $root ('configs\' + $g.cfg)
    $save = Join-Path $root ('training\' + $id + '_s42')
    $log = Join-Path $root ('logs\' + $id + '_s42_train.log')
    $status = Join-Path $root ('results\' + $id + '_s42_train_status.json')
    $exitFile = Join-Path $root ('results\' + $id + '_s42_train_exit.json')
    New-Item -ItemType Directory -Force -Path $save | Out-Null
    @{ id=$id; status='running'; started=(Get-Date).ToString('o'); config=$cfg; source=$src; seed=42; iters=18000; loss='CrossEntropyLoss weight [1.10,1.0]' } | ConvertTo-Json | Set-Content -Encoding utf8 $status
    & $py "$src\tools\train.py" --config $cfg --device gpu --save_dir $save --do_eval --use_ema --iters 18000 --batch_size 4 --seed 42 --precision fp32 --save_interval 1000 --keep_checkpoint_max 3 --log_iters 100 2>&1 | Tee-Object -FilePath $log
    $ec = $LASTEXITCODE
    @{ id=$id; exit_code=$ec; completed=(Get-Date).ToString('o'); expected_iters=18000; log=$log; save_dir=$save; config=$cfg; source=$src; seed=42 } | ConvertTo-Json | Set-Content -Encoding utf8 $exitFile
    @{ id=$id; status=$(if($ec -eq 0){'completed'}else{'failed'}); exit_code=$ec; completed=(Get-Date).ToString('o'); config=$cfg } | ConvertTo-Json | Set-Content -Encoding utf8 $status
    if ($ec -ne 0) { continue }
    $ckptDir = Join-Path $root ('checkpoints\' + $id + '_s42')
    New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
    Copy-Item (Join-Path $save 'best_model\model.pdparams') (Join-Path $ckptDir 'model.pdparams') -Force
    Copy-Item (Join-Path $save 'ema_best_model\ema_model.pdparams') (Join-Path $ckptDir 'ema_model.pdparams') -Force
    foreach ($kind in 'ordinary','ema') {
        $modelPath = if($kind -eq 'ordinary'){ Join-Path $ckptDir 'model.pdparams' } else { Join-Path $ckptDir 'ema_model.pdparams' }
        $out = Join-Path $root ('results\' + $id + '_s42_val_' + $kind + '_exact')
        & $py $eval --config $cfg --model_path $modelPath --output $out --device gpu --expected_val_path "$dataset\val.txt" --expected_count 45 --dataset_name DeepCrack --split val --provenance_tag $id
    }
    $pick = Join-Path $root ('results\' + $id + '_s42_selection.json')
    & $py -c "import json, pathlib, hashlib; r=pathlib.Path(r'$root'); oid='$id'; a=json.loads((r/'results'/f'{oid}_s42_val_ordinary_exact.json').read_text()); e=json.loads((r/'results'/f'{oid}_s42_val_ema_exact.json').read_text()); sel='ordinary' if a['mIoU']>=e['mIoU'] else 'ema'; ck=r/'checkpoints'/f'{oid}_s42'/(('model.pdparams' if sel=='ordinary' else 'ema_model.pdparams')); h=hashlib.sha256(ck.read_bytes()).hexdigest(); out={'id':oid,'selected':sel,'selection_metric':'val_mIoU','ordinary_val_mIoU':a['mIoU'],'ema_val_mIoU':e['mIoU'],'selected_checkpoint':str(ck),'selected_sha256':h,'test_not_used_for_selection':True}; (r/'results'/f'{oid}_s42_selection.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out))"
    $cfgTest = Join-Path $root ('configs\' + $id + '_test.yml')
    (Get-Content $cfg -Raw).Replace('/val.txt','/test.txt') | Set-Content $cfgTest -Encoding utf8
    $selection = Get-Content $pick -Raw | ConvertFrom-Json
    $testOut = Join-Path $root ('results\' + $id + '_s42_test_exact')
    & $py $eval --config $cfgTest --model_path $selection.selected_checkpoint --output $testOut --device gpu --expected_val_path "$dataset\test.txt" --expected_count 237 --dataset_name DeepCrack --split test --provenance_tag $id
}
@{ completed=(Get-Date).ToString('o'); groups=$groups.id; source=$src; dataset=$dataset; loss='CrossEntropyLoss weight [1.10,1.0]'; iters=18000; seed=42 } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $root 'results\queue_done.json')

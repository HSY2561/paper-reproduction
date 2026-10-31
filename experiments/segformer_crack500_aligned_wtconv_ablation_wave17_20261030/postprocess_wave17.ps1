$ErrorActionPreference = 'Stop'
$root = 'D:\论文复现\experiments\segformer_crack500_aligned_wtconv_ablation_wave17_20261030'
$src = Join-Path $root 'src\PaddleSeg-2.10.0'
$py = 'C:\venv_bisenet\Scripts\python.exe'
$env:PYTHONPATH = $src
$jobs = @('C0_B0','C1_Aligned','C2_WTConv','C3_Aligned_WTConv')

foreach ($id in $jobs) {
    $trainRoot = Join-Path $root "training\$id"
    $ordinary = Join-Path $trainRoot 'best_model\model.pdparams'
    $ema = Join-Path $trainRoot 'ema_best_model\ema_model.pdparams'
    if (-not (Test-Path $ordinary) -or -not (Test-Path $ema)) {
        throw "Missing best checkpoint for $id"
    }
    $valConfig = Join-Path $root "configs\$id.yml"
    $valOrd = Join-Path $root "results\${id}_val_ordinary"
    $valEma = Join-Path $root "results\${id}_val_ema"
    & $py (Join-Path $root 'evaluate_val_boundary.py') --config $valConfig --model_path $ordinary --output $valOrd --device gpu 2>&1 |
        Tee-Object (Join-Path $root "logs\${id}_val_ordinary_boundary.log")
    if ($LASTEXITCODE -ne 0) { throw "Validation boundary failed for $id ordinary" }
    & $py (Join-Path $root 'evaluate_val_boundary.py') --config $valConfig --model_path $ema --output $valEma --device gpu 2>&1 |
        Tee-Object (Join-Path $root "logs\${id}_val_ema_boundary.log")
    if ($LASTEXITCODE -ne 0) { throw "Validation boundary failed for $id EMA" }
    $ord = Get-Content ($valOrd + '.json') -Raw | ConvertFrom-Json
    $em = Get-Content ($valEma + '.json') -Raw | ConvertFrom-Json
    if ([double]$ord.mIoU -ge [double]$em.mIoU) {
        $selected = $ordinary
        $selectedKind = 'ordinary'
        $selectedMetrics = $ord
    } else {
        $selected = $ema
        $selectedKind = 'ema'
        $selectedMetrics = $em
    }
    $frozenDir = Join-Path $root "checkpoints\$id"
    New-Item -ItemType Directory -Force -Path $frozenDir | Out-Null
    $frozen = Join-Path $frozenDir 'model.pdparams'
    Copy-Item -Force $selected $frozen
    $hash = (Get-FileHash -Algorithm SHA256 $frozen).Hash.ToLowerInvariant()
    [ordered]@{
        id = $id
        selected = $selectedKind
        selected_source = $selected
        frozen_checkpoint = $frozen
        frozen_sha256 = $hash
        ordinary = $ord
        ema = $em
        selection_rule = 'highest exact validation mIoU from ordinary/EMA; test not used'
    } | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 (Join-Path $root "results\${id}_selection.json")
    & $py (Join-Path $src 'tools\val.py') --config $valConfig --model_path $frozen --device gpu 2>&1 |
        Tee-Object (Join-Path $root "logs\${id}_val_official.log")
    if ($LASTEXITCODE -ne 0) { throw "Official val failed for $id" }
}

foreach ($id in $jobs) {
    $testConfig = Join-Path $root "configs\${id}_test.yml"
    $frozen = Join-Path $root "checkpoints\$id\model.pdparams"
    $out = Join-Path $root "results\${id}_test"
    & $py (Join-Path $root 'evaluate_test_boundary.py') --config $testConfig --model_path $frozen --output $out --device gpu 2>&1 |
        Tee-Object (Join-Path $root "logs\${id}_test_boundary.log")
    if ($LASTEXITCODE -ne 0) { throw "Test boundary failed for $id" }
    & $py (Join-Path $src 'tools\val.py') --config $testConfig --model_path $frozen --device gpu 2>&1 |
        Tee-Object (Join-Path $root "logs\${id}_test_official.log")
    if ($LASTEXITCODE -ne 0) { throw "Official test failed for $id" }
}
@{ completed = (Get-Date).ToString('o'); jobs = $jobs; test_used_after_freeze = $true } |
    ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 (Join-Path $root 'results\wave17_postprocess_done.json')

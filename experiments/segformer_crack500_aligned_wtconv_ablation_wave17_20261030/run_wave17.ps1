$ErrorActionPreference = 'Continue'
$root = 'D:\论文复现\experiments\segformer_crack500_aligned_wtconv_ablation_wave17_20261030'
$src = Join-Path $root 'src\PaddleSeg-2.10.0'
$py = 'C:\venv_bisenet\Scripts\python.exe'
$env:PYTHONPATH = $src
$jobs = @(
    @{ Id = 'C0_B0'; Config = 'C0_B0.yml' },
    @{ Id = 'C1_Aligned'; Config = 'C1_Aligned.yml' },
    @{ Id = 'C2_WTConv'; Config = 'C2_WTConv.yml' },
    @{ Id = 'C3_Aligned_WTConv'; Config = 'C3_Aligned_WTConv.yml' }
)

foreach ($job in $jobs) {
    $id = $job.Id
    $statusPath = Join-Path $root "results\${id}_train_status.json"
    $exitPath = Join-Path $root "results\${id}_train_exit.json"
    $logPath = Join-Path $root "logs\${id}_train.log"
    $saveDir = Join-Path $root "training\${id}"
    $cfg = Join-Path $root "configs\$($job.Config)"
    @{ id = $id; status = 'running'; started = (Get-Date).ToString('o'); config = $cfg } |
        ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $statusPath
    & $py "$src\tools\train.py" --config $cfg --device gpu --save_dir $saveDir --do_eval --use_ema --iters 18000 --batch_size 4 --seed 42 --precision fp32 --save_interval 1000 --keep_checkpoint_max 3 --log_iters 100 2>&1 |
        Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    @{ id = $id; exit_code = $exitCode; completed = (Get-Date).ToString('o'); expected_iters = 18000; log = $logPath; save_dir = $saveDir } |
        ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $exitPath
    @{ id = $id; status = ($(if ($exitCode -eq 0) { 'completed' } else { 'failed' })); exit_code = $exitCode; completed = (Get-Date).ToString('o'); config = $cfg } |
        ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $statusPath
}
@{ completed = (Get-Date).ToString('o'); jobs = $jobs.Id } | ConvertTo-Json -Depth 5 |
    Set-Content -Encoding utf8 (Join-Path $root 'results\wave17_train_done.json')

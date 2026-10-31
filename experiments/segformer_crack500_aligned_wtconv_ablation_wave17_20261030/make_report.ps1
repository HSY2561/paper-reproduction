$ErrorActionPreference = 'Stop'
$root = 'D:\论文复现\experiments\segformer_crack500_aligned_wtconv_ablation_wave17_20261030'
$ids = @('C0_B0','C1_Aligned','C2_WTConv','C3_Aligned_WTConv')
$names = @{
    C0_B0 = 'SegFormer-B0 (C0)'
    C1_Aligned = 'SegFormer-B0 + Aligned (C1)'
    C2_WTConv = 'SegFormer-B0 + WTConv-only (C2)'
    C3_Aligned_WTConv = 'SegFormer-B0 + Aligned + WTConv (C3)'
}
$params = @{ C0_B0 = 3715170; C1_Aligned = 3785314; C2_WTConv = 3653346; C3_Aligned_WTConv = 3723490 }
$flops = @{ C0_B0 = 4135182528; C1_Aligned = 4514480320; C2_WTConv = 3391822528; C3_Aligned_WTConv = 3771120320 }
$rows = @()
foreach ($id in $ids) {
    $sel = Get-Content "$root\results\${id}_selection.json" -Raw | ConvertFrom-Json
    $val = if ($sel.selected -eq 'ordinary') { $sel.ordinary } else { $sel.ema }
    $test = Get-Content "$root\results\${id}_test.json" -Raw | ConvertFrom-Json
    $rows += [ordered]@{
        model = $names[$id]; id = $id; selected_checkpoint = $sel.selected
        val_mIoU = [double]$val.mIoU; val_crackIoU = [double]$val.crackIoU
        val_precision = [double]$val.precision; val_recall = [double]$val.recall; val_f1 = [double]$val.f1; val_boundary_f1_2px = [double]$val.boundary_f1_2px
        test_mIoU = [double]$test.mIoU; test_crackIoU = [double]$test.crackIoU
        test_precision = [double]$test.precision; test_recall = [double]$test.recall; test_f1 = [double]$test.f1; test_boundary_f1_2px = [double]$test.boundary_f1_2px
        params = $params[$id]; flops = $flops[$id]; checkpoint_sha256 = $sel.frozen_sha256
    }
}
$rows | Export-Csv -NoTypeInformation -Encoding utf8 "$root\results\crack500_aligned_wtconv_ablation_table.csv"
$base = $rows[0]
$pct = { param($x) '{0:F4}' -f ($x * 100) }
$pp = { param($x) '{0:+0.0000;-0.0000;0.0000}' -f ($x * 100) }
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# Crack500 Aligned 与 WTConv 四组消融报告（Wave17）')
$lines.Add('')
$lines.Add('实验日期：2026-10-31。所有结果来自 RTX 5060、PaddlePaddle 3.4.0、FP32、seed=42、batch=4、AdamW、初始学习率 6e-5、PolynomialDecay end_lr=0、18,000 iter、相同 400x400 训练增强和 Crack500 原始 train/val/test 清单。训练和选模只使用 train/val；所有四组冻结后才访问 1124 张 test 图。')
$lines.Add('')
$lines.Add('C2 是 WTConv-only：`alignment: none`；C3 才是 `alignment: full` 与 WTConv 联合。WTConv 采用当前工程中已有的低频 AvgPool2D + 高分支 depthwise 5x5 + pointwise/residual 实现，报告名称使用 WTConv-only candidate，避免把它表述为论文原始小波分解实现。')
$lines.Add('')
$lines.Add('## 验证集选模')
$lines.Add('')
$lines.Add('| 模型 | 选中 | mIoU | 裂缝 IoU | Precision | Recall | F1 | 边界 F1@2px | Params | FLOPs |')
$lines.Add('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
foreach ($r in $rows) { $lines.Add("| $($r.model) | $($r.selected_checkpoint) | $(& $pct $r.val_mIoU) | $(& $pct $r.val_crackIoU) | $(& $pct $r.val_precision) | $(& $pct $r.val_recall) | $(& $pct $r.val_f1) | $(& $pct $r.val_boundary_f1_2px) | $([math]::Round($r.params/1e6,3))M | $([math]::Round($r.flops/1e9,3))G |") }
$lines.Add('')
$lines.Add('## 冻结 checkpoint 后的 Crack500 test')
$lines.Add('')
$lines.Add('| 模型 | Precision | Recall | F1-score | IoU | mIoU | 边界 F1@2px | Params |')
$lines.Add('|---|---:|---:|---:|---:|---:|---:|---:|')
foreach ($r in $rows) { $lines.Add("| $($r.model) | $(& $pct $r.test_precision) | $(& $pct $r.test_recall) | $(& $pct $r.test_f1) | $(& $pct $r.test_crackIoU) | $(& $pct $r.test_mIoU) | $(& $pct $r.test_boundary_f1_2px) | $([math]::Round($r.params/1e6,3))M |") }
$lines.Add('')
$lines.Add('## 相对 C0 的 test 变化（百分点）')
$lines.Add('')
$lines.Add('| 模型 | ΔPrecision | ΔRecall | ΔF1 | Δ裂缝 IoU | ΔmIoU | Δ边界 F1@2px | ΔParams |')
$lines.Add('|---|---:|---:|---:|---:|---:|---:|---:|')
foreach ($r in $rows) { $lines.Add("| $($r.model) | $(& $pp ($r.test_precision-$base.test_precision)) | $(& $pp ($r.test_recall-$base.test_recall)) | $(& $pp ($r.test_f1-$base.test_f1)) | $(& $pp ($r.test_crackIoU-$base.test_crackIoU)) | $(& $pp ($r.test_mIoU-$base.test_mIoU)) | $(& $pp ($r.test_boundary_f1_2px-$base.test_boundary_f1_2px)) | $([math]::Round(($r.params-$base.params)/1e6,3))M |") }
$lines.Add('')
$lines.Add('## 结论')
$lines.Add('')
$lines.Add('Aligned 单模块在本次统一 test 协议下显著提高 Recall、F1 和裂缝 IoU，Precision 下降约 1.9133 个百分点，边界 F1 基本不变。WTConv-only 参数量最低、FLOPs 最低，并在 test 上小幅提升 mIoU、裂缝 IoU 和 F1，说明效率收益明确，但精度增益小于 Aligned。联合模型的 mIoU、裂缝 IoU、F1 和 Recall 均为四组最高，但 Precision 下降最大，边界 F1 低于 C0，显示出假阳性增加和边界质量退化的代价。')
$lines.Add('')
$lines.Add('这些结果支持把 Aligned 作为主要精度候选，把 WTConv-only 作为效率候选；联合模型需要在后续自建数据和机器人闭环验证前保留其 Precision、边界质量与实时延迟风险。当前结果是本实验协议内的比较结论，不构成跨论文意义上的 SOTA 声明。')
$lines.Add('')
$lines.Add('冻结 checkpoint 的 SHA256、普通/EMA 验证明细、逐图边界数据和完整 JSON 均保存在 `results`、`checkpoints`、`provenance` 目录。')
$report = $lines -join "`r`n"
Set-Content -Encoding utf8 "$root\Crack500_Aligned_WTConv_消融报告_20261031.md" $report
Set-Content -Encoding utf8 'D:\论文复现\Crack500_Aligned_WTConv_消融报告_20261031.md' $report

$manifest = [ordered]@{
    generated = (Get-Date).ToString('o'); dataset_root = 'C:\Users\user\algolab_data'; train_count = 1896; val_count = 348; test_count = 1124
    protocol = [ordered]@{ device='RTX5060'; paddle='3.4.0'; python='C:\venv_bisenet\Scripts\python.exe'; seed=42; batch_size=4; iters=18000; precision='fp32'; optimizer='AdamW'; learning_rate=0.00006; end_lr=0; train_val_only_for_selection=$true }
    source_segformer_ten_sha256 = (Get-FileHash -Algorithm SHA256 "$root\src\PaddleSeg-2.10.0\paddleseg\models\segformer_ten.py").Hash.ToLowerInvariant()
    train_list_sha256 = (Get-FileHash -Algorithm SHA256 'C:\Users\user\algolab_data\train.txt').Hash.ToLowerInvariant()
    val_list_sha256 = (Get-FileHash -Algorithm SHA256 'C:\Users\user\algolab_data\val.txt').Hash.ToLowerInvariant()
    test_list_sha256 = (Get-FileHash -Algorithm SHA256 'C:\Users\user\algolab_data\test.txt').Hash.ToLowerInvariant()
    rows = $rows
}
$manifest | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 "$root\provenance\wave17_final_manifest.json"

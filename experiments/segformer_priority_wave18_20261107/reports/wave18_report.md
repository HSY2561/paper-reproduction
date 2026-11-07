# wave18 六数据集 × 四模型汇总

生成时间（UTC）：`2026-11-07T09:00:21+00:00`

本报告由只读汇总脚本生成；脚本不会启动训练，也不会改写 `results/` 中已有 evaluator JSON。
输入目录：`D:\论文复现\experiments\segformer_priority_wave18_20261107\results`；CSV：`reports/wave18_summary.csv`。

## 选择规则

对每个数据集/模型组合，直接比较普通（ordinary）与 EMA 验证 evaluator JSON 中的原始 JSON 数值 `mIoU`；较高者作为 `selected_val_mIoU`。完全相等时保留 ordinary，并标记 `tie_ordinary`。测试结果优先使用同一 variant；若测试 JSON 没有 variant 标记，报告中标记 `test_variant_fallback`。

## 结果

| 数据集 | 模型 | 状态 | 选中验证权重 | ordinary val mIoU | EMA val mIoU | 选中 val mIoU | 选中 val crackIoU | 选中 val F1 | test mIoU | test F1 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepCrack | C0_B0 | missing | — | — | — | — | — | — | — | — | — |
| DeepCrack | C1_Aligned | missing | — | — | — | — | — | — | — | — | — |
| DeepCrack | C2_WTConv | missing | — | — | — | — | — | — | — | — | — |
| DeepCrack | C3_Aligned_WTConv | missing | — | — | — | — | — | — | — | — | — |
| CFD | C0_B0 | missing | — | — | — | — | — | — | — | — | — |
| CFD | C1_Aligned | missing | — | — | — | — | — | — | — | — | — |
| CFD | C2_WTConv | missing | — | — | — | — | — | — | — | — | — |
| CFD | C3_Aligned_WTConv | missing | — | — | — | — | — | — | — | — | — |
| CrackTree260 | C0_B0 | missing | — | — | — | — | — | — | — | — | — |
| CrackTree260 | C1_Aligned | missing | — | — | — | — | — | — | — | — | — |
| CrackTree260 | C2_WTConv | missing | — | — | — | — | — | — | — | — | — |
| CrackTree260 | C3_Aligned_WTConv | missing | — | — | — | — | — | — | — | — | — |
| CrackLS315 | C0_B0 | missing | — | — | — | — | — | — | — | — | — |
| CrackLS315 | C1_Aligned | missing | — | — | — | — | — | — | — | — | — |
| CrackLS315 | C2_WTConv | missing | — | — | — | — | — | — | — | — | — |
| CrackLS315 | C3_Aligned_WTConv | missing | — | — | — | — | — | — | — | — | — |
| CRKWH100 | C0_B0 | missing | — | — | — | — | — | — | — | — | — |
| CRKWH100 | C1_Aligned | missing | — | — | — | — | — | — | — | — | — |
| CRKWH100 | C2_WTConv | missing | — | — | — | — | — | — | — | — | — |
| CRKWH100 | C3_Aligned_WTConv | missing | — | — | — | — | — | — | — | — | — |
| GAPS384 | C0_B0 | missing | — | — | — | — | — | — | — | — | — |
| GAPS384 | C1_Aligned | missing | — | — | — | — | — | — | — | — | — |
| GAPS384 | C2_WTConv | missing | — | — | — | — | — | — | — | — | — |
| GAPS384 | C3_Aligned_WTConv | missing | — | — | — | — | — | — | — | — | — |

## 数据与样本量披露

- `CRKWH100` 为 `CHCrack5K` 派生数据集/划分；它不应被表述为与 CHCrack5K 完全独立的新数据来源。
- `GAPS384` 的 validation split 仅包含 5 张图像；对应 val mIoU 方差较大，不能按大样本验证集解读。

## 输入扫描

扫描 JSON：8；识别 evaluator：0；忽略非 evaluator：8；重复记录忽略/替换：0/0。
固定输出行数：24（应为 24）；状态统计：{'missing': 24}。

# 论文复现交付索引

## 当前优先入口

- `experiments/segformer_crack500_aligned_wtconv_ablation_wave17_20261030/` 保存 Crack500 四组统一验证、test、边界指标、配置、源码、checkpoint SHA 和训练回执。
- `experiments/segformer_vocdevkit3000_aligned_wtconv_ablation_20261031/` 保存 VOCdevkit3000 四组对照、官方验证、独立边界评价和失败 smoke 证据。
- `experiments/segformer_vocdevkit3000_jointfix_wave15_20261106/` 保存联合模型参数修正后的独立复核材料。
- `experiments/segformer_vocdevkit3000_aw_loss_wave16_20261106/` 保存损失函数复核的计划、源码和结果表。
- `experiments/segformer_priority_wave18_20261107/` 保存跨数据集扫描配置、汇总脚本、真实 smoke 回执和缺失状态。

## 论文写作材料

- `Crack500_Aligned_WTConv_消融报告_20261031.md`
- `VOCdevkit3000_四组消融结果表_真实审计_20261106.md`
- `SegFormer_Aligned_WTConv_方法与实验章节_20261031.md`
- `SegFormer_Aligned_WTConv_表格图示与论文段落交付_20261031.md`
- `SegFormer_Aligned_WTConv_论文4.1_4.2_中英文精简稿_20261102.md`

## 审计和环境

- `cleanup_manifest_20261024.json` 与 `cleanup_result_20261024.json` 记录既有清理动作。
- `runtime/ENVIRONMENT.md` 与 `runtime/requirements-wave17.txt` 记录复现环境。
- 各实验目录的 `provenance/`、`results/` 和 `refine-logs/` 是原始证据，不与新结果混用。

数据集、缓存和未列入发布清单的权重仍在本地。历史实验目录保留用于追溯，根目录的旧报告副本已经取消跟踪。

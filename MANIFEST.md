# 论文复现交付索引

## 最新结果

- `experiments/segformer_crack500_aw_wavelet_20261112/` 保存原始 AW 和 AW + 背景加权 CE 的 Crack500 seed42 训练、验证、test、阈值扫描、逐图 CSV、配置、provenance 和最小复现源码。
- `experiments/segformer_crack500_aligned_wtconv_ablation_wave17_20261030/` 保存 Crack500 四组 AvgPool WTConv 对照、统一 test、边界指标、配置、源码和 checkpoint SHA。
- `experiments/segformer_vocdevkit3000_aligned_wtconv_ablation_20261031/` 保存 VOCdevkit3000 四组对照和独立边界评价。
- `experiments/segformer_vocdevkit3000_jointfix_wave15_20261106/` 保存联合模型参数修正后的复核材料。
- `experiments/segformer_vocdevkit3000_aw_loss_wave16_20261106/` 保存损失函数复核计划、源码和结果表。
- `experiments/segformer_priority_wave18_20261107/` 保存跨数据集扫描和真实缺失状态。

## 论文写作与审计

- `experiments/segformer_crack500_aw_wavelet_20261112/AW精度诊断与单变量修正报告_20261112.md`
- `experiments/segformer_crack500_aw_wavelet_20261112/AW重新训练与六模型对照报告_20261112.md`
- `Crack500_Aligned_WTConv_消融报告_20261031.md`
- `VOCdevkit3000_四组消融结果表_真实审计_20261106.md`
- `SegFormer_Aligned_WTConv_方法与实验章节_20261031.md`
- `SegFormer_Aligned_WTConv_表格图示与论文段落交付_20261031.md`
- `SegFormer_Aligned_WTConv_论文4.1_4.2_中英文精简稿_20261102.md`

## 环境与追溯

- `runtime/ENVIRONMENT.md` 和 `runtime/requirements-wave17.txt` 记录复现环境。
- 各实验目录的 `provenance/`、`results/` 和 `refine-logs/` 是原始证据。
- `cleanup_manifest_20261024.json` 与 `cleanup_result_20261024.json` 记录既有清理动作。

数据集、缓存、进程锁、训练中间目录和未列入发布清单的权重保留在本地。checkpoint 选择记录、阈值记录和 SHA256 继续公开，便于复核。

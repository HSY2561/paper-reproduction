# SegFormer＋Aligned 损失函数消融进度

更新时间：2026-10-21T23:56:07+08:00

仅Crack500验证集；20主方案，不按验证成绩增加替补。所有模型结构和推理参数相同。权重变体不计独立算法。

| 编号 | 配方 | 状态 | P% | R% | F1% | IoU% | mIoU% | BF1% | Params M |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | A0 + CE | 冻结复用 | 82.6766 | 79.0974 | 80.8474 | 67.8520 | 82.8184 | 44.9285 | 3.470434 |
| L01_CE_Dice | CE + 0.1 Dice | gradient_preflight_failed | — | — | — | — | — | — | — |
| L02_CE_Dice | CE + 0.25 Dice | gradient_preflight_failed | — | — | — | — | — | — | — |
| L03_CE_Dice | CE + 0.5 Dice | gradient_preflight_failed | — | — | — | — | — | — | — |
| L04_CE_Dice | CE + 1.0 Dice | gradient_preflight_failed | — | — | — | — | — | — | — |
| L05_CE_Lovasz | CE + 0.1 Lovasz | complete | 80.6454 | 81.5407 | 81.0906 | 68.1953 | 82.9721 | 44.1729 | 3.470434 |
| L06_CE_Lovasz | CE + 0.25 Lovasz | complete | 80.4899 | 82.3360 | 81.4025 | 68.6376 | 83.2048 | 44.7172 | 3.470434 |
| L07_CE_Lovasz | CE + 0.5 Lovasz | complete | 79.9114 | 82.6848 | 81.2744 | 68.4557 | 83.0994 | 45.4401 | 3.470434 |
| L08_WeightedCE | CE weights [1.5, 1.0] | complete | 82.7779 | 77.7043 | 80.1609 | 66.8904 | 82.3095 | 44.1797 | 3.470434 |
| L09_WeightedCE | CE weights [2.0, 1.0] | complete | 85.3079 | 73.7972 | 79.1362 | 65.4755 | 81.5913 | 43.5890 | 3.470434 |
| L10_WeightedCE | CE weights [1.0, 2.0] | complete | 77.4460 | 84.4415 | 80.7926 | 67.7748 | 82.6971 | 41.7729 | 3.470434 |
| L11_TopKCE | CE top 0.5 | complete | 82.1388 | 79.8635 | 80.9851 | 68.0463 | 82.9143 | 45.1958 | 3.470434 |
| L12_TopKCE | CE top 0.25 | complete | 82.4409 | 79.5361 | 80.9625 | 68.0142 | 82.9014 | 45.0187 | 3.470434 |
| L13_OHEM | OHEM CE | complete | 81.9571 | 80.4181 | 81.1803 | 68.3223 | 83.0583 | 45.8753 | 3.470434 |
| L14_SoftmaxFocal | Softmax Focal gamma 1.0 | complete | 82.6819 | 78.9628 | 80.7796 | 67.7565 | 82.7678 | 45.0176 | 3.470434 |
| L15_SoftmaxFocal | Softmax Focal gamma 2.0 | complete | 81.6363 | 79.5059 | 80.5570 | 67.4439 | 82.5875 | 43.7721 | 3.470434 |
| L16_CE_Focal | CE + 0.5 Softmax Focal | complete | 82.1238 | 79.6862 | 80.8866 | 67.9073 | 82.8403 | 44.9137 | 3.470434 |
| L17_OHEM_Dice | OHEM + 0.25 Dice | complete | 81.9058 | 79.9594 | 80.9209 | 67.9556 | 82.8629 | 45.2092 | 3.470434 |
| L18_WeightedCE_Dice | Weighted CE [1.5,1] + 0.25 Dice | gradient_preflight_failed | — | — | — | — | — | — | — |
| L19_CE_DualTask | CE + 0.01 DualTask | complete | 80.1984 | 81.2526 | 80.7220 | 67.6755 | 82.6894 | 43.6268 | 3.470434 |
| L20_CE_RMI | CE + 0.01 RMI | complete | 82.0080 | 80.0116 | 80.9975 | 68.0637 | 82.9217 | 45.2611 | 3.470434 |

普通/EMA按最高精确val mIoU选模，冻结后独立验证。筛选不读test或自建数据；失败不填性能。
不同损失的绝对标量不可横向当作精度；参数和推理算子不变，不能将损失优化称为减参或加速。

## 2026-10-23 最终审计与公开模型对照

- 20项损失方案已合并：L01/L02/L03/L04/L18的原始Dice反向失败仍保留，兼容副本完成训练，合并表以`original_status`和`complete_compat_clone`区分来源。
- L01/L02 seed43/44复核已完成，三种子均值严格门槛均未通过，主要原因是Precision均值下降超过0.005；不宣称最终改进模型或SOTA。
- 已冻结公开checkpoint并统一访问test：SegFormer-B0、OCRNet-HRNet-W18、PP-LiteSeg-STDC1。PSPNet、DeepLabV3P、Fast-SCNN保留真实GPU smoke失败证据。
- 统一公开对照结果与逐图CSV：`D:/论文复现/experiments/segformer_public_wave9_20261023/results/public_comparison.json`、`public_comparison.csv`。
- 总报告：`D:/论文复现/SegFormer损失函数消融总报告_20261023.md`；交付压缩包：`D:/论文复现/SegFormer_wave8_wave9_report_delivery_20261023.zip`。

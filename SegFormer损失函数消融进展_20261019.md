# SegFormer＋Aligned 损失函数消融进度

更新时间：2026-10-20T02:35:13+08:00

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
| L07_CE_Lovasz | CE + 0.5 Lovasz | pending | — | — | — | — | — | — | — |
| L08_WeightedCE | CE weights [1.5, 1.0] | pending | — | — | — | — | — | — | — |
| L09_WeightedCE | CE weights [2.0, 1.0] | pending | — | — | — | — | — | — | — |
| L10_WeightedCE | CE weights [1.0, 2.0] | pending | — | — | — | — | — | — | — |
| L11_TopKCE | CE top 0.5 | pending | — | — | — | — | — | — | — |
| L12_TopKCE | CE top 0.25 | pending | — | — | — | — | — | — | — |
| L13_OHEM | OHEM CE | pending | — | — | — | — | — | — | — |
| L14_SoftmaxFocal | Softmax Focal gamma 1.0 | pending | — | — | — | — | — | — | — |
| L15_SoftmaxFocal | Softmax Focal gamma 2.0 | pending | — | — | — | — | — | — | — |
| L16_CE_Focal | CE + 0.5 Softmax Focal | pending | — | — | — | — | — | — | — |
| L17_OHEM_Dice | OHEM + 0.25 Dice | pending | — | — | — | — | — | — | — |
| L18_WeightedCE_Dice | Weighted CE [1.5,1] + 0.25 Dice | pending | — | — | — | — | — | — | — |
| L19_CE_DualTask | CE + 0.01 DualTask | pending | — | — | — | — | — | — | — |
| L20_CE_RMI | CE + 0.01 RMI | pending | — | — | — | — | — | — | — |

普通/EMA按最高精确val mIoU选模，冻结后独立验证。筛选不读test或自建数据；失败不填性能。
不同损失的绝对标量不可横向当作精度；参数和推理算子不变，不能将损失优化称为减参或加速。

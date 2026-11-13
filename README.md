# SegFormer Crack Segmentation Mainline

当前主线固定为 **SegFormer-B0 + Aligned + 标准 Haar WTConv + background-weighted CE（AW+bgCE）**。

## 主线结果

- Crack500 AW+bgCE test：Precision 73.159%、Recall 74.734%、F1 73.938%、裂缝 IoU 58.652%、mIoU 77.789%。
- VOCdevkit3000 当前标准 Haar WTConv+bgCE：587 张 val，Precision 87.598%、Recall 91.077%、F1 89.304%、IoU 80.674%、mIoU 89.743%。`test.txt` 为空。
- DeepCrack 四组当前 Haar+bgCE 消融：WTConv-only 的 IoU/mIoU/F1/边界和 hard clDice 最好；联合模型 Precision 最高，但没有超过 WTConv-only 的区域指标。

完整数字和口径见 [AW公开对照与自建集全部消融汇总_20261112.md](AW公开对照与自建集全部消融汇总_20261112.md)。

## 保留证据

| 目录/文件 | 内容 |
|---|---|
| `experiments/segformer_crack500_aw_wavelet_20261112` | AW+bgCE 主线、阈值诊断、六模型公开对照、Haar WTConv 源码和精确 JSON |
| `experiments/segformer_crack500_wtconv_bgce_20261112` | Crack500 当前标准 Haar WTConv-only 重训和 test |
| `experiments/segformer_voc_wtconv_bgce_20261112` | VOCdevkit3000 当前标准 Haar WTConv-only，587 张 val |
| `experiments/segformer_deepcrack_aw_bgce_20261112` | DeepCrack train/val/test=255/45/237 的四组消融 |
| `experiments/segformer_vocdevkit3000_aligned_wtconv_ablation_20261031` | VOCdevkit3000 旧四格历史消融，旧 WTConv candidate+普通 CE |
| `AW公开对照与自建集全部消融汇总_20261112.md` | 最新公开对照与自建集汇总 |
| `WTConv标准Haar实现与兼容说明_20261111.md` | WTConv 标准 Haar 实现说明 |

权重、原始数据、训练日志和本机环境不进入 GitHub；checkpoint SHA、配置 SHA、exit 回执和评估 JSON 保留在各实验目录。`test.txt` 为空的数据集只报告 val，历史 WTConv candidate 与当前 Haar 版本分开记录。

## 结果口径

公开模型行是历史冻结记录，训练后端、初始化、损失和 EMA 可能不同，只作对照，不能据此宣称领域 SOTA。所有主线结果均保留普通/EMA 选模、数据清单、配置和 checkpoint SHA。

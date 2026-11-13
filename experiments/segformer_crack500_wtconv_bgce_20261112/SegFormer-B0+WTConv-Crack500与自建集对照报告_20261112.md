# SegFormer-B0 + WTConv：Crack500 与 VOCdevkit3000 对照（2026-11-12）

## Crack500 当前标准 Haar WTConv-only

配置：SegFormer-B0、`alignment=none`、标准 Haar `wtconv_wavelet`（levels=1、kernel=3）、背景加权 CE `[1.10,1.0]`、seed42、18,000 iter、EMA。普通 checkpoint 按 val mIoU 选中，val mIoU=82.6652%。

| Model | Precision | Recall | F1 | IoU | mIoU | BF1@2 | hard clDice | Params(M) | FLOPs(G) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 + WTConv（当前标准 Haar + bgCE） | 74.897 | 70.832 | 72.808 | 57.242 | 77.081 | 38.665 | 78.393 | 3.640674 | 3.353103 |
| SegFormer-B0 + Aligned + WTConv（AW + bgCE） | 73.159 | 74.734 | 73.938 | 58.652 | 77.789 | 37.868 | 79.771 | 3.711458 | 3.732400 |
| 历史 SegFormer-B0 | 72.891 | 72.448 | 72.668 | 57.070 | 76.948 | — | — | 3.715170 | — |

当前 WTConv-only 相对历史 SegFormer-B0：Precision +2.006 个百分点，F1 +0.140，IoU +0.172，mIoU +0.134；Recall −1.615 个百分点。相对 AW + bgCE，它 Precision 更高，但 Recall、F1、IoU、mIoU 和 hard clDice 更低，说明 Aligned 主要补充跨层空间对应和区域连续性。

## 历史 WTConv-only 结果的口径

之前文件夹中的 WTConv-only 正式结果使用旧 AvgPool+Depthwise `candidate: wtconv`、普通 CE：P72.657%、R73.652%、F1 73.151%、IoU57.668%、mIoU77.258%、Params3.653346M、FLOPs3.391823G。它可以作为历史轻量候选，但不能与本报告的标准 Haar WTConv 混写。

## VOCdevkit3000 当前标准 Haar WTConv-only

VOCdevkit3000 使用 train=2345、val=587、test.txt为空。当前标准 Haar + bgCE 普通 val 结果：Precision 87.598%、Recall 91.077%、F1 89.304%、裂缝 IoU80.674%、mIoU89.743%、BF1@2 70.421%、hard clDice90.740%、Params3.640674M、FLOPs3.353103G。

与旧 VOC WTConv-only（旧 candidate + plain CE）相比，当前版本 Precision、Recall、F1、IoU、mIoU、BF1@2 小幅提高，test 不存在，所有数值均为 587 张 val 结果。

## 复现证据

- Crack500 训练 exit：`results/WTConv_bgCE_s42_train_exit.json`
- Crack500 选模：`results/WTConv_bgCE_s42_selection.json`
- Crack500 val：`results/WTConv_bgCE_s42_val_ordinary_exact.json`
- Crack500 test：`results/WTConv_bgCE_s42_test_exact.json`
- VOCdevkit3000 训练与 val：`D:/论文复现/experiments/segformer_voc_wtconv_bgce_20261112/results/WTConv_bgCE_s42_*`

六个公开模型历史对照仍见 Crack500 AW 六模型 CSV；由于初始化、loss、EMA 和后端不完全一致，只作公开参考。

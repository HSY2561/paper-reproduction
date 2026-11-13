# SegFormer-B0 + WTConv 自建数据集训练报告（2026-11-12）

## 1. 结果结论

本轮重新训练的是当前 AW 主线中的 **标准 Haar `wtconv_wavelet` WTConv-only**：`alignment=none`、Haar DWT/IWT、levels=1、kernel=3，并采用当前的背景加权 CE `[1.10, 1.0]`。它不是旧目录中 AvgPool+Depthwise 的 `candidate: wtconv` 结果。

训练完成 **18,000 iter，exit 0**。普通 checkpoint 的精确 val mIoU 为 89.7434%，高于 EMA 的 89.6932%，因此按固定规则选择普通 checkpoint。

## 2. 自建数据集说明

- 数据目录：`C:/voc3000_stage_20261112`，对应用户的 VOCdevkit3000 自建裂缝数据。
- train：2,345 张；val：587 张；test.txt 为空。
- 原始数据经过只读 staging，标签已规范为 0=背景、1=裂缝；原始 Desktop 数据未修改。
- 因无 test 清单，本报告只报告 val，不能称为 test 结果。

## 3. 当前标准 Haar WTConv + 背景加权 CE 结果

| Variant | Precision (%) | Recall (%) | F1 (%) | 裂缝 IoU (%) | mIoU (%) | BF1@2 (%) | hard clDice (%) | r≤1 细线 Recall (%) | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 + WTConv（标准 Haar，bgCE） | **87.598** | **91.077** | **89.304** | **80.674** | **89.743** | **70.421** | 90.740 | **84.327** | **3.640674** | **3.353103** |

参数量和 FLOPs 来自当前源代码实际构造统计；模型选择只使用 val mIoU，test 未使用。

## 4. 与之前自建集旧 WTConv 结果的区别

之前文件夹中的正式 C2 结果使用的是旧 `candidate: wtconv`（AvgPool+Depthwise）和普通 CE：Precision 87.596%、Recall 90.936%、F1 89.235%、IoU 80.562%、mIoU 89.684%、BF1@2 69.506%、hard clDice 90.909%、Params 3.653346M、FLOPs 3.391823G。

本轮当前标准 Haar + bgCE 相对该旧结果：Precision +0.002、Recall +0.142、F1 +0.069、IoU +0.113、mIoU +0.060、BF1@2 +0.915 个百分点；参数减少约 0.012672M，FLOPs 减少约 0.038721G。hard clDice 和最细裂缝 Recall 略低，说明区域/边界改善不等于拓扑指标全面提升。

## 5. Crack500 对照口径

之前 Crack500 文件夹中已有的 SegFormer-B0+WTConv-only 正式结果同样是旧 `candidate: wtconv`，不是本轮标准 Haar 版本：Precision 72.657%、Recall 73.652%、F1 73.151%、IoU 57.668%、mIoU 77.258%、BF1@2 37.988%、Params 3.653346M、FLOPs 3.391823G。

同轮旧模型对照为：SegFormer-B0 mIoU 76.943%、Aligned mIoU 77.739%、Aligned+WTConv mIoU 77.836%。旧 WTConv-only 参数和 FLOPs低于基线，IoU/mIoU小幅提高，但 Precision 略降。

这些 Crack500 数字应标注为旧 WTConv candidate 的历史结果。当前标准 Haar WTConv-only 尚未在 Crack500 单独重训，不能把本次自建集结果直接写成 Crack500 标准 Haar 结果。

## 6. 复现证据

- 训练 exit：`results/WTConv_bgCE_s42_train_exit.json`
- checkpoint 选择：`results/WTConv_bgCE_s42_selection.json`
- 精确 val：`results/WTConv_bgCE_s42_val_ordinary_exact.json`
- EMA val：`results/WTConv_bgCE_s42_val_ema_exact.json`
- 配置：`configs/WTConv_bgCE.yml`
- 普通 checkpoint SHA256：`2e88eecc58a673ee650238d593d1788975f3b095516bf1da241c3fb9c669a058`
- 训练协议：`EXPERIMENT_PLAN.md`

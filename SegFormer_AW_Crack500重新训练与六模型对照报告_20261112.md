# Crack500 SegFormer-B0 + Aligned + Wavelet WTConv（AW）重新训练与六模型对照报告

## 当前主线更新：AW + background-weighted CE

当前论文主线采用 AW 结构配合背景加权 CE `[1.10, 1.0]`。冻结 ordinary checkpoint 的 Crack500 test 结果为：Precision 73.1589%、Recall 74.7343%、F1 73.9382%、裂缝 IoU 58.6524%、mIoU 77.7894%、Params 3.711458M。完整公开对照表见 `results/AW_CEbg110_vs_six_public_test.csv`，旧 raw AW 行保留作为结构诊断对照。

本轮重新训练标准一层正交 Haar DWT/IWT 版本 `candidate: wtconv_wavelet`。训练与选模只使用 Crack500 train/val，普通 checkpoint 和 EMA checkpoint 均独立验证，冻结后才访问 test。

## 1. 固定协议

RTX5060、PaddlePaddle 3.4.0、FP32、seed=42、batch=4、AdamW、初始学习率 6e-5、PolynomialDecay end_lr=0、CE、EMA、18,000 iter、400×400 裁剪与既有增强。train/val/test 数量为 1896/348/1124。

AW 结构为完整 Aligned 融合后接 WTConvWavelet(levels=1, kernel=3)，包含 LL/LH/HL/HH Haar 分解、逐子带 depthwise filtering、逆小波重建和残差局部细节分支。

## 2. ordinary/EMA 验证选模

| checkpoint | val mIoU | crack IoU | Precision | Recall | F1 | BF1@2px | hard clDice |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary（冻结） | 82.7961 | 67.8981 | 79.4567 | 82.3555 | 80.8801 | 42.9652 | 86.8174 |
| EMA | 82.7684 | 67.7948 | 81.2598 | 80.3588 | 80.8068 | 44.3501 | 86.2601 |

ordinary 精确 val mIoU=0.8279610579，EMA=0.8276841374，因此冻结 ordinary。

## 3. AW 的 Crack500 test 结果

| Model | Precision (%) | Recall (%) | F1-score (%) | IoU (%) | mIoU (%) | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **AW** | **71.1433** | **76.2227** | **73.5955** | **58.2222** | **77.5142** | 3.710818 | 3.732400 |

AW test BF1@2px=36.7947%，hard clDice=79.8764%，最细裂缝 Recall(r≤1)=41.9703%。

## 4. 与六个公开模型的 Crack500 对照

这里按当前主线将“六个公开数据”解释为六个公开模型在 Crack500 上的冻结 test 记录；它们的历史后端、初始化、损失和 EMA 与 AW 不完全相同，因此只能作为当前复现范围内的对照。

| Model | Precision (%) | Recall (%) | F1-score (%) | IoU (%) | mIoU (%) | Params (M) | 数据状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| SegFormer-B0 + Aligned + Wavelet WTConv (AW, ours) | 71.1433 | 76.2227 | 73.5955 | 58.2222 | 77.5142 | 3.710818 | new AW seed42 test |
| SegFormer-B0 | 72.8905 | 72.4478 | 72.6684 | 57.0702 | 76.9475 | 3.715170 | historical frozen public comparison |
| OCRNet-HRNet-W18 | 70.3819 | 75.9782 | 73.0731 | 57.5710 | 77.1509 | 12.109902 | historical frozen public comparison |
| PP-LiteSeg-STDC1 | 74.6053 | 69.7870 | 72.1157 | 56.3914 | 76.6260 | 8.039122 | historical frozen public comparison |
| PSPNet-ResNet50 | 72.1991 | 70.6710 | 71.4269 | 55.5535 | 76.1313 | 67.902052 | historical frozen public comparison |
| DeepLabV3P-ResNet50 | 72.2276 | 73.5601 | 72.8877 | 57.3412 | 77.0754 | 26.789874 | historical frozen public comparison |
| Fast-SCNN | 75.0411 | 61.4485 | 67.5680 | 51.0208 | 73.8029 | 1.441332 | historical frozen public comparison |

## 5. AW 相对历史 SegFormer-B0

Precision -1.7472 pp，Recall +3.7749 pp，F1 +0.9271 pp，裂缝 IoU +1.1520 pp，mIoU +0.5667 pp，参数量 -0.004352 M。

AW 在当前六模型表中裂缝 IoU、mIoU 和 F1 均高于六个历史公开模型；Recall 高于 OCRNet、DeepLabV3P、PP-LiteSeg、PSPNet 和 Fast-SCNN，但 Precision 低于 PP-LiteSeg、Fast-SCNN 和原始 SegFormer-B0。

## 6. 与旧 AvgPool WTConv 结果的关系

旧 Wave17 C3 的 WTConv 是 AvgPool + depthwise candidate，不能当作标准 Haar WTConv。新 AW 必须使用本轮新 checkpoint；两者实现和 checkpoint 不同。当前单 seed AW test 指标低于旧 C3 历史记录，说明标准化小波实现不等于自动提升精度，后续应通过多种子和逐图分析确认稳定性。

## 7. 证据索引

- 训练 exit：`D:\论文复现\experiments\segformer_crack500_aw_wavelet_20261112\results\AW_s42_train_exit.json`
- 选模记录：`D:\论文复现\experiments\segformer_crack500_aw_wavelet_20261112\results\AW_s42_selection.json`
- 精确 val：`D:\论文复现\experiments\segformer_crack500_aw_wavelet_20261112\results\AW_s42_val_exact.json`
- 精确 test：`D:\论文复现\experiments\segformer_crack500_aw_wavelet_20261112\results\AW_s42_test_exact.json`
- 六模型 CSV：`D:\论文复现\experiments\segformer_crack500_aw_wavelet_20261112\results\AW_vs_six_public_test.csv`
- 协议：`D:\论文复现\experiments\segformer_crack500_aw_wavelet_20261112\EXPERIMENT_PLAN.md`

当前结论限定为 Crack500 本实验协议内的比较，不宣称领域 SOTA。

附注：执行期间检测到第二条同一 AW 训练入口，已在产生正式结果前停止并保留 provenance/duplicate_interruption.json；正式结果只来自 	raining/AW_s42，没有拼接或混用两次训练。

# DeepCrack AW + 背景加权 CE 四模型消融报告（2026-11-12）

## 数据与协议

- 数据根目录：`C:/Users/user/Desktop/segformer_aw_cross_dataset_wave17_20261106/normalized/DeepCrack`。
- 文件清单：train=255、val=45、test=237；本轮不修改清单。
- seed=42、FP32、batch=4、18,000 iter、AdamW、初始学习率6e-5、PolynomialDecay到0、EMA。
- 训练增强与 Crack500 AW 相同：ResizeStepScaling、400×400 RandomPaddingCrop、水平翻转、RandomDistort、Normalize。
- 四组都使用相同背景加权 CE `[1.10, 1.0]`，只改变结构；普通/EMA按精确 val mIoU 选模，冻结后评估 test。

## DeepCrack test 四组对照

| Variant | Precision (%) | Recall (%) | F1 (%) | IoU (%) | mIoU (%) | BF1@2 (%) | hard clDice (%) | Params (M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | 93.2679 | 78.4915 | 85.2441 | 74.2830 | 86.5329 | 84.9304 | 89.3151 | 3.715170 |
| SegFormer-B0 + Aligned | 93.6107 | 77.6085 | 84.8618 | 73.7043 | 86.2323 | 84.4387 | 88.9637 | 3.785314 |
| SegFormer-B0 + WTConv | 92.3970 | 80.1786 | 85.8553 | 75.2161 | 87.0159 | 85.1164 | 89.5346 | 3.641314 |
| SegFormer-B0 + Aligned + WTConv | 93.8108 | 77.7272 | 85.0150 | 73.9357 | 86.3544 | 84.3992 | 89.0117 | 3.711458 |

## 模块贡献解释

- C0 是同协议结构基线。
- C1 相对 C0 的变化用于估计 Aligned 的空间对齐贡献。
- C2 相对 C0 的变化用于估计 Haar-WTConv 的上下文与局部细节贡献。
- C3 同时启用两者，用于判断联合效果是否超过单模块，并观察是否存在互补或指标折中。
- 本表是单种子 DeepCrack 结果，不能据此宣称领域 SOTA；若要写稳定性结论，还需 seed43/44。

## Crack500 六模型公开对照

当前 AW + 背景加权 CE 与六个公开模型的统一历史 test 对照见：`../segformer_crack500_aw_wavelet_20261112/results/AW_CEbg110_vs_six_public_test.csv`。
六个公开模型为 SegFormer-B0、OCRNet-HRNet-W18、PP-LiteSeg-STDC1、PSPNet-ResNet50、DeepLabV3P-ResNet50、Fast-SCNN。它们是历史冻结记录，训练后端、初始化、loss 和 EMA 不完全一致，因此只作公开参考。

| Model | Precision (%) | Recall (%) | F1 (%) | IoU (%) | mIoU (%) | Params (M) |
|---|---:|---:|---:|---:|---:|---:|
| AW + background-weighted CE | 73.1589 | 74.7343 | 73.9382 | 58.6524 | 77.7894 | 3.711458 |
| AW + background-weighted CE + threshold 0.525 | 74.4011 | 72.9438 | 73.6653 | 58.3096 | 77.6350 | 3.711458 |
| SegFormer-B0 | 72.8905 | 72.4478 | 72.6684 | 57.0702 | 76.9475 | 3.715170 |
| OCRNet-HRNet-W18 | 70.3819 | 75.9782 | 73.0731 | 57.5710 | 77.1509 | 12.109902 |
| PP-LiteSeg-STDC1 | 74.6053 | 69.7870 | 72.1157 | 56.3914 | 76.6260 | 8.039122 |
| PSPNet-ResNet50 | 72.1991 | 70.6710 | 71.4269 | 55.5535 | 76.1313 | 67.902052 |
| DeepLabV3P-ResNet50 | 72.2276 | 73.5601 | 72.8877 | 57.3412 | 77.0754 | 26.789874 |
| Fast-SCNN | 75.0411 | 61.4485 | 67.5680 | 51.0208 | 73.8029 | 1.441332 |

阈值 0.525 是验证集诊断得到的推理策略，不是新模型。公开六模型行是历史冻结记录，不能与当前 AW 结果做严格同后端 SOTA 声明。

## 复现索引

- 训练状态/exit：`results/*_s42_train_status.json`、`results/*_s42_train_exit.json`。
- val 选择：`results/*_s42_val_ordinary_exact.json`、`results/*_s42_val_ema_exact.json`、`results/*_s42_selection.json`。
- test 结果：`results/*_s42_test_exact.json`。
- test 核验：四组均使用 test.txt 的 237 张图；评估配置为 `configs/*_test_eval.yml`，仅将 backbone pretrained 置空以绕过中文路径编码，checkpoint 完整加载并通过 state key/shape 校验。原始配置失败证据保留。
- 参数：`results/param_counts.json`。
- 配置与运行脚本：`configs/`、`run_deepcrack_ablation.ps1`。

# Crack500 SegFormer-B0 + Aligned + Haar-WTConv（AW）

本轮仅验证新实现 `candidate: wtconv_wavelet`，即 SegFormer-B0 + Aligned + 一层正交 Haar DWT/IWT 细节模块。旧 `candidate: wtconv` 的 Wave17 结果保留为历史 candidate，不复用 checkpoint。

## 固定协议

- 数据：`C:/Users/user/algolab_data`；训练/验证清单沿用 `train.txt`、`val.txt`；只在冻结验证 checkpoint 后访问 `test.txt`。
- 模型：MiT-B0、embedding_dim=256、Aligned full、wavelet_levels=1、wavelet_kernel=3。
- 训练：seed=42、FP32、batch=4、AdamW、learning rate `6e-5`、PolynomialDecay 到 0、CE、EMA、18,000 iter、400×400 RandomPaddingCrop、0.75–1.25 ResizeStepScaling、同一增强。
- 选模：普通 checkpoint 和 EMA checkpoint 中以精确 val mIoU 较高者冻结；记录 checkpoint SHA256。
- 评估：官方 val、精确零外延 1/2 px 边界 F1、逐图 CSV、hard clDice、半径≤1 细线 Recall；冻结后才统一 test。

## 对照

六个公开模型复用当前 Crack500 统一表中的合法冻结记录：SegFormer-B0、OCRNet-HRNet-W18、PP-LiteSeg-STDC1、PSPNet-ResNet50、DeepLabV3P-ResNet50、Fast-SCNN。历史公开模型的训练后端/初始化/loss/EMA存在差异，比较中如实披露。

AW 新训练完成前，旧 C3 指标不能作为 AW 结果。

# DeepCrack AW + 背景加权 CE 四组消融

固定四组：SegFormer-B0（C0）、SegFormer-B0+Aligned（C1）、SegFormer-B0+Haar-WTConv（C2）、SegFormer-B0+Aligned+Haar-WTConv（C3）。四组均使用当前 AW 训练协议和 CrossEntropyLoss 权重 `[1.10, 1.0]`，只改变结构。数据采用标准化 DeepCrack 清单：train 255、val 45、test 237，不修改清单。

训练：seed=42、FP32、batch=4、18,000 iter、AdamW、初始学习率 6e-5、PolynomialDecay 到 0、EMA、400x400 裁剪及 Crack500 同增强。普通/EMA 按精确 val mIoU 选模，冻结后评估 test。test 只用于冻结后的最终对照。

当前源代码固定为 `experiments/segformer_crack500_aw_wavelet_20261112/src/PaddleSeg-2.10.0`，使用标准 Haar DWT/IWT 的 `wtconv_wavelet`；不复用旧波18失败训练结果，不拼接日志。

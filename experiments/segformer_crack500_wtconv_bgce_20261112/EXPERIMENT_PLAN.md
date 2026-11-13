# Crack500 当前标准 Haar WTConv-only

固定 SegFormer-B0 + `alignment=none` + `candidate=wtconv_wavelet`，使用标准 Haar DWT/IWT（levels=1、kernel=3）和背景加权 CE `[1.10,1.0]`。训练 seed42、FP32、batch4、18,000 iter、AdamW、6e-5、PolynomialDecay 到0、EMA；按精确 val mIoU 选择普通/EMA，冻结后评估 test。与旧 `candidate: wtconv` 结果分开保存。

# VOCdevkit3000 SegFormer-B0 + 标准 Haar WTConv + 背景加权 CE

本轮只训练当前 AW 方案中的 WTConv-only 结构：SegFormer-B0、alignment=none、candidate=wtconv_wavelet（Haar DWT/IWT，levels=1、kernel=3）。背景加权 CE 为 `[1.10, 1.0]`，其余训练协议与 Crack500 AW 完全一致。

数据使用标准化 VOCdevkit3000：train=2345、val=587，test.txt 为空，因此只报告 val，不生成 test 结论。seed=42、FP32、batch=4、18,000 iter、AdamW、6e-5、PolynomialDecay 到 0、EMA。普通/EMA 以精确 val mIoU 选模。

# AW 精度诊断与单变量修正报告（2026-11-12）

详见实验目录中的完整报告：

[AW精度诊断与单变量修正报告_20261112.md](D:/论文复现/experiments/segformer_crack500_aw_wavelet_20261112/AW精度诊断与单变量修正报告_20261112.md)

核心结果：固定 SegFormer-B0 + Aligned + Haar WTConv，不叠加新模块；背景加权 CE 将 test Precision 从 71.143% 提升到 73.159%，验证集确定阈值 0.525 后达到 74.401%，但 Recall 和极细裂缝召回存在代价。



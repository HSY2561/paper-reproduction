# Aligned母体第二模块消融进度

更新时间：2026-10-19T09:10:12+08:00

所有结果只使用Crack500验证集。待完成、兼容性失败不填性能；旧运行速度仅为临时测量。

| 编号 | 模块 | 状态 | P% | R% | F1% | IoU% | mIoU% | BF1% | Params M |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 复用冻结对照 | 已核验 | 82.827 | 78.913 | 80.823 | 67.817 | 82.802 | 45.152 | 3.715170 |
| W128 | 复用冻结对照 | 已核验 | 80.275 | 81.142 | 80.706 | 67.653 | 82.678 | 44.029 | 3.451746 |
| A0 | 复用冻结对照 | 已核验 | 82.677 | 79.097 | 80.847 | 67.852 | 82.818 | 44.928 | 3.470434 |
| GloRe | 复用冻结对照 | 已核验 | 81.047 | 80.513 | 80.779 | 67.756 | 82.745 | 44.436 | 3.498978 |
| A01_GloRe | GloRe Graph Reasoning Unit | complete | 82.330 | 79.493 | 80.887 | 67.907 | 82.843 | 44.972 | 3.517666 |
| A02_ASPPsep | Separable Atrous Spatial Pyramid Pooling | complete | 81.342 | 80.685 | 81.012 | 68.085 | 82.923 | 44.762 | 3.574882 |
| A03_Encoding | Context Encoding Module | complete | 80.340 | 81.237 | 80.786 | 67.765 | 82.739 | 43.886 | 3.607218 |
| A04_ExternalAttention | RTFormer External Attention | complete | 81.423 | 80.612 | 81.015 | 68.089 | 82.927 | 44.982 | 3.504226 |
| A05_ContextEmbedding | BiSeNetV2 Context Embedding Block | complete | 82.227 | 78.349 | 80.242 | 67.003 | 82.362 | 43.699 | 4.128354 |
| A06_DNL | Disentangled Non-Local Block | complete | 82.197 | 79.396 | 80.772 | 67.746 | 82.756 | 45.184 | 3.537443 |
| A07_PPContext | PP-LiteSeg Simple Pyramid Context Module | complete | 82.367 | 78.774 | 80.530 | 67.407 | 82.578 | 44.501 | 3.520706 |
| A08_DAPPM | DDRNet Deep Aggregation Pyramid Pooling Module | smoke_failed | — | — | — | — | — | — | — |
| A09_APNB | Asymmetric Pyramid Non-local Block | complete | 81.872 | 80.457 | 81.159 | 68.292 | 83.041 | 44.851 | 3.652578 |
| A10_AFNB | Asymmetric Fusion Non-local Block | complete | 82.560 | 79.633 | 81.070 | 68.166 | 82.984 | 45.106 | 3.656994 |
| A11_JPU | FastFCN Joint Pyramid Upsampling | complete | 81.337 | 80.387 | 80.859 | 67.869 | 82.809 | 44.403 | 3.545570 |
| A12_DCM | DMNet Dynamic Convolutional Module | complete | 82.098 | 79.788 | 80.926 | 67.963 | 82.870 | 45.125 | 3.670882 |
| A13_EMAU | Expectation-Maximization Attention Unit | complete | 82.907 | 78.869 | 80.837 | 67.838 | 82.814 | 44.853 | 3.470434 |
| A14_NonLocal | Embedded-Gaussian Non-Local Block | complete | 81.419 | 80.623 | 81.019 | 68.094 | 82.930 | 44.701 | 3.537186 |
| A15_AttentionRefinement | BiSeNetV1 Attention Refinement Module | complete | 82.279 | 79.511 | 80.872 | 67.886 | 82.831 | 44.947 | 3.635298 |
| A16_CAM | DANet Channel Attention Module | complete | 81.115 | 80.844 | 80.979 | 68.038 | 82.896 | 44.704 | 3.470435 |
| A17_CCA | Criss-Cross Attention | complete | 79.758 | 82.642 | 81.175 | 68.314 | 83.022 | 44.356 | 3.552675 |
| A18_PAM | DANet Position Attention Module | complete | 80.117 | 82.314 | 81.200 | 68.351 | 83.047 | 44.272 | 3.552675 |
| A19_MSCA | MSCAN Multi-Scale Convolutional Attention | complete | 81.812 | 80.223 | 81.009 | 68.081 | 82.928 | 45.258 | 3.534050 |
| A20_UAFMChannel | PP-LiteSeg Channel-Attention UAFM | complete | 82.053 | 80.234 | 81.133 | 68.256 | 83.024 | 44.838 | 3.808098 |
| A21_Polarized | Polarized Self-Attention | complete | 82.235 | 79.815 | 81.007 | 68.077 | 82.932 | 45.154 | 3.503330 |
| A22_Hamburger | SegNeXt Hamburger NMF Context | complete | 81.974 | 80.285 | 81.120 | 68.237 | 83.013 | 45.261 | 3.503586 |
| A23_SE | MobileNetV3 Squeeze-and-Excitation | complete | 81.632 | 80.498 | 81.061 | 68.154 | 82.964 | 45.383 | 3.478786 |
| A24_PSP | PSPNet Pyramid Pooling Module | compatibility_deferred | — | — | — | — | — | — | — |

24个登记机制、23个当前可尝试项，PSP兼容性暂缓；数量以真实exit0/18k/冻结/独立验证完成数为准。
每项相对Aligned母体和原始B0分别归因；速度确认前full_target_pass为空。筛选结束后匹配三seed、统一同机速度、六个原始模型及最终test。

本轮计划：`SegFormer_Aligned第二模块消融执行计划_20261018.md`。逐图/空间归因在活动工程`diagnostics`中；速度交错复测仍有波动，未确认加速。

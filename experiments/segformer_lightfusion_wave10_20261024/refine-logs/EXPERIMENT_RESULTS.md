# SegFormer-B0+Aligned 轻量多尺度融合波次10

更新时间：2026-10-24T21:20:01+08:00

筛选仅使用Crack500 train/val；A0为wave7独立冻结控制。

| ID | 候选 | 状态 | P | R | F1 | crackIoU | mIoU | BF1 | Params M | Latency480 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | width128 + Aligned（复用） | complete_reused | 0.826766 | 0.790974 | 0.808474 | 0.678520 | 0.828184 | 0.449285 | 3.470434 | 17.940 |
| W96 | Aligned base width96 | complete | 0.824722 | 0.795839 | 0.810023 | 0.680705 | 0.829317 | 0.447854 | 3.417314 | 7.602 |
| W80 | Aligned base width80 | complete | 0.822110 | 0.794180 | 0.807903 | 0.677716 | 0.827695 | 0.448944 | 3.394594 | 7.858 |
| W64 | Aligned base width64 | complete | 0.829552 | 0.786743 | 0.807580 | 0.677262 | 0.827554 | 0.450029 | 3.374434 | 8.483 |
| F01_Sum128 | four-branch sum + 1x1 | complete | 0.811131 | 0.805539 | 0.808325 | 0.678310 | 0.827855 | 0.444218 | 3.421282 | 8.018 |
| F02_Weighted128 | learned scalar weighted sum | complete | 0.812977 | 0.802299 | 0.807603 | 0.677293 | 0.827342 | 0.433333 | 3.421286 | 8.979 |
| F03_ChannelWeighted128 | channel weighted sum | complete | 0.807820 | 0.809001 | 0.808410 | 0.678430 | 0.827870 | 0.443934 | 3.421794 | 7.656 |
| F04_Pair128 | pairwise low/high sum | complete | 0.822810 | 0.800163 | 0.811329 | 0.682551 | 0.830271 | 0.449252 | 3.437666 | 6.811 |
| F05_Hierarchical128 | hierarchical pair fusion | complete | 0.820640 | 0.796778 | 0.808533 | 0.678604 | 0.828145 | 0.448843 | 3.437666 | 7.569 |
| F06_LowRank32 | low-rank concat rank32 | complete | 0.809419 | 0.806045 | 0.807729 | 0.677470 | 0.827384 | 0.447772 | 3.425506 | 6.918 |
| F07_LowRank16 | low-rank concat rank16 | complete | 0.800260 | 0.819208 | 0.809623 | 0.680141 | 0.828665 | 0.441308 | 3.415202 | 6.972 |
| F08_Branch32 | per-branch bottleneck rank32 | complete | 0.822632 | 0.805304 | 0.813876 | 0.686164 | 0.832187 | 0.460658 | 3.438178 | 7.593 |
| F09_Branch16 | per-branch bottleneck rank16 | complete | 0.793183 | 0.829427 | 0.810900 | 0.681944 | 0.829514 | 0.440265 | 3.421538 | 8.723 |
| F10_DWSum128 | depthwise sum fusion | complete | 0.817961 | 0.795789 | 0.806723 | 0.676057 | 0.826756 | 0.437183 | 3.422946 | 7.790 |
| F11_Separable128 | separable sum fusion | complete | 0.792587 | 0.816073 | 0.804159 | 0.672463 | 0.824467 | 0.423426 | 3.439970 | 7.985 |
| F12_GatedPair128 | learned gated pair fusion | complete | 0.818446 | 0.803489 | 0.810899 | 0.681943 | 0.829888 | 0.448669 | 3.437670 | 7.835 |
| F13_Sum96 | sum fusion width96 | complete | 0.812230 | 0.804713 | 0.808454 | 0.678492 | 0.827967 | 0.444816 | 3.389666 | 8.534 |
| F14_LowRank96 | low-rank concat width96 rank24 | complete | 0.793583 | 0.829116 | 0.810961 | 0.682030 | 0.829565 | 0.431744 | 3.392066 | 20.616 |
| F15_DWSum96 | depthwise sum width96 | pending | — | — | — | — | — | — | — | — |
| F16_Sum64 | sum fusion width64 | pending | — | — | — | — | — | — | — | — |
| F17_Branch64 | branch bottleneck width64 rank16 | pending | — | — | — | — | — | — | — | — |

候选均为新波融合配置或宽度控制，不把旧wave7注意力加法模块重复计数。lightweight_gate需要精度门槛和参数≤A0的实测证据。
未完成/失败候选不填性能；不得据此声称SOTA。

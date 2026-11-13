# DeepCrack 四模型消融结果（AW + 背景加权 CE）

数据使用 normalized DeepCrack：train 255、val 45、test 237。四组均使用 seed42、FP32、batch4、18,000 iter、AdamW 6e-5、PolynomialDecay、EMA、相同增强和背景加权 CE [1.10,1.0]。test 仅在 val mIoU 冻结模型后评估。

| Variant | Val mIoU | Precision | Recall | F1 | IoU | mIoU | BF1@2 | hard clDice | Params(M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_B0 | 88.103 | 93.268 | 78.491 | 85.244 | 74.283 | 86.533 | 84.930 | 89.315 | 3.715170 |
| C1_Aligned | 88.097 | 93.611 | 77.609 | 84.862 | 73.704 | 86.232 | 84.439 | 88.964 | 3.785314 |
| C2_WTConvWavelet | 88.257 | 92.397 | 80.179 | 85.855 | 75.216 | 87.016 | 85.116 | 89.535 | 3.641314 |
| C3_Aligned_WTConvWavelet | 88.277 | 93.811 | 77.727 | 85.015 | 73.936 | 86.354 | 84.399 | 89.012 | 3.711458 |

## 结论

C0 为原始 SegFormer-B0；C1 单独加入 Aligned；C2 单独加入标准 Haar WTConv；C3 为 Aligned+WTConv 联合，并沿用背景加权 CE。

DeepCrack test 上 C2 的区域和连通性最好：IoU 75.216%、mIoU 87.016%、F1 85.855%、BF1@2 85.116%、hard clDice 89.535%。C3 的 Precision 最高（93.811%），但 IoU、mIoU、F1 和 BF1@2 低于 C0/C2，说明联合模型在该跨数据集上更偏向减少误检，同时牺牲召回和区域覆盖；不能声称 Aligned 与 WTConv 在 DeepCrack 上产生正交互。

C1 相比 C0 的 test mIoU 和 IoU 下降，说明 Aligned 的收益依赖数据域；C2 相比 C0 的 test IoU 提升 0.933 个百分点、mIoU 提升 0.483 个百分点，WTConv 的跨域收益更稳定。

评估配置中的 backbone pretrained 已置空，仅用于避免中文路径编码导致的评估构造失败；所有 checkpoint 完整加载并通过 state key/shape 校验，不影响训练权重。原始失败日志保留。

详细证据位于 results/C*_s42_selection.json、results/C*_s42_val_ordinary_exact.json、results/C*_s42_val_ema_exact.json、results/C*_s42_test_exact.json。

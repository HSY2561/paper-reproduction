# 4.1 Real-time crack segmentation model / 实时裂缝分割模型

## 中文

面向四轮差速底盘的长裂缝循迹和三轴机构的定点精修，视觉分割结果需要同时满足区域精度、边界保持、裂缝连通性和在线推理效率。本文以轻量级 SegFormer-B0 为基线。该模型具有较好的参数效率，但其多尺度特征在直接上采样和融合时存在空间对应偏差，高层语义经过恢复后也可能削弱细裂缝、端点和局部分支。因此，在低对比度、复杂纹理和宽度变化较大的道路表面上，基线模型容易出现裂缝断裂、边界偏移和背景误检。

针对上述问题，本文在 SegFormer-B0 的多尺度解码阶段加入 Aligned 特征恢复模块。该模块以高分辨率特征为目标网格，估计高层特征到目标网格的局部偏移，并通过双线性重采样完成空间对齐，从而减少尺度变换造成的边界和端点错位。Aligned 主要服务于细裂缝定位、裂缝连通性恢复以及后续三轴修复坐标的稳定生成。

在对齐后的融合瓶颈处，本文进一步加入 WTConv candidate 细节增强模块。该实现包含低频上下文分支和高频局部分支：低频分支通过下采样和恢复扩大有效感受野，用于保持长裂缝的连续区域；高频分支采用深度可分离卷积增强边缘、端点和局部纹理响应。该模块主要用于补充长距离上下文和局部结构表征。需要说明的是，当前工程中的 WTConv candidate 由 AvgPool2D、depthwise 5×5、pointwise 和残差结构组成，因此本文将其作为工程候选模块记录，避免将其表述为未经核对的原始小波分解实现。

最终模型由 MiT-B0 编码器、Aligned 跨尺度特征恢复、WTConv candidate 瓶颈增强和轻量级分割头组成。四个编码阶段的特征先映射到统一嵌入维度，再进行尺度对齐、拼接和通道压缩，最后由 1×1 分类头输出裂缝与背景概率图。训练采用像素级交叉熵损失；本实现未保留额外辅助预测头，推理阶段仅执行主分割路径，因此新增模块的在线代价可以由参数量、FLOPs 和目标硬件延迟直接衡量。

## English

For long-crack tracking with a differential-drive platform and point repair with a three-axis mechanism, the segmentation output must provide accurate regions, reliable boundaries, connected crack structures, and low online inference cost. We therefore use the lightweight SegFormer-B0 as the baseline. Although SegFormer-B0 is parameter-efficient, direct upsampling and fusion of its multi-scale features may introduce spatial misalignment, while the recovery of high-level features can weaken thin cracks, endpoints, and local branches. These limitations are more evident on road surfaces with low contrast, dense textures, and large variations in crack width, where the baseline may produce broken structures, shifted boundaries, and false positives.

To address the cross-scale correspondence problem, we insert an Aligned feature-restoration module into the SegFormer-B0 decoder. The module uses the high-resolution feature grid as the target coordinate system, estimates local offsets from high-level features to this grid, and performs bilinear resampling. This operation reduces the spatial displacement of crack boundaries and endpoints caused by scale changes. Aligned is mainly intended to improve thin-crack localization, structural continuity, and the stability of coordinates used by the three-axis repair mechanism.

We further place a WTConv candidate at the fused feature bottleneck. Its low-frequency branch uses downsampling and feature recovery to enlarge the effective context, while its high-frequency branch uses a depthwise separable convolution to enhance edges, endpoints, and local texture responses. The two branches target complementary requirements: long-range continuity and local crack-detail preservation. In the current implementation, the candidate consists of AvgPool2D, depthwise 5×5, pointwise, and residual operations. It is therefore reported as a WTConv candidate rather than being treated as an unverified implementation of the original wavelet-decomposition WTConv.

The final network contains the MiT-B0 encoder, Aligned cross-scale restoration, WTConv-candidate bottleneck enhancement, and a lightweight segmentation head. Features from the four encoder stages are projected to a common embedding dimension, spatially aligned, concatenated, and compressed before a 1×1 classifier predicts crack and background probabilities. Training uses pixel-wise cross-entropy. No auxiliary prediction head is retained in the present implementation; inference follows only the main segmentation path, so the deployment cost is evaluated with parameter count, FLOPs, and target-device latency.

# 4.2 Training setup and experimental results / 训练设置与实验结果

## 中文

为保证比较的一致性，实验使用 Crack500 的 1,896 张训练图、348 张验证图和 1,124 张测试图。所有模型采用相同的数据清单和 400×400 输入流程，在 RTX 5060 上使用 PaddlePaddle 3.4.0、FP32、seed=42、batch size=4、AdamW、初始学习率 (6\times10^{-5})、PolynomialDecay（end\_lr=0）训练 18,000 iter。模型选择只依据验证集 mIoU，并在普通 checkpoint 与 EMA checkpoint 中选择验证 mIoU 较高者；冻结 checkpoint 后才访问测试集。评价指标包括 Precision、Recall、F1-score、裂缝 IoU、mIoU、2 像素容差边界 F1、参数量和 FLOPs。

首先进行模块级消融。C0 为原始 SegFormer-B0，C1 为 SegFormer-B0+Aligned，C2 为仅加入 WTConv candidate，C3 为 Aligned 与 WTConv candidate 的联合模型。测试结果如下。

| Variant | Precision (%) | Recall (%) | F1-score (%) | Crack IoU (%) | mIoU (%) | Boundary F1@2px (%) | Params (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 SegFormer-B0 | 72.8613 | 72.4660 | 72.6631 | 57.0637 | 76.9434 | 37.9749 | 3.715170 |
| C1 + Aligned | 70.9480 | 77.1986 | 73.9414 | 58.6564 | 77.7387 | 37.9813 | 3.785314 |
| C2 + WTConv candidate | 72.6574 | 73.6516 | 73.1511 | 57.6679 | 77.2579 | 37.9876 | 3.653346 |
| C3 + Aligned + WTConv candidate | 70.2578 | **78.4197** | **74.1147** | **58.8748** | **77.8361** | 36.8405 | 3.723490 |

与 C0 相比，Aligned 使 Recall、F1 和裂缝 IoU 分别提高 4.7326、1.2783 和 1.5927 个百分点，说明空间对齐有助于恢复漏检的裂缝区域，但 Precision 下降 1.9134 个百分点。WTConv candidate 的参数量为四组中最低，FLOPs 也低于 C0，同时带来小幅区域指标增益，体现了精度与效率之间的折中。联合模型取得最高的 Recall、F1、裂缝 IoU 和 mIoU，参数量仅比 C0 增加 0.00832 M；但其 Precision 下降 2.6036 个百分点，边界 F1 下降 1.1344 个百分点，说明联合结构扩大了裂缝覆盖范围，同时增加了假阳性和边界粗化风险。因此，C3 的优势应表述为“更高的区域覆盖和综合区域指标”，不能表述为所有指标同时提升。

在公开模型对照中，当前 C3 的 F1、裂缝 IoU、mIoU 和 Recall 高于本实验记录中的 SegFormer-B0、OCRNet-HRNet-W18、PP-LiteSeg-STDC1、PSPNet-ResNet50、DeepLabV3P-ResNet50 和 Fast-SCNN；PP-LiteSeg-STDC1 与 Fast-SCNN 的 Precision 更高，Fast-SCNN 的参数量最低。由于公开模型记录来自不同阶段的冻结结果，初始化、损失、EMA 和运行后端尚未完全统一，表中结论限定为“当前复现范围内的比较”，不能据此声称跨论文意义上的 SOTA。

定性结果应覆盖细裂缝、低对比度或阴影裂缝、纹理密集背景和宽裂缝四类场景。预期的可视化重点是：Aligned 是否减少端点错位和局部断裂，WTConv candidate 是否改善长裂缝上下文与局部细节，联合模型是否在扩大裂缝覆盖的同时引入额外纹理误检。该定性分析应与表中的 Recall、Precision 和边界 F1 共同解释，而不能只展示视觉上最有利的样本。

## English

To ensure a consistent comparison, we use 1,896 Crack500 training images, 348 validation images, and 1,124 test images. All models use the same file lists and 400×400 input pipeline. Training is performed on an RTX 5060 with PaddlePaddle 3.4.0, FP32 arithmetic, seed 42, batch size 4, AdamW, an initial learning rate of (6\times10^{-5}), PolynomialDecay with end\_lr=0, and 18,000 iterations. Checkpoints are selected only by validation mIoU; the ordinary and EMA checkpoints are compared, and the higher-scoring one is frozen before test evaluation. We report Precision, Recall, F1-score, crack IoU, mIoU, 2-pixel-tolerance boundary F1, parameter count, and FLOPs.

We first conduct a module-level ablation. C0 denotes the original SegFormer-B0, C1 adds Aligned, C2 adds only the WTConv candidate, and C3 combines Aligned with the WTConv candidate.

| Variant | Precision (%) | Recall (%) | F1-score (%) | Crack IoU (%) | mIoU (%) | Boundary F1@2px (%) | Params (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 SegFormer-B0 | 72.8613 | 72.4660 | 72.6631 | 57.0637 | 76.9434 | 37.9749 | 3.715170 |
| C1 + Aligned | 70.9480 | 77.1986 | 73.9414 | 58.6564 | 77.7387 | 37.9813 | 3.785314 |
| C2 + WTConv candidate | 72.6574 | 73.6516 | 73.1511 | 57.6679 | 77.2579 | 37.9876 | 3.653346 |
| C3 + Aligned + WTConv candidate | 70.2578 | **78.4197** | **74.1147** | **58.8748** | **77.8361** | 36.8405 | 3.723490 |

Compared with C0, Aligned improves Recall, F1-score, and crack IoU by 4.7326, 1.2783, and 1.5927 percentage points, respectively. This indicates that cross-scale alignment recovers more crack pixels, although Precision decreases by 1.9134 points. The WTConv candidate has the lowest parameter count among the four variants and reduces the computational cost relative to C0 while providing a modest gain in region metrics. The joint model achieves the highest Recall, F1-score, crack IoU, and mIoU, while adding only 0.00832 M parameters over C0. However, its Precision decreases by 2.6036 points and its boundary F1 decreases by 1.1344 points. Thus, the joint model expands crack-region coverage but also increases false positives and boundary coarsening; it should not be described as improving every metric simultaneously.

For the public-model comparison, the current C3 record has higher F1-score, crack IoU, mIoU, and Recall than the available records for SegFormer-B0, OCRNet-HRNet-W18, PP-LiteSeg-STDC1, PSPNet-ResNet50, DeepLabV3P-ResNet50, and Fast-SCNN. PP-LiteSeg-STDC1 and Fast-SCNN provide higher Precision, and Fast-SCNN has the smallest parameter count. Because the public-model records were frozen in different experiment stages, their initialization, loss, EMA policy, and backend are not yet completely identical. The comparison should therefore be described as the best result within the current reproduction scope rather than as cross-paper SOTA.

Qualitative examples should cover thin cracks, low-contrast or shadowed cracks, texture-rich backgrounds, and wide cracks. The visual analysis should examine whether Aligned reduces endpoint displacement and local breaks, whether the WTConv candidate improves long-range context and local details, and whether the joint model introduces extra texture false positives while increasing crack coverage. These observations should be interpreted together with Recall, Precision, and boundary F1 rather than selected from only the most favorable images.

## 写作与证据说明 / Evidence note

本文的数字来自当前 Wave17 Crack500 记录；C3 的边界 F1 低于 C0，因此不应写成“边界指标全面提升”。公开模型表属于当前复现范围内的历史冻结记录，若投稿前需要严格横向结论，应使用完全相同的训练后端、预算、数据清单和选模规则重新冻结并统一测试。

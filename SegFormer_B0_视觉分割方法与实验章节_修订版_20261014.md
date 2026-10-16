# SegFormer-B0 路面裂缝视觉分割：方法、实验与结果分析

> 本文件是论文方法与实验章节的统一修订稿。它把 SegFormer-B0 的结构定义、Crack500 数据审计、受控消融、公开模型对比和后续机器人验证放在同一条证据链中。当前结果均来自 RTX 5060、PaddleSeg 官方实现和固定的 Crack500 crop-split 协议。除非特别注明，指标是全验证集或全测试集聚合混淆矩阵得到的前景裂缝指标。

## 3 Methodology

### 3.1 Problem formulation

给定路面图像 \(I\in\mathbb{R}^{3\times H\times W}\)，模型输出每个像素属于背景或裂缝的概率 \(\hat{p}_{p,c}\)。裂缝掩膜用于两项后续任务：一是沿封闭场地覆盖路径进行巡检，二是在机器人停车后生成局部裂缝修复轨迹。因而模型不仅要识别裂缝主体，还要尽可能保留裂缝的连续性、端点和边界。

### 3.2 SegFormer-B0 encoder

本文采用 SegFormer-B0 的官方 MiT-B0 编码器。MiT 使用四阶段重叠式 patch embedding 和分层高效自注意力，在不同分辨率上提取局部纹理与长程上下文。令 \(F_0=I\)，第 \(i\) 个阶段为

\[
F_i=\operatorname{MiT}_i(F_{i-1}),\qquad
F_i\in\mathbb{R}^{C_i\times H_i\times W_i},\quad i=1,2,3,4.
\]

MiT-B0 的完整配置如下：

| Stage | embed dim \(C_i\) | heads | blocks | sr_ratio | patch size | stride |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1 | 2 | 8 | 7 | 4 |
| 2 | 64 | 2 | 2 | 4 | 3 | 2 |
| 3 | 160 | 5 | 2 | 2 | 3 | 2 |
| 4 | 256 | 8 | 2 | 1 | 3 | 2 |

其余官方配置为 `mlp_ratios=[4,4,4,4]`、`qkv_bias=True`、`drop_path_rate=0.1`。对 400×400 输入，四阶段实际输出为

\[
100\times100\times32
\rightarrow 50\times50\times64
\rightarrow 25\times25\times160
\rightarrow 13\times13\times256.
\]

在自注意力中，空间缩减后的键和值可写为

\[
Q=XW_Q,\qquad K=\operatorname{SR}(X)W_K,\qquad
V=\operatorname{SR}(X)W_V,
\]

\[
\operatorname{Attn}(Q,K,V)=
\operatorname{Softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d}}\right)V,
\]

其中 \(\operatorname{SR}(\cdot)\) 表示由 `sr_ratio` 控制的空间缩减操作。该设计降低了高分辨率阶段的注意力开销，同时保留裂缝的多尺度上下文。

### 3.3 MLP decoder and pixel prediction

各阶段特征首先展平并通过独立线性层投影到统一的 256 维嵌入空间：

\[
E_i=\operatorname{MLP}_i(\operatorname{Flatten}(F_i)),
\qquad E_i\in\mathbb{R}^{(H_iW_i)\times256}.
\]

将其重排为二维特征并上采样到 100×100：

\[
\bar F_i=\operatorname{Reshape}(E_i),\qquad
\tilde F_i=\operatorname{Up}_{100,100}(\bar F_i).
\]

PaddleSeg 中四个尺度按 `[F4, F3, F2, F1]` 拼接，得到 1024 通道特征。随后依次执行融合卷积、归一化、激活、二维 dropout、分类卷积和恢复到输入尺寸的双线性上采样：

\[
F_{\mathrm{cat}}=\operatorname{Concat}[\tilde F_4,\tilde F_3,\tilde F_2,\tilde F_1],
\]

\[
F_{\mathrm{fuse}}=\operatorname{ReLU}\left(
\operatorname{BN}\left(\operatorname{Conv}_{1\times1}^{1024\rightarrow256}(F_{\mathrm{cat}})\right)\right),
\]

\[
F_{\mathrm{drop}}=\operatorname{Dropout2D}(F_{\mathrm{fuse}},p=0.1),
\qquad Z=\operatorname{Conv}_{1\times1}^{256\rightarrow2}(F_{\mathrm{drop}}),
\]

\[
\hat P=\operatorname{Softmax}\left(\operatorname{Up}_{H,W}(Z)\right).
\]

这里的 `ConvBNReLU(1024→256,kernel=1)`、`Dropout2D(p=0.1)` 和 `Conv2D(256→2,kernel=1)` 与实际 PaddleSeg 解码头一致。细裂缝可能在 400×400 裁剪、编码器下采样、跨尺度融合或最终恢复阶段变细、断裂或消失；后续消融因此分别考察细节融合和上下文建模。

### 3.4 Controlled ablation modules

#### 3.4.1 SegFormer-B0 + UAFM_SpAtten

UAFM_SpAtten 插入 `linear_fuse` 之后、`Dropout2D` 和分类卷积之前。其输入为融合特征 \(C\in\mathbb{R}^{B\times256\times100\times100}\) 与第一阶段细节特征 \(C_1\in\mathbb{R}^{B\times256\times100\times100}\)，输出尺寸仍为 \(B\times256\times100\times100\)：

\[
F_{\mathrm{UAFM}}=\operatorname{UAFM\_SpAtten}(C_1,C).
\]

该模块的研究假设是通过空间注意力和高低层特征融合改善细裂缝、端点和局部断裂。基线约 3.715M 参数、4.135G FLOPs；加入 UAFM 后约 4.897M 参数、15.943G FLOPs，验证 batch 耗时由约 0.0175 s 增至 0.0215 s。

#### 3.4.2 SegFormer-B0 + PPContextModule

PPContextModule 插入同一融合特征瓶颈，当前配置为 `PPContextModule(256,64,256,(1,3,6))`。它对融合特征建立多尺度上下文残差：

\[
F_{\mathrm{PPC}}=C+\operatorname{PPContextModule}(C),
\]

输出仍为 \(B\times256\times100\times100\)，再进入 dropout 和分类卷积。其研究假设是利用不同池化尺度改善长裂缝连续性、阴影区域和低对比度裂缝。加入后约 3.914M 参数、5.626G FLOPs，验证 batch 耗时约 0.0183 s。

两个模块只改变结构，不同时改变输入、损失、数据增强或训练预算。只有单模块通过验证门槛，才允许继续设计联合模型。

## 4 Experiments

### 4.1 Implementation details

实验使用 PaddleSeg-2.10.0 官方实现，在 RTX 5060、FP32 精度下运行。随机种子为 42，batch size 为 4，训练预算为 18,000 iterations。优化器为 AdamW，\(\beta_1=0.9\)、\(\beta_2=0.999\)、weight decay 为 0.01，初始学习率为 \(6\times10^{-5}\)。学习率采用 polynomial decay：

\[
\eta_t=\eta_0\left(1-\frac{t}{T}\right)^p,
\]

其中 \(\eta_0=6\times10^{-5}\)、\(T=18000\)、\(p=1\)，训练结束时 \(\eta_T=0\)。

训练损失为像素级交叉熵。设像素索引为 \(p\)，类别索引为 \(c\)，真实 one-hot 标签为 \(t_{p,c}\)，模型预测概率为 \(\hat p_{p,c}\)，则

\[
\mathcal{L}_{\mathrm{CE}}=-\frac{1}{N}
\sum_{p=1}^{N}\sum_{c=1}^{2}t_{p,c}\log(\hat p_{p,c}).
\]

训练阶段使用 ResizeStepScaling（0.75–1.25，步长 0.25）、400×400 RandomPaddingCrop、随机水平翻转、亮度/对比度/饱和度扰动和归一化；验证阶段只进行确定性的归一化。

普通模型和 EMA 模型都保存 checkpoint。模型选择仅使用验证集 mIoU：

\[
k^*=\arg\max_{k\in\{\mathrm{ordinary},\mathrm{EMA}\}}
\operatorname{mIoU}_{\mathrm{val}}(k).
\]

选定 checkpoint 后复制到冻结目录并计算 SHA-256，随后才允许访问测试集。测试指标不能反向参与模型选择。

### 4.2 Datasets

所有图像—标签对在训练前完成配对检查、尺寸检查、标签值检查和 SHA-256 审计。训练、验证和测试清单固定后不再修改。

#### 4.2.1 CFD dataset

CFD 章节暂作预留。正式实验需要补充原始图像数量、训练/验证/测试划分、分辨率、标签编码、源帧关系和去重规则。建议用一张示例图同时展示原图、标签、预测掩膜和边界叠加，并在附录给出清单及哈希摘要。

| Dataset | Images | Train | Validation | Test | Resolution | Label encoding | Source-frame audit |
|---|---:|---:|---:|---:|---|---|---|
| CFD | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |

#### 4.2.2 Crack500 dataset

本文沿用 Crack500 的固定 crop-split：训练 1,896 对、验证 348 对、测试 1,124 对。这里的数量是裁剪图像—标签对，不是独立原始采集 frame 数量。每个 crop 需要保留 `source_frame_id` 和 `crop_index`，并检查

\[
S_{train}\cap S_{val}=\varnothing,\qquad
S_{train}\cap S_{test}=\varnothing,\qquad
S_{val}\cap S_{test}=\varnothing.
\]

在 source-frame-disjoint 审计完成以前，本章把结果称为 **official crop-split results**。如果相邻 crop 来自同一个原始 frame，却被分到不同子集，模型可能通过相似纹理获得偏高结果，因此不能把当前 crop-split 数字直接称为 source-frame-disjoint 结果。

Crack500 覆盖细裂缝、交叉裂缝、宽裂缝、阴影、低对比度和复杂路面纹理。所有模型使用相同清单、相同输入尺寸、相同训练预算和相同验证选模规则。

#### 4.2.3 CrackSC dataset

CrackSC 为课题组自建数据集。本版只记录数据审计，不填最终 test 性能。原始训练侧 1,418 张图中发现 18 个精确或 D4 变换重复源图，去重后形成 1,400 张训练图、177 张验证图和封存的 177 张测试图。模型筛选阶段不访问自建 test。

| Dataset | Original train | Deduplicated train | Validation | Sealed test | Duplicate sources removed | Test used in selection |
|---|---:|---:|---:|---:|---:|---|
| CrackSC | 1,418 | 1,400 | 177 | 177 | 18 | No |

正式论文还需补充采集地点、路面类型、相机参数、分辨率、裂缝宽度范围、标注协议、源帧互斥规则及与机器人坐标标定的关系。

### 4.3 Evaluation metrics

以裂缝为前景类别 1、背景为类别 0。类别 \(c\) 的 IoU 为

\[
\operatorname{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c},
\qquad
\operatorname{mIoU}=\frac{\operatorname{IoU}_{bg}+\operatorname{IoU}_{crack}}{2}.
\]

本文主表中的裂缝 Precision、Recall 和 F1 为全数据集聚合混淆矩阵的前景指标：

\[
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad
F1=\frac{2PR}{P+R}.
\]

边界 F1 用 3×3 腐蚀提取内边界：

\[
B_G=G-\operatorname{Erode}_{3\times3}(G),\qquad
B_P=P-\operatorname{Erode}_{3\times3}(P).
\]

预测边界和真实边界的欧氏距离不超过 2 像素时视为匹配，由聚合匹配计数得到 \(P_b\)、\(R_b\) 和

\[
F1_b=\frac{2P_bR_b}{P_b+R_b}.
\]

参考论文的类别平均指标应单独记为 \(mP=(P_{bg}+P_{crack})/2\)、\(mR=(R_{bg}+R_{crack})/2\) 和 \(mF1=(F1_{bg}+F1_{crack})/2\)。参考论文的类别平均口径、2 像素容差和本文的聚合前景口径不能混合比较。除总体指标外，本文记录空预测数、零重叠图像数和逐图最差案例。

受控消融的验证门槛为

\[
\Delta mIoU\ge0,\quad
\Delta IoU_{crack}\ge0.005,\quad
\Delta F1\ge0.003,
\]

\[
\Delta Recall\ge-0.005,\qquad \Delta F1_b\ge0.
\]

### 4.4 Results and discussion

#### 4.4.1 Results of CFD dataset

CFD 统一训练和测试尚未完成，本版不填性能数字。正式结果应沿用相同冻结 checkpoint、混淆矩阵和边界评价流程，并至少展示细裂缝、交叉裂缝和纹理干扰三类定性样本。

#### 4.4.2 Results of Crack500 dataset

##### Validation results and checkpoint selection

SegFormer-B0 在 348 张验证图上得到 mIoU **0.8280**、裂缝 IoU **0.6782**、Precision **0.8283**、Recall **0.7891**、F1 **0.8082** 和边界 F1 **0.5894**，空预测 6 张。普通 checkpoint 在 11,000 iteration 达到最高验证 mIoU 0.8280206，因此冻结普通模型；该结果为当前控制组。

UAFM 的验证结果为 mIoU **0.8263**、裂缝 IoU **0.6752**、Precision **0.8202**、Recall **0.7924**、F1 **0.8059**。相对当前 SegFormer-B0，mIoU 约下降 0.0017、裂缝 IoU 下降 0.0030、F1 下降 0.0023，Recall 增加约 0.0033，未达到门槛，且没有可纳入最终结论的独立边界 F1 证据。因此 UAFM 记为受控消融负结果，不进入 test 对比。

PPContextModule 的验证结果为 mIoU **0.8249**、裂缝 IoU **0.6728**、Precision **0.8109**、Recall **0.7979**、F1 **0.8044** 和边界 F1 **0.5776**，空预测 7 张。相对基线，mIoU、裂缝 IoU、F1 和边界 F1 分别下降约 0.0031、0.0054、0.0039 和 0.0118；Recall 增加约 0.0088，但不足以抵消主体区域和边界质量下降。因此不启动 UAFM+PPContext 联合模型。

PP-LiteSeg-STDC1 是公开结构参考，其验证结果为 mIoU 0.8205、裂缝 IoU 0.6641、Precision 0.8250、Recall 0.7730、F1 0.7982 和边界 F1 0.5643；它不是本文的结构消融。

| Model | mIoU | Crack IoU | Precision | Recall | F1 | Boundary F1 | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| SegFormer-B0 | 0.8280 | 0.6782 | 0.8283 | 0.7891 | 0.8082 | 0.5894 | Control |
| SegFormer-B0 + UAFM | 0.8263 | 0.6752 | 0.8202 | 0.7924 | 0.8059 | 未纳入 | Negative ablation |
| SegFormer-B0 + PPContext | 0.8249 | 0.6728 | 0.8109 | 0.7979 | 0.8044 | 0.5776 | Negative ablation |
| PP-LiteSeg-STDC1 | 0.8205 | 0.6641 | 0.8250 | 0.7730 | 0.7982 | 0.5643 | Public reference |

##### Frozen test results

冻结后的 SegFormer-B0 在 Crack500 原始测试集上得到 mIoU **0.7695**、裂缝 IoU **0.5707**、Precision **0.7289**、Recall **0.7245**、F1 **0.7267**；空预测 8 张，零重叠 11 张。OCRNet-HRNet-W18 得到 mIoU **0.7715**、裂缝 IoU **0.5757**、Precision **0.7038**、Recall **0.7598**、F1 **0.7307**；空预测 13 张，零重叠 18 张。

由于 UAFM 与 PPContext 未通过验证门槛，本研究没有使用它们的 test 指标反向选择模型。当前统一协议下 OCRNet 略高于 SegFormer-B0，因而不能声称 SegFormer 已达到领域 SOTA；更准确的表述是“在当前 Crack500 crop-split 和已复现模型范围内的对比结果”。

| Model | mIoU | Crack IoU | Precision | Recall | F1 | Empty predictions | Zero-overlap images |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | 0.7695 | 0.5707 | 0.7289 | 0.7245 | 0.7267 | 8 | 11 |
| OCRNet-HRNet-W18 | 0.7715 | 0.5757 | 0.7038 | 0.7598 | 0.7307 | 13 | 18 |

#### 4.4.3 Results of CrackSC dataset

CrackSC 正式 test 结果暂不填入。本阶段仅完成去重和训练/验证审计。后续应先用验证集冻结最终模型，再访问封存测试集；同时报告分割指标、边界 F1、空预测、零重叠、骨架覆盖率、连通分量数和像素到物理坐标的转换误差。

#### 4.4.4 Computational efficiency

| Model | Parameters | FLOPs | Validation batch time |
|---|---:|---:|---:|
| SegFormer-B0 | 3.715M | 4.135G | 0.0175 s |
| SegFormer-B0 + UAFM | 4.897M | 15.943G | 0.0215 s |
| SegFormer-B0 + PPContext | 3.914M | 5.626G | 0.0183 s |
| OCRNet-HRNet-W18 | 12.110M | 32.396G | 0.0619 s |
| PP-LiteSeg-STDC1 | 8.039M | 3.568G | 0.0184 s |

UAFM 带来明显的计算量和参数增加，但没有带来符合门槛的验证收益；PPContext 增量较小，但同样没有改善总体和边界指标。论文最终还应补充机器人端相机分辨率下的单帧延迟、显存峰值、连续帧 FPS 和裂缝轨迹生成时间，不能把 RTX 5060 的 batch 时间直接当作嵌入式实时性能。

#### 4.4.5 Discussion

结果显示，SegFormer-B0 能够提供稳定的轻量分割基线，但其细裂缝表现受输入裁剪、下采样和解码恢复影响。UAFM 的 Recall 略增而裂缝 IoU、F1 和 mIoU 下降，说明增加局部融合并没有稳定恢复裂缝主体；PPContext 的 Recall 提高但 Precision、裂缝 IoU、F1 和边界 F1 下降，说明上下文响应扩大了候选区域，同时增加了阴影和纹理误检。两组结果都应作为负结果报告，不能改写为性能提升。

逐图错误分析应至少覆盖：

1. **细裂缝和端点**：检查 400×400 裁剪、缩放、下采样和最终上采样是否造成消失、变细或断裂；
2. **低对比度和阴影**：区分真正漏检与阴影误检；
3. **纹理密集路面**：统计沥青纹理被预测为裂缝的 FP；
4. **宽裂缝**：同时检查主体 IoU、边界 F1 和宽度恢复；
5. **逐图统计**：记录 IoU 变化、边界 F1 变化、FP/FN 像素数、空预测和零重叠，而不是只比较总体平均值。

当前最稳妥的论文主线是“轻量 SegFormer 基线—受控模块消融—统一协议公开模型比较—机器人闭环验证”。在 source-frame-disjoint 审计和自建数据最终测试完成前，不把 crop-split 结果写成无条件的 SOTA，也不使用未通过验证门槛的模块进行联合堆叠。

## 5 Figures and tables required for the paper

### 5.1 Figures

1. **Fig. 1 System overview**：封闭场地建图、覆盖路径、裂缝检测、停车、局部重复观测、像素—地图坐标转换、三轴/云台修复、修复后复检和回到覆盖路径。
2. **Fig. 2 SegFormer-B0 architecture**：MiT-B0 四阶段、通道数、分辨率、MLP 投影、上采样、拼接、融合卷积、dropout 和分类头。
3. **Fig. 3 Dataset and label examples**：原图、人工标签、裂缝二值图和边界叠加；CrackSC 在正式开放后补入。
4. **Fig. 4 Checkpoint selection curves**：普通模型与 EMA 模型验证 mIoU 随 iteration 变化，并标记冻结点。
5. **Fig. 5 Qualitative ablation comparison**：细裂缝、低对比度、阴影、纹理密集和宽裂缝五类样本，比较基线、UAFM、PPContext 与标签。
6. **Fig. 6 Error attribution overlays**：展示 TP、FP、FN、边界偏移，以及“边界提升但 IoU 下降”“边界与 IoU 同时提升”“完全变差/零重叠”三类图像。
7. **Fig. 7 Computational trade-off**：参数量、FLOPs、延迟和 F1 的散点图。
8. **Fig. 8 Robot validation**：巡检路径、停车局部观测、裂缝修复轨迹、修复后复检和坐标标定结果。

### 5.2 Tables

1. MiT-B0 配置和 400×400 实际张量尺寸；
2. CFD、Crack500、CrackSC 数据清单和 source-frame 审计；
3. 训练参数、checkpoint 选择规则和数据增强；
4. Crack500 验证消融结果；
5. Crack500 冻结 test 对比；
6. 参数量、FLOPs、延迟和显存；
7. 逐图错误分组统计；
8. checkpoint、数据清单和结果文件 SHA-256 附录。

## 6 Required follow-up before final submission

1. 为 Crack500 建立并发布 `source_frame_id`—`crop_index` 映射，完成 source-frame-disjoint 复核；
2. 补齐 UAFM 的同口径边界 F1 和逐图误差表；
3. 完成 CFD 和 CrackSC 的最终数据审计，封存 test 后再做一次性评估；
4. 用同一设备测量实际图像尺寸下的单帧延迟、FPS 和显存；
5. 完成相机—底盘—三轴坐标标定、覆盖路径、裂缝轨迹生成和修复后复检；
6. 在所有协议一致、结果确实超过对照模型之前，使用“当前对比范围内最优”或“统一协议下的最佳结果”，不使用无条件的“领域 SOTA”。


# VOCdevkit3000 域内消融报告：SegFormer-B0、Aligned 与 WTConv

日期：2026-11-01  
工程：`D:\论文复现\experiments\segformer_vocdevkit3000_aligned_wtconv_ablation_20261031`  
任务：在用户上传的 VOCdevkit3000 自建裂缝数据上，按 Crack500 阶段的训练协议，验证 SegFormer-B0、Aligned、WTConv 及二者联合配置。

## 1. 结论先行

四组训练均完成 18,000 iter，训练、官方验证和独立逐图验证均为真实 exit 0。由于该数据集的 `test.txt` 为空，本轮只使用 train/val，不能把结果称为 test 结果或跨域 test 结果。

在这一次 seed=42 的域内验证中，C0 原始 SegFormer-B0 的综合区域指标最高：mIoU=89.7776%、裂缝 IoU=80.7451%、F1=89.3469%。C1 Aligned 单独加入后边界 F1 略升，但 mIoU、裂缝 IoU、Recall、F1 和细线 Recall 下降；因此本数据集没有支持 Aligned 的独立正增益。C2 WTConv-only 减少参数和 FLOPs，但区域指标略低于 C0。C3 Aligned+WTConv 的 Recall、hard clDice 和最细裂缝 Recall 相对 C0 上升，同时 FLOPs 比 C0 低 8.8040%；代价是 Precision、mIoU、裂缝 IoU、F1 和边界 F1 下降。

因此，C3 可以作为“偏向召回和连通性的固定候选”保留，但本轮不能证明它全面优于 C0，也不能声称两个模块在自建域内形成了全面正互补。若论文主目标是综合分割精度，当前域内验证应保留 C0 作为精度对照；若目标更重视漏检和细裂缝连续性，C3 有可解释的补偿信号，仍需多种子复核。

## 2. 数据集说明与数据处理

原始数据目录为：`C:\Users\user\Desktop\VOCdevkit3000\VOCdevkit\VOC2007`。它是 VOC 风格的语义分割数据集，包含：

- `JPEGImages/`：2,932 张 RGB 图像；
- `SegmentationClass/`：2,932 张灰度标签；
- `ImageSets/Segmentation/train.txt`：2,345 张；
- `ImageSets/Segmentation/val.txt`：587 张；
- `ImageSets/Segmentation/test.txt`：空文件。

全部图像和标签为 448×448。原始标签像素值为 `{0,255}`，其中 0 是背景、255 是裂缝。PaddleSeg 的默认 `ignore_index=255` 会把 255 当作忽略标签，因此本轮没有直接把原始标签送入训练，而是在只读派生 staging 目录中将 255 映射为类别 1，0 保持为类别 0。原始 Desktop 数据未修改。

派生数据目录为：`C:\Users\user\AppData\Local\Temp\voc3000_stage_20261031\VOCdevkit\VOC2007`。训练前景像素比例为 5.1160%，验证前景像素比例为 5.1866%。本轮训练清单为工程内显式相对路径清单，train=2,345、val=587；清单 SHA256 记录在 `provenance/voc_ablation_manifest.json`。

## 3. 训练与评价协议

为了使自建域结果能够和 Crack500 结果对照，四组均使用相同的主要协议：RTX5060、Paddle 3.4、FP32、seed=42、batch size=4、AdamW、初始学习率 6e-5、PolynomialDecay `end_lr=0`、CrossEntropyLoss、EMA、18,000 iter、workers=0。增强为 ResizeStepScaling 0.75–1.25、RandomPaddingCrop 400×400、RandomHorizontalFlip、RandomDistort 和 Normalize。

普通 checkpoint 与 EMA checkpoint 都由验证集 mIoU 精确比较后选最高者。冻结后使用官方 val 和独立逐图评价脚本复核，共 587 张图；指标包括混淆矩阵、Precision、Recall、F1、裂缝 IoU、mIoU、1 px/2 px 边界 F1、hard clDice，以及 GT 半径 `r≤1` 的细线 Recall。所有组的 `test_used=false`。

当前工程保留了第一次 C0 smoke 因 Windows 中文路径导致 OpenCV 读取失败的 stdout、stderr、exit 记录；改用 ASCII 临时 staging 路径后重新 smoke 和正式训练成功。该故障没有被拼接进正式训练日志，也没有覆盖失败证据。

## 4. 四组结构

| 组别 | 结构 | 参数量 (M) | FLOPs (G) |
|---|---|---:|---:|
| C0 | SegFormer-B0 | 3.715170 | 4.135183 |
| C1 | SegFormer-B0 + Aligned | 3.785314 | 4.514480 |
| C2 | SegFormer-B0 + WTConv-only | 3.653346 | 3.391823 |
| C3 | SegFormer-B0 + Aligned + WTConv | 3.723490 | 3.771120 |

其中 WTConv 是当前工程中的低频 `AvgPool2D`、高频 depthwise 5×5、pointwise/residual 实现，报告将其称为本工程 WTConv candidate，不把它描述为原始小波论文实现。C2 的 `alignment=none`，C3 的 `alignment=full`；每个候选只在统一融合接口中出现一次。

## 5. Table 3 风格域内消融结果

百分数均来自 587 张 val 的冻结 checkpoint；参数和 FLOPs 不是由精度反推，而是由对应模型实际构造统计得到。

| Variant | Precision (%) | Recall (%) | F1-score (%) | IoU (%) | mIoU (%) | BF1@1 (%) | BF1@2 (%) | hard clDice (%) | r≤1 细线 Recall (%) | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | 87.1711 | **91.6342** | **89.3469** | **80.7451** | **89.7776** | 53.8799 | 69.4743 | 91.0367 | 85.6530 | 3.715170 | 4.135183 |
| SegFormer-B0 + Aligned | 87.5129 | 90.8692 | 89.1595 | 80.4394 | 89.6183 | **54.1530** | **69.6768** | 90.5840 | 84.4688 | 3.785314 | 4.514480 |
| SegFormer-B0 + WTConv-only | **87.5961** | 90.9355 | 89.2346 | 80.5618 | 89.6837 | 53.9309 | 69.5061 | 90.9089 | 84.8429 | **3.653346** | **3.391823** |
| SegFormer-B0 + Aligned + WTConv | 86.7247 | **91.9242** | 89.2488 | 80.5849 | 89.6894 | 51.9231 | 67.3322 | **91.1510** | **86.0960** | 3.723490 | 3.771120 |

模型选择均为普通 checkpoint：C0@18,000、C1@15,000、C2@15,000、C3@13,000。普通/EMA 两类候选、选模迭代、checkpoint SHA 和配置 SHA 见 `results/comparison.json` 与 `provenance/voc_ablation_manifest.json`。

## 6. 相对 C0 的模块贡献

下表的单位是百分点；参数/FLOPs 是相对变化百分比。

| Variant | ΔPrecision | ΔRecall | ΔF1 | Δ裂缝IoU | ΔmIoU | ΔBF1@1 | ΔBF1@2 | Δhard clDice | Δ细线 Recall | ΔParams | ΔFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 Aligned | +0.3418 | −0.7651 | −0.1875 | −0.3057 | −0.1593 | +0.2731 | +0.2024 | −0.4527 | −1.1842 | +1.8880% | +9.1725% |
| C2 WTConv-only | +0.4250 | −0.6987 | −0.1124 | −0.1834 | −0.0939 | +0.0510 | +0.0318 | −0.1278 | −0.8100 | **−1.6641%** | **−17.9765%** |
| C3 Aligned+WTConv | −0.4464 | +0.2900 | −0.0982 | −0.1602 | −0.0882 | −1.9568 | −2.1422 | +0.1143 | +0.4431 | +0.2239% | **−8.8040%** |

### 6.1 Aligned 的贡献

Aligned 在本数据集上把 1 px 边界 F1 提高 0.2731 个百分点、2 px 边界 F1 提高 0.2024 个百分点，同时 Precision 提高 0.3418 个百分点。这说明它有改善跨层空间对应和局部边缘定位的信号。

但 Aligned 使 Recall 下降 0.7651 个百分点，最细裂缝 Recall 下降 1.1842 个百分点，裂缝 IoU 下降 0.3057 个百分点，mIoU 下降 0.1593 个百分点。它还增加 1.8880% 参数和 9.1725% FLOPs。因此，本轮不能把 Aligned 写成“独立全面有效模块”；更准确的表述是：它偏向精确定位和 Precision，代价是漏检和容量/计算增加。

### 6.2 WTConv 的贡献

WTConv-only 将参数从 3.715170M 降至 3.653346M，减少 1.6641%；FLOPs 从 4.135183G 降至 3.391823G，减少 17.9765%，是本轮最明确的效率收益。它还把 Precision 提高 0.4250 个百分点，边界 F1 基本保持。

但 C2 的 mIoU、裂缝 IoU、Recall、F1、hard clDice 和细线 Recall 都略低于 C0。也就是说，WTConv 在这组域内实验中更像一个计算效率替换，而不是已经证明的精度增强模块。

### 6.3 Aligned+WTConv 联合贡献

C3 相对 C0 的主要正向信号是 Recall +0.2900 个百分点、hard clDice +0.1143 个百分点、最细裂缝 Recall +0.4431 个百分点，同时 FLOPs 减少 8.8040%，参数只增加 0.2239%。相对 C1，C3 还恢复了 Recall、hard clDice 和细线 Recall；相对 C2，C3 也有约 1 个百分点 Recall 和 0.24 个百分点 hard clDice 的提升。

联合配置的代价是 Precision −0.4464 个百分点、mIoU −0.0882 个百分点、裂缝 IoU −0.1602 个百分点、F1 −0.0982 个百分点，2 px 边界 F1 下降 2.1422 个百分点。由此可见，联合结构在本域中表现为“区域召回/连通性补偿 + 边界质量退化”，而不是所有指标同步提升。它可以作为偏召回部署方向的候选，但不应写成全面优于基线或已达到 SOTA。

## 7. 与用户给出的论文式消融表的对应关系

用户示例中的四行结构是“基线、模块 A、模块 D、A+D”。本轮按照同样的逻辑组织为：C0=SegFormer-B0、C1=Aligned、C2=WTConv-only、C3=Aligned+WTConv。这样可以清楚回答三个问题：

1. Aligned 单独是否有效：本轮只在边界 F1 和 Precision 上有小幅正向信号，综合区域指标未通过；
2. WTConv 单独是否有效：本轮明确减少计算和参数，但综合精度没有超过 C0；
3. 联合是否互补：联合恢复了 Recall、hard clDice 和细线 Recall，但牺牲了边界 F1 和 Precision，属于有方向性的补偿，不是无条件正交互。

## 8. 证据与复现索引

- 总状态：`results/workflow_state.json`，`stage=complete`、`successful_jobs=4`、`test_used=false`；
- 四组总表：`results/voc_ablation_table.csv`；
- 相对 C0 差值：`results/voc_ablation_deltas.csv`；
- 四组逐图指标：`results/C0_B0_custom_val_per_image.csv`、`C1_Aligned_custom_val_per_image.csv`、`C2_WTConv_custom_val_per_image.csv`、`C3_Aligned_WTConv_custom_val_per_image.csv`；
- 独立精确评价 JSON：各组 `results/*_custom_val.json`；
- 普通/EMA 选模和 SHA：`results/comparison.json`；
- 数据、协议、配置与 checkpoint 索引：`provenance/voc_ablation_manifest.json`；
- 首次中文路径 smoke 失败证据：`provenance/failed_attempt_20261031_C0_B0_smoke*`；
- W12 旧失败目录清理清单：`D:\论文复现\experiments\segformer_simplify_wave12_20261026\provenance\deleted_artifacts_20261031.json`。

## 9. 后续建议

如果论文主线强调综合精度，应把 C0 作为自建域的精度基准，把 C3 作为召回/连通性方向的补充实验，并在至少 seed=43、44 上复核 C3 的正向 Recall、hard clDice 和细线 Recall 是否稳定。如果部署主线强调有限推理预算，C2 是最直接的轻量化对照，C3 则提供更好的召回信号但需要重新检查边界退化、阈值和误检来源。

在完成多种子之前，不应把本轮单种子差异写成普遍模块规律，也不应访问不存在的 test 或把 val 结果称为 SOTA。报告之外的后续训练应继续复用本轮数据清单、标签转换规则、源代码快照和评价脚本。

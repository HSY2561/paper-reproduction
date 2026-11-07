# Crack500 Aligned 与 WTConv 四组消融报告（Wave17）

实验日期：2026-10-31。所有结果来自 RTX 5060、PaddlePaddle 3.4.0、FP32、seed=42、batch=4、AdamW、初始学习率 6e-5、PolynomialDecay end_lr=0、18,000 iter、相同 400x400 训练增强和 Crack500 原始 train/val/test 清单。训练和选模只使用 train/val；所有四组冻结后才访问 1124 张 test 图。

C2 是 WTConv-only：`alignment: none`；C3 才是 `alignment: full` 与 WTConv 联合。WTConv 采用当前工程中已有的低频 AvgPool2D + 高分支 depthwise 5x5 + pointwise/residual 实现，报告名称使用 WTConv-only candidate，避免把它表述为论文原始小波分解实现。

## 验证集选模

| 模型 | 选中 | mIoU | 裂缝 IoU | Precision | Recall | F1 | 边界 F1@2px | Params | FLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 (C0) | ordinary | 82.6909 | 67.6106 | 82.7207 | 78.7296 | 80.6758 | 45.1936 | 3.715M | 4.135G |
| SegFormer-B0 + Aligned (C1) | ordinary | 83.2467 | 68.7224 | 80.2780 | 82.6816 | 81.4621 | 45.1555 | 3.785M | 4.514G |
| SegFormer-B0 + WTConv-only (C2) | ordinary | 82.9554 | 68.1473 | 81.2478 | 80.8664 | 81.0567 | 44.5430 | 3.653M | 3.392G |
| SegFormer-B0 + Aligned + WTConv (C3) | ordinary | 82.8976 | 68.0902 | 79.4177 | 82.6805 | 81.0163 | 43.4725 | 3.723M | 3.771G |

## 冻结 checkpoint 后的 Crack500 test

| 模型 | Precision | Recall | F1-score | IoU | mIoU | 边界 F1@2px | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 (C0) | 72.8613 | 72.4660 | 72.6631 | 57.0637 | 76.9434 | 37.9749 | 3.715M |
| SegFormer-B0 + Aligned (C1) | 70.9480 | 77.1986 | 73.9414 | 58.6564 | 77.7387 | 37.9813 | 3.785M |
| SegFormer-B0 + WTConv-only (C2) | 72.6574 | 73.6516 | 73.1511 | 57.6679 | 77.2579 | 37.9876 | 3.653M |
| SegFormer-B0 + Aligned + WTConv (C3) | 70.2578 | 78.4197 | 74.1147 | 58.8748 | 77.8361 | 36.8405 | 3.723M |

## 相对 C0 的 test 变化（百分点）

| 模型 | ΔPrecision | ΔRecall | ΔF1 | Δ裂缝 IoU | ΔmIoU | Δ边界 F1@2px | ΔParams |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 (C0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0M |
| SegFormer-B0 + Aligned (C1) | -1.9134 | +4.7326 | +1.2783 | +1.5927 | +0.7952 | +0.0064 | 0.07M |
| SegFormer-B0 + WTConv-only (C2) | -0.2039 | +1.1856 | +0.4880 | +0.6042 | +0.3144 | +0.0127 | -0.062M |
| SegFormer-B0 + Aligned + WTConv (C3) | -2.6036 | +5.9537 | +1.4516 | +1.8111 | +0.8926 | -1.1344 | 0.008M |

## 结论

Aligned 单模块在本次统一 test 协议下显著提高 Recall、F1 和裂缝 IoU，Precision 下降约 1.9133 个百分点，边界 F1 基本不变。WTConv-only 参数量最低、FLOPs 最低，并在 test 上小幅提升 mIoU、裂缝 IoU 和 F1，说明效率收益明确，但精度增益小于 Aligned。联合模型的 mIoU、裂缝 IoU、F1 和 Recall 均为四组最高，但 Precision 下降最大，边界 F1 低于 C0，显示出假阳性增加和边界质量退化的代价。

这些结果支持把 Aligned 作为主要精度候选，把 WTConv-only 作为效率候选；联合模型需要在后续自建数据和机器人闭环验证前保留其 Precision、边界质量与实时延迟风险。当前结果是本实验协议内的比较结论，不构成跨论文意义上的 SOTA 声明。

冻结 checkpoint 的 SHA256、普通/EMA 验证明细、逐图边界数据和完整 JSON 均保存在 `results`、`checkpoints`、`provenance` 目录。

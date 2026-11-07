# AW 结构与损失修正实验计划

日期：2026-11-06  
数据：VOCdevkit3000，完整 train=2345、val=587  ̄

## 目标

验证完整 Aligned 融合后接 WTConv 的结构，在不修改验证集的前提下，是否能通过单变量损失设计改善区域精度、边界质量和细裂缝召回。

## 固定协议

seed=42，FP32，batch=4，AdamW，6e-5，PolynomialDecay end_lr=0，CrossEntropy/候选损失，EMA，18,000 iter，workers=0，原始增强、训练清单和验证清单固定。ordinary/EMA 仅按完整 587 张验证集 exact mIoU 选模。

## 结构母体

使用修正后的 AW 母体。保留原生 Aligned 1024→256 融合，再接 256 通道零初始化残差 WTConv。原 C3 的 1024→128→256 瓶颈结构不再用于新损失比较。

## 候选

| Run | 单变量改变 | 目的 | 状态 |
|---|---|---|---|
| C6 | CE + 0.25 Dice，MixedLoss 对每个损失提供独立 labels.clone() | 提高区域重叠、Recall 和细线召回 | 完成 |
| C7 | 边界加权 CE，标签局部转移像素权重 1+edge | 改善边界和局部定位 | 完成 |

## 通过标准

完整验证集上同时比较 mIoU、裂缝 IoU、F1、Precision、Recall、BF1@1/2、hard clDice 和 r≤1 细线 Recall。任何候选都不能通过删除样本、调整清单或反向使用 test 选择。

## 证据输出

结果表为 `results/aw_loss_comparison.csv`。C6/C7 的训练、官方验证和独立边界验证 receipt 均在 `results/` 中，原始数据和验证清单保持不变。

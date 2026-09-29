from pathlib import Path
abl = r'''# Crack500 BiSeNetV2 模型消融与最终组合报告

更新时间：2026-09-25  
服务器工程：`/hy-tmp/crack_project`  
数据集：`/hy-tmp/crack_project/data/Crack500_data_u8`  
统一测试集：1124 张原始 Crack500 test 图像  
主基线：PaddleSeg 官方 BiSeNetV2；CE 是默认训练损失，不写入模型名称

## 最终结论

本轮 A0～A7 已按“验证集筛选、冻结后测试”的流程完成。通过真实服务器日志，最优组合仍是：

**BiSeNetV2 + OHEM Cross Entropy + DiceLoss + EMA**。

它相对 BiSeNetV2 默认 CE 的测试 mIoU 提升 0.0088，但测试 mIoU 仍为 0.7694，低于 OCRNet-HRNet-W18 的 0.7760；裂缝 IoU 0.5705，低于 0.5819；前景 F1 0.7264，低于 0.7356。因此当前结果不能称为严格 SOTA。A2～A7 中没有一个模块满足进入最终组合的筛选条件，最终组合只保留 A1 的损失与 EMA 配置。

## A0～A7 真实消融记录

| 编号 | 实际配置 | 训练/选模 | 最佳验证 mIoU | 结果 | 服务器证据 |
|---|---|---:|---:|---|---|
| A0 | BiSeNetV2 + 默认 CE | 18k iter | 0.8057 | 主基线 | `logs/bisenetv2_train.log` |
| A1 | OHEM + Dice + EMA | 18k iter | **0.8181** | 当前最佳，进入最终测试 | `logs/bisenetv2_ohem_dice_ema.log` |
| A2 | A1 + PaddleSeg 官方 PPModule | 6k iter | 0.7999（EMA） | 未通过，未冻结到 test | `logs/bisenetv2_a2_ppm_6k_20260925.log` |
| A3 | A1 + PaddleSeg 官方 ASPPModule | 6k iter | 0.7859（EMA） | 未通过，未冻结到 test | `logs/bisenetv2_a3_aspp_6k_20260925.log` |
| A4 | 边界监督候选：RelaxBoundary/OhemEdgeAttention | 既有筛选 | 0.7642；OhemEdge 不兼容 | 未通过 | `logs/abl3_ohem_relax.log`、`logs/abl2_ohem_edge.log` |
| A5 | 连通性候选：SemanticConnectivityLoss | 既有筛选 | 0.3053 | 训练失效，未进入组合 | `logs/abl2_driver.log`、`configs/bisenetv2_ablation_semconn.yml` |
| A6 | A1 + PaddleSeg 官方 SE 通道注意力 | 3k iter 筛选 | 0.7756（非 EMA） | 未通过 | `logs/bisenetv2_a6_se_3k_20260925.log` |
| A7 | A1 + PPModule + SE | 3k iter 筛选 | 0.7825（非 EMA） | 未通过 | `logs/bisenetv2_a7_ppm_se_3k_20260925.log` |

说明：A2、A3、A6、A7 是结构模块筛选实验，使用同一原始 Crack500 训练/验证划分和随机种子 42；筛选阶段不使用 test 集。PPModule、ASPPModule 和 SE 均来自服务器 PaddleSeg 2.10.0 官方实现，未自定义新的算法模块。

## 冻结后的统一 test 结果

| 模型 | mIoU | 裂缝 IoU | 裂缝 Precision | 裂缝 Recall | 裂缝前景 F1 |
|---|---:|---:|---:|---:|---:|
| BiSeNetV2（默认 CE） | 0.7606 | 0.5545 | 0.7146 | 0.7122 | 0.7134 |
| **BiSeNetV2 + OHEM + Dice + EMA（最终组合）** | **0.7694** | **0.5705** | **0.7296** | **0.7233** | **0.7264** |
| PP-LiteSeg-STDC1 | 0.7661 | 0.5634 | 0.7527 | 0.6915 | 0.7219 |
| OCRNet-HRNet-W18 | **0.7760** | **0.5819** | **0.7601** | 0.7128 | **0.7356** |
| DeepLabV3P-ResNet50 | 0.7257 | 0.4955 | 0.5995 | **0.7406** | 0.6619 |
| SegFormer-B0 | 0.7570 | 0.5466 | 0.7388 | 0.6776 | 0.7069 |
| PIDNet-S | 0.5005 | 0.0547 | 0.8036 | 0.0554 | 0.1036 |
| HrSegNet-B32 | 0.7473 | — | — | — | — |

A1 的 test 结果来自 `logs/recheck_bisenetv2_ohem_dice_ema_test_20260924.log`；基线和开源模型结果来自同一批 recheck 日志。裂缝前景 F1 由裂缝类别 Precision 与 Recall 计算，不能直接用 PaddleSeg 的两类宏平均 Dice 替代。

## 为什么结构模块没有保留

- PPM 在验证集最佳 EMA mIoU 为 0.7999，低于 A1 的 0.8181，且参数量约 2.84M，已经超过轻量增量门槛。
- ASPP 在 6k 筛选中最佳 EMA mIoU 为 0.7859，低于 A1；参数量约 2.50M，未抵消精度损失。
- SE 和 PPM+SE 的最佳非 EMA mIoU 分别为 0.7756 和 0.7825，均没有达到 A1。
- RelaxBoundaryLoss、Weighted/OHEM、RMI、DualTask、SemanticConnectivityLoss、噪声/模糊增强和 Lovasz/Focal 组合在既有筛选中也没有超过 A1；OhemEdgeAttentionLoss 要求 `edge_logit` 输出，与官方 BiSeNetV2 输出结构不兼容。

## 最终组合方案

论文和工程中固定使用：

1. 网络：PaddleSeg 官方 BiSeNetV2；
2. 主损失：OHEM Cross Entropy 与 DiceLoss，比例 0.8/0.2；
3. 训练稳定化：EMA；
4. 训练协议：AdamW、PolynomialDecay、原始 Crack500 train/val/test 划分；
5. 选模指标：验证集裂缝 IoU 优先，mIoU、Recall、前景 F1 作为并列参考；
6. 测试：只对冻结后的 A1 checkpoint 做一次统一 test 评估。

当前不把 PPM、ASPP、SE、SCSegamba 作为最终组合。SCSegamba 已按项目决定归档，不进入主比较。

## 后续任务

1. 保留 A1 作为当前视觉模型，补齐参数量、FLOPs、GPU/Jetson Nano 延迟和显存记录。
2. 若论文必须追求超过 OCRNet 的数值目标，需要另行批准更大范围的结构改动，例如替换为更强的官方编码器或完整的多尺度解码器；这将不再是“只在 BiSeNetV2 上加轻量模块”的严格消融。
3. 在最终模型未改变前，开始自建数据集迁移训练和相机—底盘—三轴机构坐标标定。
4. 机器人实验按粗定位循迹、停靠、三自由度精定位、裂缝宽度测量和修复完整度依次进行。
5. 论文中应写“当前统一协议下具有竞争力的轻量模型”，不能写“达到 SOTA”。
'''
Path('模型消融文档.md').write_text(abl, encoding='utf-8')
summary = r'''# Crack500 实验结果汇总与后续计划

更新时间：2026-09-25

## 当前结论

A0～A7 消融已完成。最终保留的 BiSeNetV2 方案为 **OHEM + Dice + EMA**。结构模块 PPM、ASPP、SE 以及既有边界/连通性候选均未在验证集超过该方案，因此没有理由把它们写入最终组合。最终 test mIoU 为 0.7694，仍低于 OCRNet-HRNet-W18 的 0.7760，当前不能声称 SOTA。

## 消融表（验证集筛选）

| 方案 | 最佳验证 mIoU | 状态 |
|---|---:|---|
| BiSeNetV2 + CE | 0.8057 | A0 主基线 |
| OHEM + Dice + EMA | **0.8181** | A1 最终组合 |
| A1 + PPModule | 0.7999（EMA） | A2 淘汰 |
| A1 + ASPPModule | 0.7859（EMA） | A3 淘汰 |
| 边界监督候选 | 0.7642；部分接口不兼容 | A4 淘汰 |
| SemanticConnectivityLoss | 0.3053 | A5 训练失效 |
| A1 + SE | 0.7756 | A6 淘汰 |
| A1 + PPModule + SE | 0.7825 | A7 淘汰 |

## 最终模型与开源模型（原始 test）

| 模型 | mIoU | 裂缝 IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| BiSeNetV2（CE） | 0.7606 | 0.5545 | 0.7146 | 0.7122 | 0.7134 |
| BiSeNetV2 + OHEM + Dice + EMA | **0.7694** | **0.5705** | **0.7296** | **0.7233** | **0.7264** |
| PP-LiteSeg-STDC1 | 0.7661 | 0.5634 | 0.7527 | 0.6915 | 0.7219 |
| OCRNet-HRNet-W18 | **0.7760** | **0.5819** | **0.7601** | 0.7128 | **0.7356** |
| DeepLabV3P-ResNet50 | 0.7257 | 0.4955 | 0.5995 | 0.7406 | 0.6619 |
| SegFormer-B0 | 0.7570 | 0.5466 | 0.7388 | 0.6776 | 0.7069 |
| PIDNet-S | 0.5005 | 0.0547 | 0.8036 | 0.0554 | 0.1036 |

## 后续执行顺序

1. 冻结 A1 checkpoint，记录参数量、FLOPs、显存和实际推理延迟。
2. 迁移到自建数据集，重新建立 train/val/test 划分；自建数据只用于后续应用实验，不回填 Crack500 的公开测试结果。
3. 完成相机内参、相机—底盘、相机—三轴机构的标定，建立像素到物理坐标转换。
4. 从裂缝分割结果提取中心线、方向和局部宽度，驱动四轮差速底盘完成粗跟踪。
5. 底盘停靠后，用三自由度机构完成细定位、喷头/修复工具轨迹规划和填充。
6. 依次测试循迹误差、停靠误差、定位误差、裂缝宽度测量误差、修复完整度和实时性。
7. 若后续仍要求数值超过 OCRNet，需要先单独批准更强结构替换实验；当前轻量模块筛选没有提供可靠提升。
'''
Path('实验结果汇总与后续计划.md').write_text(summary, encoding='utf-8')
print('updated docs')

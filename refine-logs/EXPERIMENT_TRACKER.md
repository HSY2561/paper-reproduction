# 实验执行跟踪表

更新时间：2026-09-25（A0～A7 服务器实验完成）
服务器工程：`/hy-tmp/crack_project`

## 本轮执行状态核验（服务器回执）

服务器最终回执：SCSegamba 原始固定划分和 source-frame-disjoint 运行均已完成，但现已归档，不再作为主比较。HrSegNet-B32 训练/test 已完成，mIoU 0.7473。当前主线是 BiSeNetV2 基线与成熟模块消融。

## 阶段状态

| 事项 | 状态 | 证据 |
|---|---|---|
| BiSeNetV2 工程基线 | 已确定 | PaddleSeg 官方配置，默认训练损失为 CE；服务器测试 mIoU 0.7606 |
| BiSeNetV2 消融 | A0～A7 已完成 | A1（OHEM+Dice+EMA）验证 mIoU 0.8181、测试 mIoU 0.7694；A2/A3/A6/A7 未超过 A1，A4/A5 已淘汰 |
| 公开模型统一测试 | 已完成一轮 | OCRNet 0.7760，PP-LiteSeg 0.7661，DeepLabV3P 0.7257，SegFormer 0.7570，PIDNet-S 0.5005 |
| SCSegamba | 已归档，不参与主比较 | 原始固定划分和 source-frame-disjoint 结果均已保存，仅作审计，不再作为后续对照或优化对象 |
| HrSegNet-B32 | 已完成 | 官方 best checkpoint；原始 test mIoU 0.7473，低于 OCRNet 0.7760 |
| Source-frame-disjoint | 已建立，暂不作为当前优化门槛 | 442 个源组、6,736 个裁块；当前优先沿用已完成公开模型对比的原始 Crack500 划分优化 BiSeNetV2 |
| MixerCSeg | 未完成，服务器环境待处理 | 服务器工程存在 `/hy-tmp/crack_project/src/MixerCSeg`；此前 selective-scan 扩展编译遇到 CUDA/PyTorch 兼容问题，尚未形成有效训练或测试结果 |
| SOTA 验收 | 未通过 | 最终候选 test mIoU=0.7694，低于 OCRNet 0.7760；当前不能声称 SOTA |
| 源帧互斥 split | 已建立并复核完成 | 源组跨集合交集为空；后续用于 BiSeNetV2、消融模型和开源对照统一重跑 |

## 已核验运行

| Run ID | 变体 | 数据 | 状态 | 结果 |
|---|---|---|---|---|
| B001 | BiSeNetV2（默认 CE） | Crack500 test 1124 | 完成 | mIoU 0.7606 |
| B002 | BiSeNetV2 + OHEM + Dice + EMA | Crack500 test 1124 | 完成 | mIoU 0.7694 |
| C001 | OCRNet-HRNet-W18 | Crack500 test 1124 | 完成 | mIoU 0.7760 |
| C002 | PP-LiteSeg-STDC1 | Crack500 test 1124 | 完成 | mIoU 0.7661 |
| C003 | DeepLabV3P-ResNet50 | Crack500 test 1124 | 完成 | mIoU 0.7257 |
| C004 | SegFormer-B0 | Crack500 test 1124 | 完成 | mIoU 0.7570 |
| C005 | PIDNet-S | Crack500 test 1124 | 完成但失败表现 | mIoU 0.5005，裂缝 Recall 0.0554 |
| S001 | SCSegamba 官方代码，错误标签编码 | train 1896 / val 348 | 作废 | 0/1 掩码被阈值 127 清成背景，loss≈0；不可排名 |
| S002 | SCSegamba 官方代码，0/255 专用标签副本 | train 1896 / val 348 / 原始 test 1124 | 完成 | 验证集冻结 checkpoint14；val mIoU 0.820511，原始 test mIoU 0.762768，裂缝 IoU 0.558078，P/R/F1=0.727022/0.706020/0.716367 |
| H001 | HrSegNet-B32 官方实现 | train/val/test 1896/348/1124 | 完成 | 官方 best checkpoint；test mIoU 0.7473（监督结果 JSON 的早期 -1 占位已更正） |
| SD001 | Source-frame-disjoint 划分 | manifest train/val/test 4714/674/1348；unique files 4706/674/1348 | 已建立 | 原 SCSegamba 复核已归档；该划分转用于 BiSeNetV2 基线、消融模型和开源对照统一重跑 |

中途独立 test 回执仅作趋势记录：checkpoint6/checkpoint_best mIoU 0.757861，checkpoint9 mIoU 0.758827。原始固定划分正式冻结结果为 checkpoint14：val mIoU 0.820511，原始 test mIoU 0.762768、裂缝 IoU 0.558078、P/R/F1=0.727022/0.706020/0.716367。source-frame-disjoint 正式冻结结果为 checkpoint20：val mIoU 0.7850416，独立 test mIoU 0.7957967、裂缝 IoU 0.6218454、P/R/F1=0.7327194/0.8042867/0.7668369。

## 已完成的 BiSeNetV2 优化运行

| Run ID | 变体 | 预算 | 最佳验证 mIoU | 状态 |
|---|---|---:|---:|---|
| A0 | BiSeNetV2 + CE | 18k iter | 0.8057 | 主基线，test 0.7606 |
| A1 | OHEM + Dice + EMA | 18k iter | **0.8181** | 最终保留，test 0.7694 |
| A2 | A1 + 官方 PPModule | 6k iter | 0.7999（EMA） | 淘汰 |
| A3 | A1 + 官方 ASPPModule | 6k iter | 0.7859（EMA） | 淘汰 |
| A4 | RelaxBoundary/OhemEdgeAttention | 既有筛选 | 0.7642；部分不兼容 | 淘汰 |
| A5 | SemanticConnectivityLoss | 既有筛选 | 0.3053 | 训练失效，淘汰 |
| A6 | A1 + 官方 SE | 3k iter | 0.7756 | 淘汰 |
| A7 | A1 + PPModule + SE | 3k iter | 0.7825 | 淘汰 |
| S1 | A1 + PPContextModule | 18k iter | 0.8187 | 已完成；test mIoU 0.7677、裂缝 IoU 0.5673，淘汰 |
| S2 | A1 + UAFM_SpAtten | 18k iter | 0.8158 | 已完成；test mIoU 0.7638、裂缝 IoU 0.5594，淘汰 |
| S3 | A1 + PPContextModule + UAFM_SpAtten | 18k iter | 0.8197 | 已完成；test mIoU 0.7669、裂缝 IoU 0.5651，淘汰 |

## 下一步

1. 固定 A1 作为当前视觉模型；S1-S3 已完成统一 test，均不进入最终组合。
2. 在自建数据集上建立独立 train/val/test 划分并开展迁移训练。
3. 完成相机、四轮底盘和三自由度机构的坐标标定及像素—物理坐标转换。
4. 开展粗定位循迹、停靠、三轴精定位、宽度测量和裂缝修复实验。
5. 若论文仍要求超过 OCRNet，需要另行批准更强结构替换；当前轻量模块筛选没有达到 SOTA。

## 证据路径

服务器日志：

- `/hy-tmp/crack_project/logs/recheck_bisenetv2_ce_test_20260924.log`
- `/hy-tmp/crack_project/logs/recheck_bisenetv2_ohem_dice_ema_test_20260924.log`
- `/hy-tmp/crack_project/logs/recheck_ocrnet_hrnetw18_test_20260924.log`
- `/hy-tmp/crack_project/logs/recheck_ppliteseg_stdc1_test_20260924.log`
- `/hy-tmp/crack_project/logs/recheck_deeplabv3p_r50_test_20260924.log`
- `/hy-tmp/crack_project/logs/recheck_segformer_b0_testonly_20260924.log`
- `/hy-tmp/crack_project/logs/recheck_pidnet_s_testonly_20260924.log`
- `/hy-tmp/crack_project/logs/scsegamba_valselect_20260924.log`（作废：标签编码错误）
- `/hy-tmp/crack_project/logs/scsegamba_valselect_u8_20260924.log`（正确编码 50 epoch 训练，已完成）
- `/hy-tmp/crack_project/logs/scsegamba_unified_final_20260924.json`（验证集冻结与原始 test 最终 JSON）
- `/hy-tmp/crack_project/logs/hrsegnet_b32_test_20260924.log`（HrSegNet 官方 test）
- `/hy-tmp/crack_project/logs/struct_s1_context_18k_train_20260925.log`（S1 训练与验证）
- `/hy-tmp/crack_project/logs/struct_s2_uafm_18k_train_20260925.log`（S2 训练与验证）
- `/hy-tmp/crack_project/logs/struct_s3_context_uafm_18k_train_20260925.log`（S3 训练与验证）
- `/hy-tmp/crack_project/logs/struct_s1_context_18k_test_result.json`（S1 test）
- `/hy-tmp/crack_project/logs/struct_s2_uafm_18k_test_result.json`（S2 test）
- `/hy-tmp/crack_project/logs/struct_s3_context_uafm_18k_test_result.json`（S3 test）
- `/hy-tmp/crack_project/logs/scsegamba_source_disjoint_fixed_train_20260925.log`（有效 50 epoch 训练）
- `/hy-tmp/crack_project/logs/scsegamba_source_disjoint_fixed_eval_20250925.log`（逐 checkpoint 验证评估）
- `/hy-tmp/crack_project/logs/scsegamba_source_disjoint_fixed_result_20260925.json`（最终冻结与独立 test 结果）


# -*- coding: utf-8 -*-
from pathlib import Path
p=Path('refine-logs/EXPERIMENT_TRACKER.md')
s=p.read_text(encoding='utf-8')
s=s.replace('更新时间：2026-09-25（服务器状态核验至 05:04 CST）','更新时间：2026-09-25（A0～A7 服务器实验完成）')
s=s.replace('| BiSeNetV2 消融 | 部分完成 | OHEM、Dice、Focal、Lovasz、EMA、增强及失败模块已有服务器日志；最佳候选测试 mIoU 0.7694 |','| BiSeNetV2 消融 | A0～A7 已完成 | A1（OHEM+Dice+EMA）验证 mIoU 0.8181、测试 mIoU 0.7694；A2/A3/A6/A7 未超过 A1，A4/A5 已淘汰 |')
s=s.replace('| SOTA 验收 | 优化进行中 | 当前最佳候选 BiSeNetV2 + OHEM + Dice + EMA 的原始 test mIoU=0.7694，目标是超过 OCRNet 参考值 0.7760 |','| SOTA 验收 | 未通过 | 最终候选 test mIoU=0.7694，低于 OCRNet 0.7760；当前不能声称 SOTA |')
start=s.find('## 待执行的 BiSeNetV2 优化运行')
end=s.find('## 下一步',start)
new='''## 已完成的 BiSeNetV2 优化运行\n\n| Run ID | 变体 | 预算 | 最佳验证 mIoU | 状态 |\n|---|---|---:|---:|---|\n| A0 | BiSeNetV2 + CE | 18k iter | 0.8057 | 主基线，test 0.7606 |\n| A1 | OHEM + Dice + EMA | 18k iter | **0.8181** | 最终保留，test 0.7694 |\n| A2 | A1 + 官方 PPModule | 6k iter | 0.7999（EMA） | 淘汰 |\n| A3 | A1 + 官方 ASPPModule | 6k iter | 0.7859（EMA） | 淘汰 |\n| A4 | RelaxBoundary/OhemEdgeAttention | 既有筛选 | 0.7642；部分不兼容 | 淘汰 |\n| A5 | SemanticConnectivityLoss | 既有筛选 | 0.3053 | 训练失效，淘汰 |\n| A6 | A1 + 官方 SE | 3k iter | 0.7756 | 淘汰 |\n| A7 | A1 + PPModule + SE | 3k iter | 0.7825 | 淘汰 |\n\n'''
s=s[:start]+new+s[end:]
s=s.replace('1. 固定 BiSeNetV2 为主基线，CE 作为默认训练配置，继续沿用已经完成公开模型对比的原始 Crack500 划分。\n2. 依次测试成熟损失、边界监督、上下文/注意力、多尺度融合和数据增强模块，针对 mIoU、裂缝 IoU、Recall 和 F1 的短板进行消融。\n3. 以 OCRNet 参考 mIoU 0.7760、裂缝 IoU 0.5819、F1 0.7356 为超越目标；每轮先用验证集筛选，达到目标后冻结模型并只做一次最终 test。\n4. 补充参数量、FPS/延迟和多随机种子结果，再迁移到自建数据集和机器人闭环实验。', '1. 冻结 A1 作为当前视觉模型，补充参数量、FLOPs、显存和 Jetson Nano 延迟。\n2. 在自建数据集上建立独立 train/val/test 划分并开展迁移训练。\n3. 完成相机、四轮底盘和三自由度机构的坐标标定及像素—物理坐标转换。\n4. 开展粗定位循迹、停靠、三轴精定位、宽度测量和裂缝修复实验。\n5. 若论文仍要求超过 OCRNet，需要另行批准更强结构替换；当前轻量模块筛选没有达到 SOTA。')
p.write_text(s,encoding='utf-8')
print('tracker updated')


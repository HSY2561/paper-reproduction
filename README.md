# 道路裂缝视觉分割与移动机器人精细修复

本仓库保存论文相关的训练配置、验证与测试结果、审计回执、图表、复现源码和环境说明。历史波次的原始证据保留在各自的 `experiments/` 目录中。

## 最新进度

### AW + 背景加权 CE（2026-11-12）

[AW 精度诊断与单变量修正报告](experiments/segformer_crack500_aw_wavelet_20261112/AW精度诊断与单变量修正报告_20261112.md)固定 SegFormer-B0 + Aligned + Haar WTConv，仅将 CrossEntropyLoss 权重设为 `[background=1.10, crack=1.0]`。训练 exit=0，ordinary checkpoint 按验证 mIoU 选中。Crack500 test 的 Precision 为 73.159%、Recall 为 74.734%、F1 为 73.938%、裂缝 IoU 为 58.652%、mIoU 为 77.789%。使用验证集确定的阈值 0.525 后，Precision 为 74.401%，但 Recall 和细线召回继续下降。

checkpoint SHA256 为 `8b5ed4b1cef32e9e9bdc9dd2f657176a566cc106242e3800fde39692add26a09`。该结果是单种子、单变量修正和验证集阈值诊断，适合精度优先部署策略分析，不能直接写成结构全面提升或 SOTA 结论。

- [AW + 背景加权 CE 结果目录](experiments/segformer_crack500_aw_wavelet_20261112/results/)
- [阈值扫描脚本](experiments/segformer_crack500_aw_wavelet_20261112/threshold_sweep_aw.py)
- [单变量配置](experiments/segformer_crack500_aw_wavelet_20261112/configs/AW_CEbg110.yml)

### 原始 AW 对照

[AW 重新训练与六模型对照报告](experiments/segformer_crack500_aw_wavelet_20261112/AW重新训练与六模型对照报告_20261112.md)保留未加权 AW 的完整结果，用于判断背景加权 CE 的变化来源。

### 其他论文结果

- [Crack500 Wave17 四组 AvgPool WTConv 消融](Crack500_Aligned_WTConv_消融报告_20261031.md)
- [VOCdevkit3000 四组审计结果](VOCdevkit3000_四组消融结果表_真实审计_20261106.md)
- [Wave16 损失函数复核](experiments/segformer_vocdevkit3000_aw_loss_wave16_20261106/results/aw_loss_comparison.csv)
- [Wave18 多数据集扫描](experiments/segformer_priority_wave18_20261107/reports/wave18_report.md)

## 复现材料

`experiments/` 按波次保存配置、源码、provenance、真实退出回执、指标 CSV 和图表。运行环境见 [runtime/ENVIRONMENT.md](runtime/ENVIRONMENT.md)，依赖清单见 [runtime/requirements-wave17.txt](runtime/requirements-wave17.txt)。

数据集、训练缓存和未列入发布清单的权重保留在本地环境。AW 和加权 CE 的 checkpoint 路径、选择记录和 SHA256 已上传，权重本体按仓库发布约定留在本地。使用软件时请按环境说明配置 PaddlePaddle、Pillow、NumPy 以及本地 checkpoint 路径。

## 结论边界

结果文件中的 `test_used`、退出码、选择迭代、阈值和 SHA256 是审计字段。训练完成不自动等于通过科学结论门槛。加权 CE 的收益主要体现为减少背景误检，代价是 Recall 和细线连通性下降。多数据集、跨种子、稳定延迟、显存和机器人闭环结果需要分别依据各波次回执判断。

## 仓库整理

根目录只保留当前论文入口、结果索引、图表和环境说明。历史实验目录中的原始日志、指标、失败证据、预登记计划、配置、源码和复现依赖保留。缓存、进程锁、训练中间目录、数据集和下载的第三方全文不上传。

远端仓库：[HSY2561/paper-reproduction](https://github.com/HSY2561/paper-reproduction)

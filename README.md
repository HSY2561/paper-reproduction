# 道路裂缝视觉分割与移动机器人精细修复

本仓库保存论文相关的训练配置、验证与测试结果、审计回执、图表、复现源码和环境说明。历史波次的原始证据保留在各自的 `experiments/` 目录中。

## 最新进度

### Crack500 AW（2026-11-12）

[AW 重新训练与六模型对照报告](experiments/segformer_crack500_aw_wavelet_20261112/AW重新训练与六模型对照报告_20261112.md)记录了标准一层正交 Haar DWT/IWT 版本的 seed42 结果。训练 exit=0，ordinary 验证 mIoU=82.7961%，冻结后在 1124 张 Crack500 test 上得到 mIoU=77.5142%、裂缝 IoU=58.2222%、F1=73.5955%、边界 F1@2px=36.7947%。checkpoint SHA256 为 `44f64fd658d16decfd59d5807c2a80fc95d25453f563020b0642ff972e79f2a2`。当前结果是单种子协议内的比较，不能直接作为多种子稳定性或 SOTA 结论。

- [AW 结果与逐图 CSV](experiments/segformer_crack500_aw_wavelet_20261112/results/)
- [AW 执行计划](experiments/segformer_crack500_aw_wavelet_20261112/EXPERIMENT_PLAN.md)
- [Haar WTConv 实现](experiments/segformer_crack500_aw_wavelet_20261112/src/PaddleSeg-2.10.0/paddleseg/models/wtconv_wavelet.py)

AW 与六个历史公开模型的 Crack500 对照也已保存。由于公开模型使用历史冻结后端和初始化，表格用于当前复现范围内的对照。

### 已有论文结果

- [Crack500 Wave17 AvgPool WTConv 四组消融](Crack500_Aligned_WTConv_消融报告_20261031.md)
- [VOCdevkit3000 四组审计结果](VOCdevkit3000_四组消融结果表_真实审计_20261106.md)
- [Wave16 损失函数复核](experiments/segformer_vocdevkit3000_aw_loss_wave16_20261106/results/aw_loss_comparison.csv)
- [Wave18 多数据集扫描](experiments/segformer_priority_wave18_20261107/reports/wave18_report.md)
- [Wave18 Crack500 evaluator 审计](analysis/wave18_crack500_evaluator_audit_20261109.md)

## 复现材料

`experiments/` 按波次保存配置、源码、provenance、真实退出回执、指标 CSV 和图表。运行环境见 [runtime/ENVIRONMENT.md](runtime/ENVIRONMENT.md)，依赖清单见 [runtime/requirements-wave17.txt](runtime/requirements-wave17.txt)。

数据集、训练缓存和未列入发布清单的权重保留在本地环境。AW 的 checkpoint 路径、选择记录和 SHA256 已上传，权重本体按仓库发布约定留在本地。使用软件时请按环境说明配置 PaddlePaddle、Pillow、NumPy 以及本地 checkpoint 路径。

## 结论边界

结果文件中的 `test_used`、退出码、选择迭代和 SHA256 是审计字段。训练完成不自动等于通过科学结论门槛。多数据集、跨种子、稳定延迟、显存和机器人闭环结果需要分别依据各波次回执判断。

## 仓库整理

根目录只保留当前论文入口、结果索引、图表和环境说明。历史实验目录中的原始日志、指标、失败证据、预登记计划、配置、源码和复现依赖保留。缓存、进程锁、训练中间目录、数据集和下载的第三方全文不上传。

远端仓库：[HSY2561/paper-reproduction](https://github.com/HSY2561/paper-reproduction)

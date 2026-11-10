# 道路裂缝视觉分割与移动机器人精细修复

本仓库保存论文相关的训练配置、验证与测试结果、审计回执、图表、复现源码和运行环境说明。当前 `main` 的内容按最新结果组织，历史波次的原始证据仍保留在各自的 `experiments/` 目录中。

## 当前结果入口

- [Crack500 Wave17 四组 Aligned/WTConv 消融报告](Crack500_Aligned_WTConv_消融报告_20261031.md)
- [Crack500 Wave17 结果表](experiments/segformer_crack500_aligned_wtconv_ablation_wave17_20261030/results/crack500_aligned_wtconv_ablation_table.csv)
- [VOCdevkit3000 四组审计结果](VOCdevkit3000_四组消融结果表_真实审计_20261106.md)
- [VOCdevkit3000 Wave15 原始结果](experiments/segformer_vocdevkit3000_aligned_wtconv_ablation_20261031/results/voc_ablation_table.csv)
- [Wave16 损失函数复核结果](experiments/segformer_vocdevkit3000_aw_loss_wave16_20261106/results/aw_loss_comparison.csv)
- [Wave18 多数据集汇总](experiments/segformer_priority_wave18_20261107/reports/wave18_report.md)

Wave17 的 Crack500 表包含四组模型冻结验证和 1124 张 test 的统一比较。Wave15/Wave16 的 VOCdevkit3000 结果保留训练、官方验证和独立边界评价回执。Wave18 是跨数据集优先级扫描，目前只保留真实扫描结果和缺失状态，不补写不存在的指标。

## 复现材料

- `experiments/` 按波次保存配置、源码、provenance、真实退出回执、指标 CSV 和图表。
- [Wave17 工程](experiments/segformer_crack500_aligned_wtconv_ablation_wave17_20261030/)
- [VOCdevkit3000 Wave15 工程](experiments/segformer_vocdevkit3000_jointfix_wave15_20261106/)
- [VOCdevkit3000 Wave16 工程](experiments/segformer_vocdevkit3000_aw_loss_wave16_20261106/)
- [Wave18 工程](experiments/segformer_priority_wave18_20261107/)
- [运行环境说明](runtime/ENVIRONMENT.md)
- [Wave17 依赖清单](runtime/requirements-wave17.txt)
- [图表与方法章节](SegFormer_Aligned_WTConv_方法与实验章节_20261031.md)
- [论文段落与表图交付](SegFormer_Aligned_WTConv_表格图示与论文段落交付_20261031.md)

数据集、训练缓存和大部分训练权重保留在本地环境。仓库中的 checkpoint 仅按现有发布清单保留，并与对应 SHA256 和配置关联。使用软件时请按运行环境说明配置 PaddlePaddle、Pillow、NumPy 以及本地 checkpoint 路径。

## 数据与结论边界

结果文件中的 `test_used`、退出码、选择迭代和 SHA256 是审计字段。训练完成不自动等于通过科学结论门槛。多数据集、跨种子、稳定延迟、显存和机器人闭环结果需要分别依据各波次回执判断。

## 仓库清理说明

本次整理移除了根目录中被实验目录和最新结果报告替代的旧时间戳报告、旧计划和重复快照。实验目录中的原始日志、指标、失败证据、预登记计划、配置、源码和复现依赖保留，便于导师追踪论文进度。

远端仓库：[HSY2561/paper-reproduction](https://github.com/HSY2561/paper-reproduction)

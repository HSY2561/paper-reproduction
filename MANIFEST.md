# 交付索引

此索引从本次交付开始登记，不声称枚举此前全部工程文件。

## 2026-10-19：Crack500 文件核验与复核进度更新

- [数据来源、目录与完整性结论](analysis/crack500_integrity_20261019/README.md)
- [机器可读核验结果](analysis/crack500_integrity_20261019/audit.json)、[逐文件 SHA256](analysis/crack500_integrity_20261019/file_sha256.json)
- 3,368 对裁剪图像/标签全部存在并可解码，与本地原包的清单和前景标签一致；文件名前缀跨划分风险独立记录，未擅自调整划分。
- 候选复核新增 A0 seed43 的完整结果；A21 seed43 的保存状态为训练中，三 seed 尚未齐备。

## 2026-10-19：Wave7 筛选完成与候选复核

- [阶段报告](SegFormer_Aligned模块筛选阶段报告_20261019.md)、[代码与证据入口](experiments/segformer_aligned_wave7_20261018/README.md)
- [全量结果 CSV](experiments/segformer_aligned_wave7_20261018/results/ablation_percent_20261019.csv)、[独立审计](experiments/segformer_aligned_wave7_20261018/results/audit_complete_20261019.json)
- [归因与多 seed 复核进度](experiments/segformer_aligned_wave7_20261018/followup_20261019/refine-logs/RESULTS.md)
- 状态：22 项完整筛选训练已完成，A21/A10 已冻结进入复核；速度稳定性未通过，多 seed、六个公开模型和本轮统一 test 仍待完成。

## 2026-10-17：Wave6 模块消融训练结果

- [最新结果与改进建议](SegFormer模块消融结果与改进建议_20261017.md)
- [代码、配置与证据入口](experiments/segformer_balance_wave6_20261016/README.md)
- [完整消融结果表](experiments/segformer_balance_wave6_20261016/refine-logs/EXPERIMENT_RESULTS.md)
- 10 月 18 日更新：SE 已重新完成 18k 训练与验证，累计 15 个完整组（13 模块、2 宽度）；PSP smoke 失败。10 月 17 日超时仅是历史记录，当前结论仍限于单 seed 验证集筛选。

## 2026-10-16：10×6.5 m 场地方案

- [完整实验方案](10m×6.5m场地_麦克纳姆扫描建图与定点修复完整实验方案_20261016.md)
- [本轮数据、图表、模板与证据清单](experiments/field_10x6p5_20261016/MANIFEST.md)
- 状态：方案与数值规划已完成；单张航拍推理成功执行但视觉结果不合格；实车采集、标定、建图和修复待执行。

## 2026-10-17：缩小验证区域与效率改进

- [文献依据及执行建议](实验范围缩小与效率改进_文献依据及执行建议_20261017.md)
- [小区域路线计算](experiments/field_efficiency_20261017/roi_calculations.json)
- [采集预算图](experiments/field_efficiency_20261017/采集时间预算_范围与视野对照.png)
- 状态：文献核验与数值规划完成；推荐2×1.5 m先导、3×2 m主验证。实际区域待量测，未运行实车。

# 道路裂缝视觉分割与移动机器人精细修复

本仓库提供当前代码、训练证据与论文材料，供导师查看项目进展。最新同步日期：2026-10-20。

## 最新训练结果

最新 [Wave8 损失实验](experiments/segformer_loss_wave8_20261019/README.md)已完成 L05–L13 九项配置的 18k 训练与独立验证。最新完成的 L13（OHEM CE）验证裂缝 IoU 68.3223%、F1 81.1803%、mIoU 83.0583%、边界 F1 45.8753%；这是 seed42 单项结果，不代表最终门槛通过。L14（Softmax Focal gamma 1.0）的最近状态为训练中；L01–L04 已记录为梯度预检失败，性能留空。各方案固定同一 Aligned 结构，不能将损失变更写成减参或推理加速。详见[当前进度表](SegFormer损失函数消融进展_20261019.md)和[预登记计划](SegFormer损失函数优化20方案执行计划_20261019.md)。

[全部实验结果总报告](当前全部实验结果总报告_20261020.md)汇总历史结果、三 seed 对照、负结果与局限，附[CSV、图表及来源索引](实验结果汇总_20261020/复现索引.md)。报告快照时间见正文，Wave8 后续进度以上述实时进度表为准。

新增[原始 SegFormer-B0 训练与验证曲线](figures/segformer_training_curve_20261020/绘图说明与图注.md)，附原始 CSV、来源 SHA 和绘图脚本。这是历史 width256、seed42 基线的曲线，不是最新损失候选；蓝线为验证 mIoU，原日志没有记录验证 Loss。

2026-10-20 新增[阶段核验与兼容修复说明](SegFormer损失函数优化阶段核验与兼容修复说明_20261020.md)：Dice 组合损失的独立兼容源码及 CPU 预检证据已保存，恢复流程最近处于等待主队列结束状态，尚未开始 GPU 运行；原失败证据不覆盖。详情见[执行补充](experiments/segformer_loss_wave8_20261019/refine-logs/PLAN_ADDENDUM_20261020.md)。CPU 预检通过不代表 GPU 预检或完整训练完成。

**Wave7 的 22 项筛选训练及后续 9 项归因/多 seed 训练已全部完成。** 后续队列于 2026-10-19 20:19 生成完成回执，B0、A0、A21、A10 的 seed42/43/44 结果均已齐备。下表直接来自已保存的三 seed 汇总，数值为验证集均值 ± 样本标准差（百分数）。

| 模型 | 裂缝 IoU（%） | F1（%） | mIoU（%） | 参数量（M） |
|---|---:|---:|---:|---:|
| B0：原始 SegFormer-B0 | 68.039 ± 0.219 | 80.980 ± 0.155 | 82.908 ± 0.106 | 3.715170 |
| A0：W128 + Aligned | 68.335 ± 0.420 | 81.189 ± 0.297 | 83.063 ± 0.212 | 3.470434 |
| A21：Aligned + Polarized | 68.526 ± 0.393 | 81.324 ± 0.277 | 83.161 ± 0.200 | 3.503330 |
| A10：Aligned + AFNB | 68.102 ± 0.151 | 81.025 ± 0.107 | 82.942 ± 0.073 | 3.656994 |

这些是 **Crack500 验证集结果**。三 seed 训练完成不代表最终门槛已经通过，仍需按预登记条件复核模块贡献、精度与速度。已有速度测量稳定性未通过，且候选中位延迟慢于 B0；六个公开模型及本轮统一 test 仍待完成。

Wave7 的 DAPPM 反向 smoke 失败、PSP 兼容性暂缓，保留失败证据并不填性能。此前 Wave6 已完成 15 个完整组（13 种模块、2 个宽度对照），包括重新完成的 SE。

- [最新三 seed 汇总与完成证据](experiments/segformer_aligned_wave7_20261018/followup_20261019/README.md)、[全部 9 项结果](experiments/segformer_aligned_wave7_20261018/followup_20261019/refine-logs/RESULTS.md)、[机器可读均值与标准差](experiments/segformer_aligned_wave7_20261018/followup_20261019/results/three_seed_summary.json)。
- [Wave7 代码与证据入口](experiments/segformer_aligned_wave7_20261018/README.md)、[seed42 筛选阶段报告](SegFormer_Aligned模块筛选阶段报告_20261019.md)、[22 项核验记录](experiments/segformer_aligned_wave7_20261018/results/audit_complete_20261019.json)。
- [归因与多 seed 预登记计划](experiments/segformer_aligned_wave7_20261018/followup_20261019/refine-logs/PLAN.md)：候选冻结、9 项预算与判定规则。
- [Wave6 单模块结果](SegFormer模块消融结果与改进建议_20261017.md)、[Wave6 复现入口](experiments/segformer_balance_wave6_20261016/README.md)：保留母体与独立模块的对照证据。

## 当前论文与机器人方案

- [10 月 19 日修复路线选择](修复实验路线选择与空鼓检测论文借鉴_20261019.md)：先完成单窗口局部闭环，再验证多个独立站点；长裂缝跨窗修复作为扩展。目前仍是方案，尚未实车执行。
- [场地范围与效率方案（10月17日）](实验范围缩小与效率改进_文献依据及执行建议_20261017.md)：先做 2×1.5 m 先导验证，再做 3×2 m 主验证；目前是数值规划，尚未实车验证。
- [视觉方法与实验章节](SegFormer_B0_视觉分割方法与实验章节_修订版_20261014.md)：保留论文结构与既有实验说明，最新数字以本页链接的 Wave7 阶段报告为准。
- [标定与扫描建图路线](标定与弓字形扫描建图_实施路线与资料清单_20261016.md)、[两阶段先检后修方案](两阶段先检后修裂缝地图与定点修复方案_20261015.md)：系统接口与待执行实物实验。
- [真实分割到轨迹处理](P0_real_segformer_trajectory/README.md)：真实 SegFormer 掩膜、骨架及离线轨迹；毫米坐标仍使用占位映射。
- [教学仿真](experiments/two_pass_demo_20261016/一条裂缝_两遍作业仿真说明.md)、[10×6.5 m 场地完整方案](10m×6.5m场地_麦克纳姆扫描建图与定点修复完整实验方案_20261016.md)：保留算法演示及完整场地设计，首次实测范围按 10 月 17 日建议缩小。
- [项目复盘](论文项目完整复盘报告_20261014.md)、[交付索引](MANIFEST.md)。

## Crack500 数据集

当前训练读取 `C:\Users\user\algolab_data` 的 `traincrop/valcrop/testcrop` 与同目录的划分清单，分别为 1,896/348/1,124 对图像和标签。该版本是公开 Crack500 的裁剪分割基准，原始数据集的“500”指高分辨率原图数。3,368 对文件完整性检查通过；跨划分有 6 个疑似同源的文件名前缀，尚需源图级确认，不能据文件齐全断言划分完全独立。核验方法、原作者来源、实际目录及完整结果见 [Crack500 完整性核验](analysis/crack500_integrity_20261019/README.md)。数据本体保留本地。

## 仓库整理与复现说明

当前目录保留最新说明及仍有用的论文、实验材料；被替代的旧草稿和自动生成的时间戳副本已取消跟踪，本地仍保留。此前已发布的版本可从 [Git 历史](https://github.com/HSY2561/paper-reproduction/commits/main) 查看。历史实验的原始代码、配置、失败/中断记录和结果保留，作为对照与可追溯证据，不与本轮结果混用。

最新训练源码在 `experiments/segformer_aligned_wave7_20261018/src/PaddleSeg-2.10.0/`，当前复核入口在该目录的 `followup_20261019/`。复现前阅读波次 README 和运行契约，按自己的机器调整 Python、数据及权重路径；旧波次使用各自的配置和源码版本。

数据集、训练权重、环境缓存和本机进程锁不上传；权重选模记录与 SHA256 保留。`MANIFEST.md` 和 `manifest_paths.json` 中的实验登记可能包含这些本地文件，不代表每个文件都托管在 GitHub。所有日志、heartbeat 和状态文件均是提交快照，查看最新状态时应以时间戳和退出回执为准。

远端仓库：[HSY2561/paper-reproduction](https://github.com/HSY2561/paper-reproduction)。

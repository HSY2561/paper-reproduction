# 道路裂缝视觉分割与移动机器人精细修复

本仓库提供当前代码、训练证据与论文材料，供导师查看项目进展。最新同步日期：2026-10-19。

## 最新训练结果

**Wave7 的 Aligned 第二模块筛选已完成 22 项完整训练**，登记的另外两项分别为 DAPPM 反向 smoke 失败、PSP 兼容性暂缓。各完成组保留 18,000 iter 日志、真实退出码、冻结权重 SHA256、独立验证和逐图指标。A21（Polarized）与 A10（AFNB）已冻结为复核候选。

| 配置 | 裂缝 IoU（%） | F1（%） | mIoU（%） | 参数量（M） |
|---|---:|---:|---:|---:|
| 原始 SegFormer-B0 | 67.8174 | 80.8228 | 82.8021 | 3.715170 |
| A0：W128 + Aligned | 67.8520 | 80.8474 | 82.8184 | 3.470434 |
| A21：Aligned + Polarized | 68.0773 | 81.0071 | 82.9321 | 3.503330 |
| A10：Aligned + AFNB | 68.1660 | 81.0699 | 82.9836 | 3.656994 |

以上是 **Crack500 验证集、seed42 筛选结果**，尚不是最终模型结论。PAM 虽然区域 IoU 最高，但未通过完整贡献门槛。A21/A10 进入归因与匹配 seed43/44 复核；三 seed 汇总、稳定速度、六个公开模型及本轮统一 test 尚未完成。本轮速度测量稳定性未通过，且两个候选的中位延迟均慢于 B0，不能声称加速。

复核队列共 9 项新训练，当前已取得 AFNB 单模块 seed42 与 B0 seed43 的完成记录，其余进度以 [复核结果表](experiments/segformer_aligned_wave7_20261018/followup_20261019/refine-logs/RESULTS.md)及各项状态/退出回执为准。训练继续产生新记录，GitHub 展示本次提交快照。

此前 Wave6 的 SE 已于 10 月 18 日补训完成，Wave6 共 **15 个完整组（13 种模块、2 个宽度对照）**；PSP 仍为 smoke 失败。旧的 SE 超时记录作为历史保留，不代表最新状态。

- [10 月 19 日筛选阶段报告](SegFormer_Aligned模块筛选阶段报告_20261019.md)：完整结果、候选选择、图表与速度限制。
- [Wave7 代码与证据入口](experiments/segformer_aligned_wave7_20261018/README.md)、[完整 CSV](experiments/segformer_aligned_wave7_20261018/results/ablation_percent_20261019.csv)、[22 项核验记录](experiments/segformer_aligned_wave7_20261018/results/audit_complete_20261019.json)。
- [归因与多 seed 复核计划](experiments/segformer_aligned_wave7_20261018/followup_20261019/refine-logs/PLAN.md)：候选冻结、9 项预算与预登记判定。
- [Wave6 单模块结果](SegFormer模块消融结果与改进建议_20261017.md)、[Wave6 复现入口](experiments/segformer_balance_wave6_20261016/README.md)：保留母体与独立模块的对照证据。

## 当前论文与机器人方案

- [场地范围与效率方案（10月17日）](实验范围缩小与效率改进_文献依据及执行建议_20261017.md)：先做 2×1.5 m 先导验证，再做 3×2 m 主验证；目前是数值规划，尚未实车验证。
- [视觉方法与实验章节](SegFormer_B0_视觉分割方法与实验章节_修订版_20261014.md)：保留论文结构与既有实验说明，最新数字以本页链接的 Wave7 阶段报告为准。
- [标定与扫描建图路线](标定与弓字形扫描建图_实施路线与资料清单_20261016.md)、[两阶段先检后修方案](两阶段先检后修裂缝地图与定点修复方案_20261015.md)：系统接口与待执行实物实验。
- [真实分割到轨迹处理](P0_real_segformer_trajectory/README.md)：真实 SegFormer 掩膜、骨架及离线轨迹；毫米坐标仍使用占位映射。
- [教学仿真](experiments/two_pass_demo_20261016/一条裂缝_两遍作业仿真说明.md)、[10×6.5 m 场地完整方案](10m×6.5m场地_麦克纳姆扫描建图与定点修复完整实验方案_20261016.md)：保留算法演示及完整场地设计，首次实测范围按 10 月 17 日建议缩小。
- [项目复盘](论文项目完整复盘报告_20261014.md)、[交付索引](MANIFEST.md)。

## 仓库整理与复现说明

当前目录保留最新说明及仍有用的论文、实验材料；被替代的旧草稿和自动生成的时间戳副本已取消跟踪，本地仍保留。此前已发布的版本可从 [Git 历史](https://github.com/HSY2561/paper-reproduction/commits/main) 查看。历史实验的原始代码、配置、失败/中断记录和结果保留，作为对照与可追溯证据，不与本轮结果混用。

最新训练源码在 `experiments/segformer_aligned_wave7_20261018/src/PaddleSeg-2.10.0/`，当前复核入口在该目录的 `followup_20261019/`。复现前阅读波次 README 和运行契约，按自己的机器调整 Python、数据及权重路径；旧波次使用各自的配置和源码版本。

数据集、训练权重、环境缓存和本机进程锁不上传；权重选模记录与 SHA256 保留。`MANIFEST.md` 和 `manifest_paths.json` 中的实验登记可能包含这些本地文件，不代表每个文件都托管在 GitHub。所有日志、heartbeat 和状态文件均是提交快照，查看最新状态时应以时间戳和退出回执为准。

远端仓库：[HSY2561/paper-reproduction](https://github.com/HSY2561/paper-reproduction)。

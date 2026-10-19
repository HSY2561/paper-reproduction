# 道路裂缝视觉分割与移动机器人精细修复

本仓库提供当前代码、训练证据与论文材料，供导师查看项目进展。最新同步日期：2026-10-19。

## 最新训练结果

新增 [Wave8 损失实验准备目录](experiments/segformer_loss_wave8_20261019/)：损失配置目录代码、梯度预检脚本和上一轮复核审计已开始整理。本次快照尚未取得该波正式训练完成回执，不能据准备代码填写性能；已完成结果仍见下方 Wave7 三 seed 表。

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

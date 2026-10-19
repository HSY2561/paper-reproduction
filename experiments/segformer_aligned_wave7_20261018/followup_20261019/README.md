# 候选归因与三 seed 复核：训练队列已完成

最新完成回执：2026-10-19 20:19（UTC+8）。本轮 9 项新训练均 complete：AFNB 独立模块 seed42 一项，以及 B0/A0/A21/A10 各 seed43/44 共八项。各模型 seed42 复用既有冻结结果，因此四模型的三 seed 已齐备。

## 已保存的验证集汇总

以下直接对应 `results/three_seed_summary.json` 的 `mean` 和 `sample_std`，数值为百分数，种子为 42/43/44；标准差采用样本标准差。

| 模型 | 裂缝 IoU（%） | F1（%） | mIoU（%） |
|---|---:|---:|---:|
| B0 | 68.039 ± 0.219 | 80.980 ± 0.155 | 82.908 ± 0.106 |
| A0：W128 + Aligned | 68.335 ± 0.420 | 81.189 ± 0.297 | 83.063 ± 0.212 |
| A21：Aligned + Polarized | 68.526 ± 0.393 | 81.324 ± 0.277 | 83.161 ± 0.200 |
| A10：Aligned + AFNB | 68.102 ± 0.151 | 81.025 ± 0.107 | 82.942 ± 0.073 |

这里记录的是训练完成及已保存统计，不替代预登记门槛审查。已有速度测量不稳定，且候选中位延迟慢于 B0；不能称为已加速或已通过完整效率目标。六个原始公开模型、本轮冻结统一 test 与最终交付尚未完成。新结果均来自 train/val，没有用 test 选模。

## 可核对的证据

- [九项运行结果表](refine-logs/RESULTS.md)、[逐次 CSV](results/seed_runs.csv)、[完整三 seed 数值](results/three_seed_summary.json)。
- [队列完成回执](results/queue_done.json)、[预登记计划](refine-logs/PLAN.md)、[运行契约](provenance/RUNTIME.md)。
- `results/*_train_exit.json`、`*_official_val_exit.json`、`*_pixel_boundary_exit.json`、`*_profile_exit.json`：真实退出码。
- `logs/`：完整训练与独立验证记录；`checkpoints/*/selection.json`：选模来源与 SHA256；`*_val_per_image.csv`：全验证集逐图指标。
- [Wave7 主波入口](../README.md)及[seed42 筛选报告](../../../SegFormer_Aligned模块筛选阶段报告_20261019.md)：保留初始筛选结果和候选冻结依据。

主队列和本轮九项队列均已完成。旧 heartbeat、active_process、PID 和文档中的启动时间属于历史快照，不表示当前仍有训练进程；查看后续阶段应先核对新回执。无需为同步仓库再次启动已完成队列。权重和原始数据保留本地，GitHub 保存代码、配置、日志、指标及选模记录。

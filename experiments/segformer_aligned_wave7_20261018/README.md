# Wave7：SegFormer-B0 + Aligned 第二模块筛选与复核

2026-10-19 同步状态：登记 24 项第二模块机制，22 项完成 18k 训练、冻结选模与独立验证，DAPPM 反向 smoke 失败，PSP 兼容性暂缓。B0、W128、A0（W128 + Aligned）复用核验过的冻结对照。

已冻结 A21（Polarized）和 A10（AFNB）进入归因及匹配多 seed 复核。当前仍为验证集单 seed 筛选证据，不能视作最终模型。速度复测稳定性未通过，且两个候选的中位延迟均慢于 B0；参数减少不等于加速。

## 当前阅读入口

- [阶段报告](../../SegFormer_Aligned模块筛选阶段报告_20261019.md)：全部结果、选择理由、负结果与速度限制。
- [筛选结果表](refine-logs/EXPERIMENT_RESULTS.md)、[百分数 CSV](results/ablation_percent_20261019.csv)、[审计记录](results/audit_complete_20261019.json)。
- [预注册计划](refine-logs/EXPERIMENT_PLAN_20261018.md)、[当前执行计划](refine-logs/EXPERIMENT_PLAN.md)、[模块目录](provenance/module_catalog.md)。
- [逐图诊断](diagnostics/VISUAL_ERROR_DIAGNOSIS.md)：保留全 348 张验证图的配对与空间误差记录。
- [候选复核计划](followup_20261019/refine-logs/PLAN.md)、[复核结果](followup_20261019/refine-logs/RESULTS.md)、[冻结候选](followup_20261019/provenance/frozen_candidates.json)。

## 代码与运行说明

独立 PaddleSeg 源码在 `src/PaddleSeg-2.10.0/`，模型配置在 `configs/`。主波次编排为 `workflow.py` / `aligned_queue.py`；当前归因与多 seed 阶段位于 `followup_20261019/`，入口是 `followup_queue.py`。主波次已完成，不应为查看进度重新启动训练。

运行前阅读 [主波运行契约](provenance/RUNTIME.md)及[复核运行契约](followup_20261019/provenance/RUNTIME.md)，核对 Python、Crack500 train/val、预训练/冻结权重和数据清单的本机路径。当前实现包含本地 Windows/Paddle3.4/RTX5060 环境路径，其他机器需先调整。训练中的进程、系统任务和锁应按运行契约核实，避免重复启动。

最新正式结果由各组 `*_status.json`、`*_train_exit.json`、选模记录、独立验证及日志共同确认；同名汇总表在完成阶段刷新，可能落后于正在运行的状态文件。本次提交时已有 AFNB 单模块 seed42、B0 seed43 完成记录，其他运行见复核目录。三 seed 未齐前不报告均值和标准差。

`summarize_wave7.py` 用于汇总已有结果和生成图，`diagnostics/` 保留诊断代码；`candidate_latency.py`、`paired_latency.py` 保留速度测量实现。复现实验所需的源代码、配置、日志、CSV、退出回执和权重 SHA256 均随仓库保留，权重/数据集/环境缓存/本机锁不上传。

GitHub 文件是提交快照；自动生成的时间戳重复报告留在本地，初始预注册及失败证据保留。`MANIFEST.md` 是本地过程登记，可能包含未上传的权重和临时路径。本轮未用 test 筛选；历史阶段曾有 test 结果，不能称整个项目从未观察 test。

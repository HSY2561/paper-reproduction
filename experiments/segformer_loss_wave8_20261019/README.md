# Wave8：固定 Aligned 结构的损失函数消融

本波固定 SegFormer-B0 width128 + Aligned（A0）结构，登记 20 个损失配置，先核验共同基线、数据与损失梯度，再按既定训练协议执行。当前是进行中的实验，不能把准备、smoke 或训练完成直接当作最终目标通过。

## 2026-10-20 同步快照

- L05（CE + 0.1 Lovasz）已 complete：真实训练和独立验证/边界评估/profile 回执均为 exit0，训练日志到 18000/18000。验证裂缝 IoU 68.1953%、F1 81.0906%、mIoU 82.9721%、边界 F1 44.1729%。它是 seed42 单项结果。
- L06（CE + 0.25 Lovasz）的最近状态为 training。汇总表在阶段完成时刷新，可能仍显示 pending；实际运行进度应结合 `results/*_status.json`、heartbeat、日志和退出回执核对。
- L01–L04 已记录为 gradient_preflight_failed，保留梯度预检证据，性能不填数值。其他方案按各项回执更新。
- C0 是复用的合法 A0 + CE 冻结基线，不计作新训练。两个原替补配置在启动前停用并保存在 `provenance/prelaunch_disabled_reserves/`，不根据验证成绩增加替补。

## 阅读入口

- [完整执行计划](refine-logs/EXPERIMENT_PLAN.md)、[根目录计划](../../SegFormer损失函数优化20方案执行计划_20261019.md)。
- [进度表](refine-logs/RESULTS.md)、[CSV](results/loss_ablation.csv)、[机器可读结果](results/comparison.json)。
- [梯度预检](results/gradient_preflight.json)、[预检退出回执](results/gradient_preflight_exit.json)、[运行契约](provenance/RUNTIME.md)。
- [L05 训练回执](results/L05_CE_Lovasz_train_exit.json)、[独立验证](results/L05_CE_Lovasz_val.json)及同目录逐图 CSV。
- [上一轮九项审计](provenance/previous_followup_audit.json)、[上一轮三 seed 完成说明](../segformer_aligned_wave7_20261018/followup_20261019/README.md)。

## 代码及结果边界

入口 `loss_queue.py`，损失目录 `loss_catalog.py`，独立预检 `gradient_preflight.py`；正式源码在 `src/PaddleSeg-2.10.0/`，配置在 `configs/`。复现前按运行契约核对实际 Python、数据、预训练及冻结控制权重路径。查看或同步进度无需重新启动队列。

本轮只使用 Crack500 train/val 筛选，失败和未完成项不填写性能。损失函数变更不改变模型参数量或推理结构，不能声称损失优化减少参数或加快推理。新损失的多 seed、公开模型与本轮最终 test 尚待后续核验。

GitHub 展示提交时的代码和结果快照；训练中的日志继续增长。原始数据、训练权重、本机进程锁和重复时间戳报告留在本地；配置、源码、日志、逐图指标、真实退出回执与选模 SHA 保留在仓库。

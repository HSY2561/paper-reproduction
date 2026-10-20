# Wave8：固定 Aligned 结构的损失函数消融

本波固定 SegFormer-B0 width128 + Aligned（A0）结构，登记 20 个损失配置，先核验共同基线、数据与损失梯度，再按既定训练协议执行。当前是进行中的实验，不能把准备、smoke 或训练完成直接当作最终目标通过。

## 2026-10-20 同步快照

- L05（CE + 0.1 Lovasz）已 complete：真实训练和独立验证/边界评估/profile 回执均为 exit0，训练日志到 18000/18000。验证裂缝 IoU 68.1953%、F1 81.0906%、mIoU 82.9721%、边界 F1 44.1729%。它是 seed42 单项结果。
- L06（CE + 0.25 Lovasz）于 02:35 已 complete，18k 训练和独立验证回执均成功；验证裂缝 IoU 68.6376%、F1 81.4025%、mIoU 83.2048%、边界 F1 44.7172%，是 seed42 单项结果。
- L07（CE + 0.5 Lovasz）于 04:22 已 complete，18k 训练和独立验证回执均成功；验证裂缝 IoU 68.4557%、F1 81.2744%、mIoU 83.0994%、边界 F1 45.4401%，是 seed42 单项结果。
- L08（加权 CE [1.5, 1.0]）于 06:04 已 complete，18k 训练、独立验证、边界评估与 profile 四项回执均为 exit0；验证裂缝 IoU 66.8904%、F1 80.1609%、mIoU 82.3095%、边界 F1 44.1797%，是 seed42 单项结果。选中普通 best_model 的第 5000 步权重，选模 SHA 已保存，权重文件仅留本地。
- L09（加权 CE [2.0, 1.0]）于 07:42 已 complete，18k 训练、独立验证、边界评估与 profile 四项回执均为 exit0；验证裂缝 IoU 65.4755%、F1 79.1362%、mIoU 81.5913%、边界 F1 43.5890%，是 seed42 单项结果。选中普通 best_model 的第 5000 步权重，选模 SHA 已保存，权重文件仅留本地。
- L10（加权 CE [1.0, 2.0]）于 09:25 已 complete，18k 训练、独立验证、边界评估与 profile 四项回执均为 exit0；验证裂缝 IoU 67.7748%、F1 80.7926%、mIoU 82.6971%、边界 F1 41.7729%，是 seed42 单项结果。选中普通 best_model 的第 9000 步权重，选模 SHA 已保存，权重文件仅留本地。
- L11（CE top 0.5）于 10:47 已 complete，18k 训练、独立验证、边界评估与 profile 四项回执均为 exit0；验证裂缝 IoU 68.0463%、F1 80.9851%、mIoU 82.9143%、边界 F1 45.1958%，是 seed42 单项结果。选中普通 best_model 的第 8000 步权重，选模 SHA 已保存，权重文件仅留本地。
- L12（CE top 0.25）于 11:45 已 complete，18k 训练、独立验证、边界评估与 profile 四项回执均为 exit0；验证裂缝 IoU 68.0142%、F1 80.9625%、mIoU 82.9014%、边界 F1 45.0187%，是 seed42 单项结果。选中普通 best_model 的第 8000 步权重，选模 SHA 已保存，权重文件仅留本地。
- L13（OHEM CE）于 12:47 已 complete，18k 训练、独立验证、边界评估与 profile 四项回执均为 exit0；验证裂缝 IoU 68.3223%、F1 81.1803%、mIoU 83.0583%、边界 F1 45.8753%，是 seed42 单项结果。选中普通 best_model 的第 15000 步权重，选模 SHA 已保存，权重文件仅留本地。
- L14（Softmax Focal gamma 1.0）于 13:53 已 complete，18k 训练、独立验证、边界评估与 profile 四项回执均为 exit0；验证裂缝 IoU 67.7565%、F1 80.7796%、mIoU 82.7678%、边界 F1 45.0176%，是 seed42 单项结果。选中普通 best_model 的第 8000 步权重，选模 SHA 已保存，权重文件仅留本地。
- L15（Softmax Focal gamma 2.0）于 14:25 标记为 `needs_review`：队列写入 `heartbeat.json` 时发生 `PermissionError / WinError 5`。异常后日志仍有增长，但没有正式训练退出回执，性能留空；不能据此认定已恢复正常、训练完成或该损失精度差。原日志与[异常状态及 traceback](results/L15_SoftmaxFocal_status.json)保留，原因仍待排查。
- L16 的 smoke 已真实 exit0，最近状态为 training。汇总表在阶段完成时刷新，可能仍显示 pending；实际运行阶段应结合 `results/*_status.json`、heartbeat、日志和退出回执核对，不能将 smoke 通过视为正式训练完成。本次 GitHub 同步不修复或重启训练。
- L01–L04 已记录为 gradient_preflight_failed，保留梯度预检证据，性能不填数值。其他方案按各项回执更新。
- C0 是复用的合法 A0 + CE 冻结基线，不计作新训练。两个原替补配置在启动前停用并保存在 `provenance/prelaunch_disabled_reserves/`，不根据验证成绩增加替补。
- 新增 Dice 兼容恢复副本：已有 CPU 预检与独立源码，最近 heartbeat 为 `waiting_for_parent`、`gpu_started: false`。仅等待原队列结束，不代表 GPU 预检或正式训练成功；原失败记录保留。见[执行补充](refine-logs/PLAN_ADDENDUM_20261020.md)、[兼容审查](provenance/DICE_COMPAT_REVIEW.md)、[独立恢复计划](dice_compat_recovery_20261020/refine-logs/EXPERIMENT_PLAN.md)与[恢复心跳](dice_compat_recovery_20261020/results/heartbeat.json)。

## 阅读入口

- [完整执行计划](refine-logs/EXPERIMENT_PLAN.md)、[根目录计划](../../SegFormer损失函数优化20方案执行计划_20261019.md)。
- [进度表](refine-logs/RESULTS.md)、[CSV](results/loss_ablation.csv)、[机器可读结果](results/comparison.json)。
- [梯度预检](results/gradient_preflight.json)、[预检退出回执](results/gradient_preflight_exit.json)、[运行契约](provenance/RUNTIME.md)。
- [L05 训练回执](results/L05_CE_Lovasz_train_exit.json)、[独立验证](results/L05_CE_Lovasz_val.json)及同目录逐图 CSV。
- [L06 训练回执](results/L06_CE_Lovasz_train_exit.json)、[独立验证](results/L06_CE_Lovasz_val.json)及同目录逐图 CSV。
- [L07 训练回执](results/L07_CE_Lovasz_train_exit.json)、[独立验证](results/L07_CE_Lovasz_val.json)及同目录逐图 CSV。
- [L08 训练回执](results/L08_WeightedCE_train_exit.json)、[独立验证](results/L08_WeightedCE_val.json)、[逐图 CSV](results/L08_WeightedCE_val_per_image.csv)、[选模 SHA](checkpoints/L08_WeightedCE/selection.json)。
- [L09 训练回执](results/L09_WeightedCE_train_exit.json)、[独立验证](results/L09_WeightedCE_val.json)、[逐图 CSV](results/L09_WeightedCE_val_per_image.csv)、[选模 SHA](checkpoints/L09_WeightedCE/selection.json)。
- [L10 训练回执](results/L10_WeightedCE_train_exit.json)、[独立验证](results/L10_WeightedCE_val.json)、[逐图 CSV](results/L10_WeightedCE_val_per_image.csv)、[选模 SHA](checkpoints/L10_WeightedCE/selection.json)。
- [L11 训练回执](results/L11_TopKCE_train_exit.json)、[独立验证](results/L11_TopKCE_val.json)、[逐图 CSV](results/L11_TopKCE_val_per_image.csv)、[选模 SHA](checkpoints/L11_TopKCE/selection.json)。
- [L12 训练回执](results/L12_TopKCE_train_exit.json)、[独立验证](results/L12_TopKCE_val.json)、[逐图 CSV](results/L12_TopKCE_val_per_image.csv)、[选模 SHA](checkpoints/L12_TopKCE/selection.json)。
- [L13 训练回执](results/L13_OHEM_train_exit.json)、[独立验证](results/L13_OHEM_val.json)、[逐图 CSV](results/L13_OHEM_val_per_image.csv)、[选模 SHA](checkpoints/L13_OHEM/selection.json)。
- [L14 训练回执](results/L14_SoftmaxFocal_train_exit.json)、[独立验证](results/L14_SoftmaxFocal_val.json)、[逐图 CSV](results/L14_SoftmaxFocal_val_per_image.csv)、[选模 SHA](checkpoints/L14_SoftmaxFocal/selection.json)。
- [上一轮九项审计](provenance/previous_followup_audit.json)、[上一轮三 seed 完成说明](../segformer_aligned_wave7_20261018/followup_20261019/README.md)。

## 代码及结果边界

入口 `loss_queue.py`，损失目录 `loss_catalog.py`，独立预检 `gradient_preflight.py`；正式源码在 `src/PaddleSeg-2.10.0/`，配置在 `configs/`。复现前按运行契约核对实际 Python、数据、预训练及冻结控制权重路径。查看或同步进度无需重新启动队列。

本轮只使用 Crack500 train/val 筛选，失败和未完成项不填写性能。损失函数变更不改变模型参数量或推理结构，不能声称损失优化减少参数或加快推理。新损失的多 seed、公开模型与本轮最终 test 尚待后续核验。

GitHub 展示提交时的代码和结果快照；训练中的日志继续增长。原始数据、训练权重、本机进程锁和重复时间戳报告留在本地；配置、源码、日志、逐图指标、真实退出回执与选模 SHA 保留在仓库。

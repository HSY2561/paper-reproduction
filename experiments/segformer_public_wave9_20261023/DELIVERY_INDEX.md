# Wave8/Wave9交付索引

- 总报告：`../../SegFormer损失函数消融总报告_20261023.md`
- 损失合并表：`../segformer_loss_wave8_20261019/dice_compat_recovery_20261020/results/loss_ablation_consolidated.csv`
- 三种子摘要：`../segformer_loss_wave8_20261019/matched_seeds_20261022/results/three_seed_summary.json`
- 公开模型 JSON/CSV：`results/public_comparison.json` / `results/public_comparison.csv`
- 公开配置与真实日志：本目录 `configs/` 与 `logs/`
- 正式 checkpoint 只保留在原实验目录，未打入报告包。

## 2026-10-24 独立兼容环境补跑
- 兼容环境记录：[RUNTIME.md](compat_env_20261024/RUNTIME.md)
- GPU/CPU smoke 真实回执：[results](compat_env_20261024/results/)
- 最新对照 JSON/CSV：[public_comparison_20261024.json](compat_env_20261024/results/public_comparison_20261024.json) / [public_comparison_20261024.csv](compat_env_20261024/results/public_comparison_20261024.csv)
- 最新中文表：[六模型对照表_最新状态.md](../../实验结果汇总_20261024/六模型对照表_最新状态.md)

## 2026-10-25 Ubuntu SM120 训练完成快照

- [最新训练说明](sm120_ubuntu24_formal/README.md)：三项 18,000 步训练 exit=0，不等于独立评估、test 完成或科学门槛通过。
- [选模来源与权重 SHA256](sm120_ubuntu24_formal/results/training_snapshot_20261025.json)、[队列完成标记](sm120_ubuntu24_formal/results/queue_done.txt)。
- 原 `queue_status.json` 是 10 月 24 日早期运行快照；目前完成状态以三份 `*_exit.json` 与 `queue_done.txt` 为准。

# Ubuntu SM120 公共模型训练快照（2026-10-25）

只同步已经存在的证据；本次没有启动、恢复或修改实验。三份训练退出回执均为 exit=0，原始日志均包含 18000/18000；`results/queue_done.txt` 时间为 2026-10-25T02:28:21+08:00。

| 模型 | 训练步数 | exit | 日志显示的最佳验证 mIoU | 对应步数 |
|---|---:|---:|---:|---:|
| PSPNet-ResNet50 | 18000 | 0 | 0.8180 | 7000 |
| DeepLabV3P-ResNet50 | 18000 | 0 | 0.8270 | 16000 |
| Fast-SCNN | 18000 | 0 | 0.8019 | 15000 |

上述 mIoU 是日志四位小数显示值，不是新增独立验证的精确结果。训练结束不等于通过性能门槛或支持 SOTA。当前目录尚无三项独立验证、边界评估、统一 test 或完整推理测量回执，不将其填入此前的 test 比较表。旧 Windows GPU smoke 失败证据保留，不能用 Linux 成功记录抹除。

- [选模来源、权重 SHA256 与证据边界](results/training_snapshot_20261025.json)。SHA256 为本次读取本地 best_model 权重计算；并非当时冻结记录的补造。
- [PSPNet 回执](results/PSPNet_ResNet50_exit.json)、[DeepLabV3P 回执](results/DeepLabV3P_ResNet50_exit.json)、[Fast-SCNN 回执](results/FastSCNN_exit.json)。原始训练日志保留在 `logs/`。
- 原有 `results/queue_status.json` 是 2026-10-24 的运行中快照，已过时，不用它覆盖完成回执。
- 模型权重、数据、缓存和进程锁不上传；原始日志、配置、脚本与回执保留。

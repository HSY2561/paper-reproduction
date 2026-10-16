# SegFormer B0 精度与效率消融工程

本波次继承SegFormer-B0和Crack500固定协议，新增14种官方模块单项及2个MLP宽度控制。独立源码在src，历史工程保持可追溯。

GitHub 中的日志、heartbeat 和进程状态是提交时保存的快照，不代表查看时的实时运行状态。`queue.lock` 是本机防重入锁，不上传到仓库。

- 执行计划：[refine-logs/EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md)
- 自动更新的消融表：[refine-logs/EXPERIMENT_RESULTS.md](refine-logs/EXPERIMENT_RESULTS.md)
- 机器可读对照表：[results/ablation_table.csv](results/ablation_table.csv)
- 结构与性能总表：[results/comparison.json](results/comparison.json)
- 当前进程与进度：results/active_process.json、heartbeat.json、各status.json
- 每次真实退出回执：results/*_exit.json
- 旧结果选模、边界协议修正：[provenance/historical_corrections.json](provenance/historical_corrections.json)

未训练、失败、未核验的条目不填性能。smoke验证不算完成训练。队列只读train/val，不能使用test挑模型。参数压缩和额外模块必须同时对照width128母体和width256原始基线。最终候选验证通过后，才进入多种子和六个公开对照阶段。

运行方式见provenance/RUNTIME.md。不要在队列运行时另起第二个runner；queue.lock用于防重入。所有输出位于D盘本工程。后台进度监控已在当前任务建立。

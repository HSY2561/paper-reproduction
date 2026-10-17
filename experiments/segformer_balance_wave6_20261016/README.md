# SegFormer B0 精度与效率消融工程

本波次继承SegFormer-B0和Crack500固定协议，新增14种官方模块单项及2个MLP宽度控制。独立源码在src，历史工程保持可追溯。

GitHub 中的日志、heartbeat 和进程状态是提交时保存的快照，不代表查看时的实时运行状态。`queue.lock` 是本机防重入锁，不上传到仓库。

## 最新状态（2026-10-17）

14 个新训练组（12 种模块、2 个宽度控制）已完成 18k 训练、验证与选模核验；原始 B0 为冻结复用基线。Aligned 的单 seed 验证裂缝 IoU 为 67.852%，F1 为 80.847%，参数为 3.470434M；区域指标略高于 B0，但边界 F1 较低，速度与多种子结论仍待核验。

SE 最新退出回执为 17:49:47 超时（`exit_code=1`、`needs_review`），日志止于 3200/18000；PSP 为反向 smoke 失败。两者的性能保持空白。旧 heartbeat/active_process 文件是最后的运行记录，不表示进程仍在运行。

## 阅读与复现入口

- [结果分析与图示](refine-logs/RESULT_ANALYSIS.md)
- 执行计划：[refine-logs/EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md)
- 自动更新的消融表：[refine-logs/EXPERIMENT_RESULTS.md](refine-logs/EXPERIMENT_RESULTS.md)
- 机器可读对照表：[results/ablation_table.csv](results/ablation_table.csv)
- 结构与性能总表：[results/comparison.json](results/comparison.json)
- 当前进程与进度：results/active_process.json、heartbeat.json、各status.json
- 每次真实退出回执：results/*_exit.json
- 旧结果选模、边界协议修正：[provenance/historical_corrections.json](provenance/historical_corrections.json)

未训练、失败、未核验的条目不填性能。smoke验证不算完成训练。队列只读train/val，不能使用test挑模型。参数压缩和额外模块必须同时对照width128母体和width256原始基线。最终候选验证通过后，才进入多种子和六个公开对照阶段。

运行方式见 [provenance/RUNTIME.md](provenance/RUNTIME.md)。训练入口是 `experiment_queue.py --run`，代码位于 `src/PaddleSeg-2.10.0/`，逐项配置在 `configs/`。运行前按实际环境核对 Python、Crack500 数据、预训练/冻结基线权重和数据清单路径；本机配置不能直接作为其他机器的路径。不要在队列运行时另起第二个 runner。

仅刷新已有结果表可运行 `python experiment_queue.py --refresh-report`，不启动训练或推理；`summarize_results.py` 另会核验本地数据与权重并生成分析报告、CSV 和图。`checkpoints/*/selection.json` 保留选模及 SHA256，权重文件需在本地准备。

当前仓库只展示同名最新汇总，带时间戳的重复文档留在本地，初始预注册计划单独保留；原始日志、退出码、逐图指标及中断现场继续保留。实验 MANIFEST 是本地全过程登记，可能列出未上传的权重和旧文档。

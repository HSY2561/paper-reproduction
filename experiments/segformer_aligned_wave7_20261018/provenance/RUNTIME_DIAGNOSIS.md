# Wave6 SE 超时与 Wave7 运行条件诊断

核验日期：2026-10-18；资源快照为北京时间 10:24:01–10:24:46。本次仅只读检查日志、队列代码和系统资源；唯一新建文件为本说明。没有启动训练、GPU 探针、后台任务，没有结束进程、安装软件或修改计划任务。

## 结论

Wave6 SE 的最近失败有真实退出回执：训练子进程在 7200.474 秒后由 queue 超时终止，exit_code=1、timed_out=true；最后到 3200/18000，不能视为完整结果。SE 在此前同一源码/框架的运行中很快，最近这次是训练与验证一起显著变慢，优先怀疑运行资源和调度条件，而非 SE 模块固有复杂度。没有历史 CPU/GPU 时间序列，不能把今天的热点进程倒推为昨日超时的确定原因。

当前没有发现全局 python/pythonw 进程、指定旧 queue PowerShell 或历史一次性任务，GPU 查询也没有 Python 训练进程；但 CPU 已饱和且桌面显存占用较多，因此只能称“未发现训练在跑”，不能称机器资源空闲。

## 日志耗时：同一 SE 的前后对照最有说服力

| 运行 | 最后迭代 | 训练 batch_cost 中位数（秒/批） | reader_cost 中位数（秒） | 单次 348 张验证耗时 | 结果 |
|---|---:|---:|---:|---|---|
| SE 旧运行，13:30–13:34 | 2000 | 0.0838 | 0.000390 | 已完成两次，各 8 秒 | 在 2000 验证阶段中断，无真实退出码 |
| SE 最近运行，15:49–17:49 | 3200 | 1.1891 | 0.024055 | 121、222、337、312、198、277 秒；中位 249.5 秒 | 7200 秒超时，exit 1 |
| UAFMchannel | 18000 | 0.08665 | 0.000490 | 中位 9 秒，范围 7–10 秒，共 36 次 | exit 0，2393.93 秒 |
| GloRe | 18000 | 0.08925 | 0.000530 | 中位 9 秒，范围 8–10 秒，共 36 次 | exit 0，2460.64 秒 |
| W128 | 18000 | 0.07130 | 0.000490 | 中位 7 秒，范围 6–10 秒，共 36 次 | 完成 |

最近 SE 的 batch_cost 中位数为旧 SE 的约 **14.19 倍**，单次验证中位数约 **31.19 倍**。六次验证共耗时 **1467 秒（24.45 分钟）**；因此不是单次验证永久挂起，也不能把整段两小时都归到验证。1600→1700 的 100 步曾耗时约 628 秒，当段 batch_cost=5.0357 秒而 reader_cost=0.01742 秒，显示慢点不限于数据读取。

最近 SE 训练启动回执 15:49:47，首条环境信息 15:52:37，首个 100 iter 回执 15:55:16，初始化也有约 170 秒。按 3200 步/两小时粗略线性缩放，18k 约为 11.25 小时；这只是该次拥挤运行条件下的尺度，不能当作正常训练预算或稳定预测。

新旧 SE 日志均为 Python 3.12.10、PaddlePaddle 3.4.0、cuDNN 90.9、同一 NVCC 标识，CUDA enabled、单 GPU；GPU 内存峰值也相同：reserved=1108 MB、allocated=882 MB。日志没有 OOM、ResourceExhausted、Python traceback 或数值发散证据。最近 best/EMA checkpoint 留在 iter3000，不能替代 iter18000 完整结果。

注意 batch_cost 的计时范围：`paddleseg/core/train.py:289` 在每步 EMA 更新（332–333）之前记录 batch_cost；验证、保存及 EMA 等开销也不全计入它。因此应以完整墙钟回执估预算，不直接用 batch_cost×18000 作为任务总耗时。

证据路径（Wave6 根目录 `D:\论文复现\experiments\segformer_balance_wave6_20261016`）：

- `logs/M01_SE_train.log`
- `provenance/interruption_20261017_154341/logs/M01_SE_train.log`
- `results/M01_SE_train_exit.json`、`M01_SE_status.json`、`active_process.json`、`heartbeat.json`
- `logs/M14_UAFMchannel_train.log`、`logs/M10_GloRe_train.log`、`logs/W128_train.log`
- `results/M14_UAFMchannel_train_exit.json`、`results/M10_GloRe_train_exit.json`

## 当前全局资源证据

默认沙箱进程视图不可靠；以下来自获批的 require_escalated 只读 CIM / nvidia-smi 查询。首次资源命令最终返回码 1 来自指定旧任务不存在；各 GPU/CPU/内存/磁盘查询均有实测输出。后续显式处理任务不存在的复核命令退出码 0。

| 项目 | 实测 |
|---|---|
| GPU | RTX 5060，驱动 596.21，GPU 7%，内存利用率 4%，46°C，P3，约 23.98 W |
| GPU 内存 | 5410 / 8151 MiB 已用，名义剩余 2741 MiB |
| CPU | 20 逻辑 CPU；总利用率 100% |
| 内存 | 首次物理可用约 7.29 GiB / 31.71 GiB；复核可用 7419 MiB；commit 87% |
| 分页 | 复核瞬间 PagesInput/sec、PageReads/sec、PagesOutput/sec、PageWrites/sec 均为 0 |
| 系统盘 C: | 321408000000 字节剩余，约 299.3 GiB |
| 项目盘 D: | 12122005504 字节剩余，约 11.29 GiB |
| 物理磁盘瞬时 | 队列长度 0，读约 0.063 MB/s、写约 3.447 MB/s，未见当刻磁盘饱和 |
| 全局训练进程 | 没有 python/pythonw；没有指定 queue PowerShell；GPU 列表无 Python |
| 指定旧任务 | `Codex-SegFormer-wave6-once-20261017_154341` 不存在 |
| Wave6 文件状态 | queue.lock、scheduled_queue_exit_20261017.json、queue_done.json 均不存在；active_process/heartbeat 为昨日失败前的旧记录 |

10:24:46 CPU 热点（PerfProc 的百分比按单逻辑 CPU 为 100%，故需要除以 20 才是整机百分比）：

| 进程 | PID | PerfProc CPU% | 占整机 CPU 约 |
|---|---:|---:|---:|
| verge-mihomo | 18664 | 1537 | 76.85% |
| LenovoAppStore | 58000 | 104 | 5.20% |
| BaiduNetdiskUnite | 86644 | 90 | 4.50% |
| svchost | 44616 | 63 | 3.15% |
| ToDesk | 102116 | 31 | 1.55% |

这些是当前状态，无法证明昨日 15:49–17:49 也相同。没有操作这些进程。

显存 5.4 GiB 桌面占用**不能直接判定必然 OOM**：当前名义余 2741 MiB，大于历史 SE reserved 1108 MB，算术上仍有约 1.6 GiB 余量；但 WDDM 下占用动态变化、驱动/context/workspace 及其他应用增长都可能侵蚀余量。nvidia-smi 在本机 WDDM 进程列表的单进程显存为 N/A，不能精确归属全部 5410 MiB，也不能把名义剩余量当作可保证的训练容量。当前无历史 OOM 证据。

## 运行条件差异与优先级竞态

RUNTIME 记载一次性任务默认 BelowNormal，后来“仅把 queue 恢复到 Normal”。但 `active_process.json` 显示训练 child 已在 **15:49:47** 创建；说明中的 Normal 调整发生在约 15:50。队列 `run_command` 的 Popen 没有显式优先级标志。Windows 在没有指定优先级时，若创建者为 BelowNormal/Idle，子进程可继承该较低优先级；这是 [Microsoft CreateProcessW 文档](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw) 明确规定的行为。

因此存在“child 先以 BelowNormal 创建、随后只调父 queue”的可能；修改父进程不会自动遍历已存在子进程。**这只是有时间证据支持的候选原因，缺少当时实际 trainer 的 PriorityClass 回执，不能写成已证实根因。** CPU 饱和时，CPU 数据增强、Python 调度、算子提交、EMA 与验证都可能受影响，与训练/验证一起慢的现象相容。

下次应在创建 trainer 前确保 queue 为 Normal，必要时为 Windows 子进程显式设置 NORMAL_PRIORITY_CLASS，并查询实际执行训练的 python 的 PriorityClass。虚拟环境启动器 PID 与其实际 python 子进程可能不同，应核对整条父子链，不能只查 wrapper 或 queue。

## experiment_queue.py 约束审计

| 位置 | 已有行为 | 限制及恢复影响 |
|---|---|---|
| 155–174 | 每个子命令默认硬超时 7200 秒；超时 terminate、wait(60)，保存真实 timed_out 和退出码 | 包含该 trainer 的初始化、训练、全部普通/EMA 验证与保存；不是“无进展 2 小时”。只终止直接 child，没有保证整个后代树退出；wait 本身超时后没有兜底回执 |
| 175–181 | 每 20 秒更新心跳、迭代、日志大小、D 盘剩余 | 没有 CPU、RAM、commit、GPU 空闲/占用、进程优先级记录；磁盘值仅观察，运行中没有低空间停止逻辑 |
| 263–275 / 394 | 独立 val / boundary / profile 分别 600 秒，smoke 900 秒 | 机器拥挤时也可能误杀正常进展的验证；应和训练时限一起复核 |
| 364–370 | 固定 batch4、18k、seed42、FP32、workers0、EMA；每1000步评估，keep_checkpoint_max=1 | 共36次348张验证；目前不改 workers、batch、验证频率、精度或18k协议 |
| 405–406 | 每个待训练组开始前检查 D 盘至少4 GiB | 没有 RAM/GPU 内存限制；没有 C 盘/pagefile 余量检查；不是“全程硬保留4 GiB” |
| 408–418 | 已有非零训练退出回执时直接跳过该组；无回执但已有训练目录则拒绝盲重启 | 直接重新运行 --run 不会自动重试本次 SE；需要先保留失败证据，再采用明确的新运行目录/恢复流程 |
| 419–428 | 要求 iter18000 checkpoint、最终日志和冻结验证一致后才标 complete | 正确防止3200或3000的中途结果混入完整表 |
| 373–376 / 441–443 | 独占创建 queue.lock，finally 删除 | 当前无锁不等于成功；外部强停可能没有 finally/父退出 JSON，必须结合全局进程和真实回执 |

历史任务的注册回执设置外层 `execution_limit=PT2H`，而 trainer 自身也允许7200秒，另有约6分钟预检/启动及后处理；外层2小时没有覆盖完整工作流的预算余量。本次已确认由 queue 写出 child 超时回执，但缺失 scheduled_queue_exit 和 queue_done，不能假设父包装流程也正常完成。

## 可执行的恢复建议

1. 先保留 Wave6 的14个完整组和SE失败现场，新工作在 Wave7 单独记录；不把 partial checkpoint 的验证分数当完整18k结果。不因旧 active_process/heartbeat 继续存在而判断仍在运行。
2. 先保证实际 trainer 自创建起为 Normal；记录 wrapper、queue、venv启动器及实际 python 的 PID/PPID/PriorityClass。保持原 batch、workers0、seed、18k、FP32、EMA 和验证策略。不要把优先级设为 High。
3. 放宽时限并让外层包装覆盖所有串行组、数据哈希、初始化、验证、保存和退出回执。可先按每1000步墙钟耗时及两次完整val测算 `18×训练块 + 36×单次val + 预后处理`，再留至少1.5倍余量；未复测前，昨日慢状态外推约11.25小时，2小时显然不足。若必须沿用这种慢状态，可考虑单组16小时的保守上限，外层还需另计全部剩余组与收尾。时限是故障上限，不是目标运行时间。
4. 本轮四模型同进程、batch1、交错顺序的基准可以在相同拥挤条件下比较相对差异；应预热各模型、统一输入/精度/同步方法、交错或反转顺序并重复，报告中位数和波动，同时记录CPU饱和/显存占用。不要把这些绝对耗时称为独占硬件性能，约60MB参数也不能代表全部激活/context/workspace显存。
5. CPU线程 cap 暂不默认设置。OMP/MKL/Paddle intraop cap 会改变并行调度，影响幅度应单独验证；若未来采用，应在导入框架前对所有比较模型使用同一设置，检查相同权重/输入/seed/FP32下的 logits、loss、梯度及一次优化更新是否在预先记录容差内一致，并用同样负载/优先级做耗时对照。不能用改 workers、batch 或精度代替这个验证。
6. 恢复前再读一次 CPU/GPU/RAM/D盘快照；开始后检查首100iter、首1000iter和两次val的墙钟进展及实际PriorityClass。日志持续推进且数值正常时，不因旧工具session丢失盲目重启。父/子最终都应有真实退出回执。

## 本次文件变更

| 文件 | 类型 | 内容 |
|---|---|---|
| `provenance/RUNTIME_DIAGNOSIS.md`（Wave7） | 新建 | 本只读审计报告；没有添加或修改运行时插桩 |


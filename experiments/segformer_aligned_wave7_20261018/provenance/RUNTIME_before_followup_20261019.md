# Wave7 本地运行契约

当前工程 `D:\论文复现\experiments\segformer_aligned_wave7_20261018`，Python `C:\venv_bisenet\Scripts\python.exe`，独立PaddleSeg源码 `src\PaddleSeg-2.10.0`；子进程环境通过PYTHONPATH明确选择该源码。RTX5060、FP32、Paddle3.4，batch4、seed42、18k。数据只读 `C:\Users\user\algolab_data`，仅train/val。

## 实际运行与证据

一次性Windows任务 `Codex-SegFormer-wave7-once-20261018`，实际执行 `run_workflow.ps1` → `workflow.py` → `experiment_queue.py`（兼容入口）→ `aligned_queue.py`。后台workflow自启动起设Normal，GPU子进程显式Normal；任务运行上限7天、每组训练12小时。30分钟日志不更新或盘余不足4GiB时保留回执并停止；不以改变训练预算解决性能问题。

启动器第一次因ctypes Windows进程句柄类型声明错误在训练前退出；真实exit1及错误保存在 `provenance/launcher_failure_20261018`。句柄声明修复后重新启动同一任务，未新建重复任务。旧波SE已完整18k/exit0，官方验证、独立边界验证及profile全部exit0，11:04工作流进入等待审阅阶段。

15:57:25源/config哈希审阅放行；15:57:51进入wave7，runner85160；首项A01_GloRe smoke在15:57:57进入子进程97832。PID是这一时刻的历史证据，当前状态应重新读results/heartbeat.json、active_process.json、真实exit和日志。

## 独立检查

- `results/constructor_checks.json`：24项CPU构造通过，仅用于接口与参数检查，非反向/训练成功。
- `provenance/A0_source_equivalence.json`：原Aligned冻结SHA 000b90f9668e35d0a7cae0fd535134802c5a64a7ced2eab05f869cef726cd5ea，在新源码下重新推理得到完全相同混淆矩阵。
- `results/A0_source_recheck_exit.json`：真实exit0；评估n348，指标与复用A0相同。
- `results/spatial_audit_exit.json`：真实exit0；B0/Aligned/GloRe三组各348图重新推理，全部CM与旧冻结结果完全相同。
- `provenance/source_hashes.json`、`config_hashes.json`与`RELEASE.json`：正式训练前的源码/配置哈希；模型公式只能来自冻结官方模块，运行中不改源代码。
- `provenance/reused_controls.json`：B0/W128/A0/GloRe冻结checkpoint、来源及SHA。

正常训练只串行运行一项GPU任务。首次同进程交错速度测量已经exit0，但轮间比值仍显著波动；只能称共享资源条件下的配对测量，未证明稳定加速。候选完整速度门槛留空。原资源热点诊断见RUNTIME_DIAGNOSIS.md；现在的负载需重新查询，不能把此前快照当成长期状态。

## 后续核验

每项依次smoke20iter → 正式18k → 普通/EMA最高精确val选模 → 冻结SHA → 官方val.py → 全348混淆矩阵与精确零外延2px边界验证 → Params和临时延迟。仅当真实exit0、最终iter18000 checkpoint和独立指标一致时状态为complete。

外层任务或旧exec session消失不能证明GPU进程消失；默认沙箱可能看不到其它会话进程。恢复前须检查全局实际进程、锁、任务状态和最新日志，防止重复。任何退出码没有回执时保留未知。

当前所有24项只是在Aligned母体上增加第二模块的登记；PSP暂缓，23项正式尝试。没有训练就不填写性能，没有统一稳定测量就不宣称加速，没有冻结统一test和多seed就不宣称最终结论。

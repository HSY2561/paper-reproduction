# 最新任务指向：Wave8损失函数消融（覆盖下面历史启动状态）

2026-10-19 23:11：用户要求结合最新规划开始约20方案。可读网页用户方向为损失优化；助手正文未取回。主波22组、后续9项训练均已结束，逐阶段exit0/18k/冻结/独立验证已审计。后续wrapper最终退出回执缺失、系统中止0xC000013A，不能写wrapper exit0；旧任务注册已清理，XML及审计保留。不要重新启动主波或9项旧队列。

当前唯一新活动工程：`D:/论文复现/experiments/segformer_loss_wave8_20261019`。先读其`refine-logs/EXPERIMENT_PLAN.md`、`provenance/RUNTIME.md`、`RELEASE.json`、`protocol.json`及实时heartbeat/active_process/status/exit。新任务`Codex-SegFormer-loss-wave8-once-20261019`调用`run_loss_queue.ps1`/`loss_queue.py --run`；先控制复现和梯度预检，再20项串行smoke/18k。当前是否已进入正式训练以实际日志为准，注册不等于开始。

A21三seedIoU均值增益0.0019078953<原0.002门槛，A10均值退化；固定A0=width128＋Aligned，采用20种官方损失配置，不用Polarized/AFNB做新母体。损失不减参、不加速推理，Precision/Recall/BF1完整门槛不事后放宽。旧速度不稳定且中位数更慢，不声称加速或SOTA。

仅train/val，不访问test或自建数据。Python C:/venv_bisenet/Scripts/python.exe，PYTHONPATH必须新波src/PaddleSeg-2.10.0。Normal优先级，每stage真实exit，源/配置/数据SHA守卫，盘余<4GiB、日志30min不变、NaN/断梯度/非零退出查明后再恢复，不能重复启动。先前9项成功smoke重复副本按SHA清单回收0.724GiB，正式/最佳/冻结权重和原始数据未删。

健康无实质变化保持安静；完成/重要结果/故障通知。本轮20项结束后冻结最多top2并多seed复核，再继续六原始公开模型及统一test，最终报告表图CSV、复现索引和报告压缩包发送给用户后才停止监控。最终交付合同沿用主波refine-logs/FINAL_DELIVERY.md。

---

# 10月19日当前活动状态（覆盖下面历史PID和旧队列）

> 20:19 后更新：followup_20261019 的九项新训练均已完成，见[完成说明](../followup_20261019/README.md)和该目录 `results/queue_done.json`。下文正在训练、活动任务和 PID 的表述保留为历史，不能用来判断当前运行；不应重复启动主波或已完成复核队列。

主波已05:18正常结束，22完整组已经独立审计，错误0；不要恢复或重训主波。当前活动工程是 followup_20261019，入口followup_queue.py，任务Codex-SegFormer-wave7-followup-once-20261019。请首先读 followup_20261019/provenance/RUNTIME.md、refine-logs/PLAN.md，以及该子目录results下的heartbeat、真实exit、各status。

已冻结A21_Polarized/A10_AFNB供复核；9项新运行是AFNB单独seed42和B0/A0/A21/A10各seed43/44。9:09速度复核exit0但不稳定且当前中位数更慢，不得声称加速；9:09:39进入AFNB单独正式训练，9:15:40已3200/18000。后续请读取实时日志，勿以这些历史时间冒充当前进度。

主波阶段报告在根目录SegFormer_Aligned模块筛选阶段报告_20261019.md；图和CSV已更新。一次性主任务exit0已仅清理其注册，新任务在运行。原4488数据未变，正式权重/失败证据保留，仅已完成smoke重复副本按哈希清单回收1.77GiB。

监控工具阶段说明更新调用未及时取得回执；当前真实活动队列以此段和子目录RUNTIME为准，勿因旧主波heartbeat停更而重复启动。最后仍要完成三seed/归因/速度、六个原始公开模型及冻结统一test、完整报告交付，再停止监控。

---

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

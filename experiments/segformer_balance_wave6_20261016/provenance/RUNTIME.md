# 本地运行契约

> 2026-10-17 最新退出状态：SE 训练在 17:49:47 达到 7200 秒超时，`results/M01_SE_train_exit.json` 为 exit_code=1、timed_out=true，`M01_SE_status.json` 为 needs_review。最后日志为 3200/18000，性能未完成。下文15:50及更早的进程/PID是历史记录，不能表示当前仍在运行。本次 GitHub 同步只整理结果与说明，没有重启训练或修改系统任务。

复用 Python `C:\venv_bisenet\Scripts\python.exe`，PaddlePaddle 3.4.0，RTX5060 8GB，FP32。不安装新框架。数据只读 `C:\Users\user\algolab_data`；新文件仅写本波次目录。

启动前原样执行下面的 GPU 和源码见证。stdout 必须有 `SOURCE_WITNESS` 及 `GPU_WITNESS`，返回码必须是0。该检查只验证运行环境，不替代每个候选的正式训练入口 smoke。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='D:\论文复现\experiments\segformer_balance_wave6_20261016\src\PaddleSeg-2.10.0'
& 'C:\venv_bisenet\Scripts\python.exe' -c "import paddle,paddleseg; paddle.seed(42); paddle.set_device('gpu:0'); x=paddle.randn([4,32,25,25]); m=paddle.nn.Conv2D(32,32,3,padding=1); y=m(x); y.square().mean().backward(); print('SOURCE_WITNESS',paddleseg.__file__); print('GPU_WITNESS',paddle.__version__,list(y.shape),float(y.mean()),paddle.device.cuda.get_device_name())"
```

正式队列：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\venv_bisenet\Scripts\python.exe' 'D:\论文复现\experiments\segformer_balance_wave6_20261016\experiment_queue.py' --run
```

队列先复核原B0固定权重和数据哈希，再做全部smoke，然后串行18k训练、验证选模、冻结、独立val.py和精确边界评估、参数及速度测量。`queue.lock` 防止重复运行。子进程使用新的PYTHONPATH，覆盖旧editable安装；日志与真实退出码逐项保存。系统盘的原数据、框架和权重只读使用。

### 2026-10-17 运行恢复

旧队列最后回执为04:28:45 PSA iter9700，09:02检测时已超过4小时无更新，原终端session不可用。旧进程退出码未知，不伪造失败码或成功码。中断现场完整移入 `provenance/interruption_20261017_090416`；完成的7组不改动。

官方续训未恢复EMA和数据随机状态，因此只对未完成的PSA按原seed和18k预算重跑。隐藏启动工具的PID回执不足以证明持久运行；实际恢复使用 `exec_command` 持久session **54465**，runner **102120**，child **80492**。新训练09:04:55启动，09:05:17记录100/18000。用heartbeat及日志连续进展、最终exit JSON和checkpoint核验健康。没有第二个活动队列。

### 2026-10-17 12:08 最新运行状态（覆盖旧session/PID）

12个单项已经complete，含10模块和2宽度控制。旧session54465在11:58后再次失效，GloRe中断于1800/18000且没有退出回执；获批的全局进程查询也未找到旧runner102120或child78952，退出码留未知。现场保存在 `provenance/interruption_20261017_120620`，12个完成项保持原样。

已用5秒自动结束的探针验证独立进程能在父命令结束后继续写入回执；`results/process_persistence_probe_20261017.json`记录survived_parent。随后使用获批的Python `CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` 独立运行同一queue，stdin为DEVNULL、输出写独立项目日志，不依赖工具session。12:07:24新runner为81416、GloRe child为92144；12:08:03日志200/18000。真实启动与输出路径见 `results/independent_queue_launch_20261017_120715.json` 和 `logs/queue_independent_20261017_120715.log`。只重跑未完成GloRe，之后继续原计划UAFM通道和SE；数据4488文件重新核验无变化，未访问test。

默认shell曾在创建进程前返回路径不存在，未进入shell；显式指定已验证存在的 `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` 且login=false后成功执行。Windows PowerShell管道给Python传入含中文路径代码时显式设置UTF8 OutputEncoding/PYTHONIOENCODING。后续监控不因旧exec session消失而重启：先核对独立queue的heartbeat、日志、全局进程和退出码，避免重复启动。

### 2026-10-17 15:50 最新恢复（覆盖先前PID与启动方式）

GloRe和UAFM通道组已完成，累计14个完整组（12模块+2width）。旧独立进程在13:34又中断于SE的2000iter后验证阶段，没有exit回执；原因未知，不能仅凭breakaway探针认定此后永不停止。全部现场保留在 `provenance/interruption_20261017_154341`，只对SE重新跑满相同18k。

已获准注册**一次性**Windows任务 `Codex-SegFormer-wave6-once-20261017_154341`，执行现有 `run_queue_once_20261017.ps1`，实际于15:43:55运行；时限2小时、过期后1小时自动删除。注册回执在 `results/scheduled_queue_registration_20261017.json`，父日志 `logs/scheduled_queue_20261017.log`，父进程结束的真实回执将写 `results/scheduled_queue_exit_20261017.json`。任务Running只证明调度器动作未退出，不等于GPU已开始。

新队列锁15:45:12记录runner78068，虚拟环境启动器86844、PowerShell87884。15:50全局查询确认队列进程在运行且有154MB以上I/O读取，仍在train/val数据哈希预检，尚无新GPU训练回执；旧13:34heartbeat不能冒充新进展。任务默认BelowNormal，已仅把queue恢复到与普通训练同样的Normal优先级，未设High。不要在预检过程中重复启动；超过30分钟无进展再核实日志/CPU/I/O及任务错误。

此环境默认沙箱看不到计划任务服务和其它会话进程；获批全局查询可见Schedule服务Running及上述任务/进程。因此默认Get-Process无输出不能单独证明进程消失。后续需先核对已注册任务和全局实际进程，再处理旧锁，防止重复。正常完成后仅清理注册回执中这个任务，或由既有过期机制删除，不触碰其它系统任务。

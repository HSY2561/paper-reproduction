# W11 执行记录

2026-10-25，Asia/Shanghai。用户已明确授权“开始训练”，执行既有 W11-v1 四格及三seed计划，门槛不变。

## 环境与范围

- 活动工程：`${PROJECT_ROOT}\experiments\segformer_complementarity_wave11_20261025`。
- Python：`${WINDOWS_ENV}\Scripts\python.exe`，Python3.12.10、Paddle3.4.0、CUDA12.9、cuDNN9.9.0，RTX5060 8GiB；复用现有环境，未安装或升级依赖。
- 每次启动的PYTHONPATH必须指向本轮 `src\PaddleSeg-2.10.0`；kernel_witness已记录实际import位置。
- 训练只读Crack500 train/val。4488文件和两个清单已复核，`data_preflight.json`无不一致；禁止筛选读取test/自建集。
- 18:42全局只读核实无python训练进程。GPU桌面占用约2.2GiB、利用率约3%，不终止用户其他应用；不能据显存非零断言已有训练。
- 旧loss两项计划任务处于Ready，未启动本轮或被清理；只登记/清理本轮自己的任务。

## 已完成检查

1. `kernel_witness.json`：seed42 GPU FP32 Conv前后向、有限梯度、真实命令exit0。
2. `data_preflight.json`：4488源文件及划分清单全部SHA一致，命令exit0；未读test。
3. `model_preflight.json`：新完整模型CPU参数/尺寸/反向/坐标/关闭M2一致性通过。CPU通过不等于GPU smoke通过。
4. 原SegFormer/mix_transformer/sfnet模型文件保持不改，新隔离类实现M1/M2与压缩构造器。

## 接下来的放行条件

派生ImageNet骨干、12份共享完整初始化及SHA核验 → 冻结本轮release → 新源A0全348精确复现 → 4组GPU20iter smoke → 12组完整18k（每seed四格），逐组真实exit/冻结普通与EMA精确val最高checkpoint/独立官方和像素边界评价。

共享数据随机性通过官方训练入口构造完成后的重设及样本SHA见证控制，保留增强公式/分布；不能将模块初始化随机数消耗误当相同数据流。

任何中断保留现场与未知退出码，不能拼接日志冒充完整运行。保护只针对本轮拥有的child：30min日志停滞、单阶段12h、磁盘<4GiB需停并登记。不得删除原数据或正式冻结checkpoint。

操作状态以 `results/workflow_state.json`、`heartbeat.json`、`active_process.json`和各真实exit为准；`protocol.json`中的planned字段是原预登记快照，不是实时进度。不得以Windows任务Running冒充GPU正式训练。

## 预计时间

近期真实18k每组26–29min；暂按25–35min估计。首批四组训练与评价约2–3h，全部12组纯训练约5–7GPU小时；包括当前实现前检、评价、速度和报告，顺利约8–15h，按真实进度更新。性能结果目前为空。

### 正式启动前的配对控制补充

复核发现旧A0随机头及训练RNG不能视为W11严格配对。详见 `refine-logs/EXECUTION_ADDENDUM_20261025.md`：12四格不变，其后新增3个同入口、同E0头初始化的原架构R0，总15次18k。旧控制不覆盖，最终严格门槛使用新R0；顺利总预算修订为约9–17小时。新增控制初始权重已CPU严格加载核验，但尚无正式结果。

## 固定24图可视化准备（2026-10-25 20:32核验）

`qualitative_w11.py`、`QUALITATIVE_API.md`与独立`qualitative_release.json`已准备并独立审查。release SHA333c1c065c0d04a953a5e5a3d244bb0982837fd4ec751a43557c5af9d212cb21。只做过CPU编译、小数组PNG roundtrip和训练期间`--render`必须拒绝的门控验收，故该预检exit1是预期拒绝，不是训练失败；未GPU/未实际渲染。

不要在训练或效率测量期间运行渲染。15训练及自动后处理、全部相应wrapper/child真实exit0且锁/竞争检查通过后，由监控调用同一Python执行 `qualitative_w11.py --render`。它用官方infer API、五组冻结seed42、已有fixed24清单，匹配原348逐图计数/强制预测和GT SHA，并对齐原交付bundle_manifest；4488数据前后SHA核验、独立英文输出目录/manifest/补充ZIP。旧报告包只核SHA、不覆盖。完整API与独立review见provenance/qualitative_independent_review.json。
## 后处理桌面PID刷新（2026-10-26 09:02）

15个正式训练、独立验证及主wrapper已02:03真实exit0。原v2后处理等待器没有启动GPU；微信/登录界面/手写输入等桌面应用在夜间重启产生5个新PID，严格旧PID基线将其视为未知，持续延后测量。未关闭任何用户应用，也没有新的训练失败。

已保存原阻塞状态、只读Win32/Toolhelp/XML身份及原release；通过STOP_REQUEST让自有v2等待器正常抛错退出，真实exit1保留，不能写成0。停止标记随后移入provenance。只隔离新增v3后处理/可视化入口与当前43个特定桌面PID基线，仍要求同PID/类型/名称/可见路径和低稳定背景窗口；纯C、Python及任何未登记C+G继续拒绝。训练源码、三seed汇总、数据、选模和门槛均不改。

### v4：已核桌面二进制换PID与wrapper回执

v3仍未GPU；已核WindowsTerminal同一二进制又产生新PID65360，固定PID策略再度误阻塞。v4只允许已冻结的非空绝对exe路径、basename和当前binary SHA全部一致的已核C+G更换PID；纯C/Python/未登记或改SHA/新空路径仍拒绝，低稳定负载窗及统计/训练门槛不改。仅31条已核二进制身份，不是放行任意桌面应用。原v3和各版本证据保留。

v3通过自有STOP_REQUEST退出，wrapper真实receipt中的exit_code为null（原PowerShell native stderr异常导致赋值前退出），故记录未知，不能写成exit0/1。v4隔离wrapper改成Start-Process隐藏启动自有Python、WaitForExit后读取实际ExitCode并保留child PID；不会结束其他应用。15个正式训练的exit0回执完全未改。

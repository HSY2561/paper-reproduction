# 最新任务指向：Wave8损失函数消融（覆盖下面历史启动状态）

2026-10-19 23:11：用户要求结合最新规划开始约20方案。可读网页用户方向为损失优化；助手正文未取回。主波22组、后续9项训练均已结束，逐阶段exit0/18k/冻结/独立验证已审计。后续wrapper最终退出回执缺失、系统中止0xC000013A，不能写wrapper exit0；旧任务注册已清理，XML及审计保留。不要重新启动主波或9项旧队列。

当前唯一新活动工程：`D:/论文复现/experiments/segformer_loss_wave8_20261019`。先读其`refine-logs/EXPERIMENT_PLAN.md`、`provenance/RUNTIME.md`、`RELEASE.json`、`protocol.json`及实时heartbeat/active_process/status/exit。新任务`Codex-SegFormer-loss-wave8-once-20261019`调用`run_loss_queue.ps1`/`loss_queue.py --run`；先控制复现和梯度预检，再20项串行smoke/18k。当前是否已进入正式训练以实际日志为准，注册不等于开始。

A21三seedIoU均值增益0.0019078953<原0.002门槛，A10均值退化；固定A0=width128＋Aligned，采用20种官方损失配置，不用Polarized/AFNB做新母体。损失不减参、不加速推理，Precision/Recall/BF1完整门槛不事后放宽。旧速度不稳定且中位数更慢，不声称加速或SOTA。

仅train/val，不访问test或自建数据。Python C:/venv_bisenet/Scripts/python.exe，PYTHONPATH必须新波src/PaddleSeg-2.10.0。Normal优先级，每stage真实exit，源/配置/数据SHA守卫，盘余<4GiB、日志30min不变、NaN/断梯度/非零退出查明后再恢复，不能重复启动。先前9项成功smoke重复副本按SHA清单回收0.724GiB，正式/最佳/冻结权重和原始数据未删。

健康无实质变化保持安静；完成/重要结果/故障通知。本轮20项结束后冻结最多top2并多seed复核，再继续六原始公开模型及统一test，最终报告表图CSV、复现索引和报告压缩包发送给用户后才停止监控。最终交付合同沿用主波refine-logs/FINAL_DELIVERY.md。

---

# 10月19日候选复核运行契约

> 最新完成状态（20:19）：`../results/queue_done.json` 记录全部 9 项 complete，四模型三 seed 已齐备。以下启动、PID、heartbeat 为历史，不代表仍在运行；无需重启已完成队列。下一步仍按原计划审核预登记门槛与速度，再处理六个原始模型及本轮统一 test，详见 [完成说明](../README.md)。

Python C:\venv_bisenet\Scripts\python.exe；PYTHONPATH必须为主wave7的src/PaddleSeg-2.10.0。正式源码与数据协议沿用主波已冻结哈希。活动入口followup_queue.py --run，profile子命令也由该文件处理。

一次性任务Codex-SegFormer-wave7-followup-once-20261019执行run_followup.ps1。上一轮主任务正常exit0，已仅移除其注册，保留XML与真实退出证据。新任务第一次在训练前因Windows默认文本编码损坏元数据路径被SHA预检拦截，真实exit1保存在provenance/launch_path_encoding_failure_20261019。重建实际路径并核对原SHA后9:07:56重启同一任务；候选ID、门槛、数据和seed预算不变。以后所有元数据read_text必须显式encoding='utf-8'或调用read_json。

候选速度于9:09:11真实exit0；P90/P10和配对IQR不稳定，中位数也慢于B0，没有加速证据。9:09:39进入X_AFNB_s42正式18k，日志9:09:57 iter100。PID34044是当时runner，98524是当时训练启动器，当前请以heartbeat、actual_process、日志和真实exit为准，不把旧PID当持续运行证据。

队列9项：AFNB单独seed42；B0/A0/A21/A10各seed43/44。seed42对照复用主波；旧波M02_PSA复用为Polarized单独对照。只用train/val，不读test或自建数据；每组仍18k/FP32/batch4/seed登记值/相同ImageNet共同初始化。普通EMA精确val选模、冻结SHA、官方val和独立2px边界核验。

首轮22项已完成，DAPPM与PSP负证据保留。原4488数据哈希未变。为后续磁盘预算，已按逐文件SHA清理1.77GiB已完成smoke的重复产物，正式训练/最佳/冻结权重与失败日志不动。主波provenance/smoke_cleanup_*保存路径与哈希清单。

仅阶段一结束，全部任务未完成。完成9项后审核三seed均值与差值、速度代价，再推进六个原始公开模型缺项与统一test、总报告/图/交付压缩包。无需重复跑22项，不应因主波旧heartbeat不更新而重启主队列。

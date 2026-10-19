# 10月19日候选复核运行契约

Python C:\venv_bisenet\Scripts\python.exe；PYTHONPATH必须为主wave7的src/PaddleSeg-2.10.0。正式源码与数据协议沿用主波已冻结哈希。活动入口followup_queue.py --run，profile子命令也由该文件处理。

一次性任务Codex-SegFormer-wave7-followup-once-20261019执行run_followup.ps1。上一轮主任务正常exit0，已仅移除其注册，保留XML与真实退出证据。新任务第一次在训练前因Windows默认文本编码损坏元数据路径被SHA预检拦截，真实exit1保存在provenance/launch_path_encoding_failure_20261019。重建实际路径并核对原SHA后9:07:56重启同一任务；候选ID、门槛、数据和seed预算不变。以后所有元数据read_text必须显式encoding='utf-8'或调用read_json。

候选速度于9:09:11真实exit0；P90/P10和配对IQR不稳定，中位数也慢于B0，没有加速证据。9:09:39进入X_AFNB_s42正式18k，日志9:09:57 iter100。PID34044是当时runner，98524是当时训练启动器，当前请以heartbeat、actual_process、日志和真实exit为准，不把旧PID当持续运行证据。

队列9项：AFNB单独seed42；B0/A0/A21/A10各seed43/44。seed42对照复用主波；旧波M02_PSA复用为Polarized单独对照。只用train/val，不读test或自建数据；每组仍18k/FP32/batch4/seed登记值/相同ImageNet共同初始化。普通EMA精确val选模、冻结SHA、官方val和独立2px边界核验。

首轮22项已完成，DAPPM与PSP负证据保留。原4488数据哈希未变。为后续磁盘预算，已按逐文件SHA清理1.77GiB已完成smoke的重复产物，正式训练/最佳/冻结权重与失败日志不动。主波provenance/smoke_cleanup_*保存路径与哈希清单。

仅阶段一结束，全部任务未完成。完成9项后审核三seed均值与差值、速度代价，再推进六个原始公开模型缺项与统一test、总报告/图/交付压缩包。无需重复跑22项，不应因主波旧heartbeat不更新而重启主队列。

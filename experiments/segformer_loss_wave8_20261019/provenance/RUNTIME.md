# 2026-10-20 执行补充：范围确认及Dice标签存储兼容修复

用户已明确回复“固定Aligned，只优化损失”。原20方案、权重、门槛和训练预算保持不变。

新源码C0已真实exit0重现原混淆矩阵，4488源文件SHA均一致。原反向预检15/20通过；L01–04及L18逐组件正常，但MixedLoss共享标签在Dice原地改写ignore时造成CE反向Tensor版本冲突。完整错误在原results/gradient_preflight.json，未删除/覆盖，未把这些方案计作训练成功。

独立兼容副本dice_compat_recovery_20261020仅把官方mixed_loss.py内调用改为loss(logits, labels.clone())。独立审查与5项CPU预检均已通过，前向值不变、基础标签不被改写、反向有限非零。该副作用隔离不是新损失公式；主源码和15合法组不改不重训。CPU通过不代表GPU或18k完成。

恢复队列先等待父队列queue_done、锁移除、最后runner/child实际退出，期间不分配GPU。之后用独立快照重现C0，再GPU反向预检、smoke20及原5项完整18k，冻结/独立val门槛不变。正式合并表注明原失败和compat_clone恢复来源，不把5项另计新方案。

以实际heartbeat、exit、最终18k和SHA判断进度；外层计划任务Running不代表训练。后续最多top2冻结后多seed，再原六模型对照及统一test；不访问自建数据。

---

# Wave8 当前执行契约

2026-10-19：旧结构及9项复核训练已完成，新轮固定A0=SegFormer-B0 width128＋Aligned，按EXPERIMENT_PLAN执行20种官方loss配置。先C0同源码重现/数据SHA/梯度预检，再串行smoke20→正式18k→冻结→独立val。不得重启旧队列，不读test或自建数据。

Python C:\venv_bisenet\Scripts\python.exe；PYTHONPATH指向本工程src/PaddleSeg-2.10.0，覆盖旧editable；源未修改。wrapper入口run_loss_queue.ps1，调用loss_queue.py --run，Normal优先级。实际运行必须读取heartbeat/active_process/日志变化/真实exit，任务Running不是训练证据。

仅复用旧C0合法冻结，所有新loss从共同ImageNet初始化。普通/EMA精确val mIoU最大值冻结SHA，独立2px边界评估，参数应恒为3470434。损失不改变参数量或推理图；记录训练成本，不声称loss加速。

旧9项训练均有真实exit0及独立验证；旧wrapper最终回执缺失，任务系统中止码0xC000013A，不将它记exit0。详见provenance/previous_followup_audit.json。

主门槛沿用A0：IoU+.002、F1+.001、mIoU不降、P/R/BF1>=-.005，最多top2冻结后补43/44。20主项无按结果触发替补。计划结束后继续六个原始对照及冻结统一test、最终报告图和交付包，不因本轮完成就误称整个项目已完成。

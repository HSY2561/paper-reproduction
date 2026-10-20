# 2026-10-20 执行补充：范围确认及Dice标签存储兼容修复

用户已明确回复“固定Aligned，只优化损失”。原20方案、权重、门槛和训练预算保持不变。

新源码C0已真实exit0重现原混淆矩阵，4488源文件SHA均一致。原反向预检15/20通过；L01–04及L18逐组件正常，但MixedLoss共享标签在Dice原地改写ignore时造成CE反向Tensor版本冲突。完整错误在原results/gradient_preflight.json，未删除/覆盖，未把这些方案计作训练成功。

独立兼容副本dice_compat_recovery_20261020仅把官方mixed_loss.py内调用改为loss(logits, labels.clone())。独立审查与5项CPU预检均已通过，前向值不变、基础标签不被改写、反向有限非零。该副作用隔离不是新损失公式；主源码和15合法组不改不重训。CPU通过不代表GPU或18k完成。

恢复队列先等待父队列queue_done、锁移除、最后runner/child实际退出，期间不分配GPU。之后用独立快照重现C0，再GPU反向预检、smoke20及原5项完整18k，冻结/独立val门槛不变。正式合并表注明原失败和compat_clone恢复来源，不把5项另计新方案。

以实际heartbeat、exit、最终18k和SHA判断进度；外层计划任务Running不代表训练。后续最多top2冻结后多seed，再原六模型对照及统一test；不访问自建数据。

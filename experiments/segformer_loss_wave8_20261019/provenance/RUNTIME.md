# Wave8 当前执行契约

2026-10-19：旧结构及9项复核训练已完成，新轮固定A0=SegFormer-B0 width128＋Aligned，按EXPERIMENT_PLAN执行20种官方loss配置。先C0同源码重现/数据SHA/梯度预检，再串行smoke20→正式18k→冻结→独立val。不得重启旧队列，不读test或自建数据。

Python C:\venv_bisenet\Scripts\python.exe；PYTHONPATH指向本工程src/PaddleSeg-2.10.0，覆盖旧editable；源未修改。wrapper入口run_loss_queue.ps1，调用loss_queue.py --run，Normal优先级。实际运行必须读取heartbeat/active_process/日志变化/真实exit，任务Running不是训练证据。

仅复用旧C0合法冻结，所有新loss从共同ImageNet初始化。普通/EMA精确val mIoU最大值冻结SHA，独立2px边界评估，参数应恒为3470434。损失不改变参数量或推理图；记录训练成本，不声称loss加速。

旧9项训练均有真实exit0及独立验证；旧wrapper最终回执缺失，任务系统中止码0xC000013A，不将它记exit0。详见provenance/previous_followup_audit.json。

主门槛沿用A0：IoU+.002、F1+.001、mIoU不降、P/R/BF1>=-.005，最多top2冻结后补43/44。20主项无按结果触发替补。计划结束后继续六个原始对照及冻结统一test、最终报告图和交付包，不因本轮完成就误称整个项目已完成。

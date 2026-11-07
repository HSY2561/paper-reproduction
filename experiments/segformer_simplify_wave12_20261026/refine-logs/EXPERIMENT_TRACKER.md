# W12执行跟踪

|组别|阶段|状态|备注|
|---|---|---|---|
|R0|复用控制|CPU证据已核，GPU全348重现待做|新源通过后复用W11三seed|
|D00|部署重排|CPU数值通过，GPU全348/速度待做|不重训不计模块|
|D01|首轮seed42|CPU通过，GPU smoke/18k待启动|无Aligned、W96|
|D02|首轮seed42|CPU通过，GPU smoke/18k待启动|无Aligned、W128|
|D03|首轮seed42|CPU通过，GPU smoke/18k待启动|无Aligned、W160|
|D04|首轮seed42|CPU通过，GPU smoke/18k待启动|Aligned流通道32、W128|
|D05|首轮seed42|CPU通过，GPU smoke/18k待启动|Aligned流通道16、W128|
|D06|首轮seed42|CPU通过，GPU smoke/18k待启动|DW3x3+PW流预测32、W128|
|D07|首轮seed42|CPU通过，GPU smoke/18k待启动|C2尺度CoarseFlow32、W128|
|D08|首轮seed42|CPU通过，GPU smoke/18k待启动|NativeSum128|
|D09|首轮seed42|CPU通过，GPU smoke/18k待启动|NativeSum192|
|D10|首轮seed42|CPU通过，GPU smoke/18k待启动|NativeSum256|
|C01|首轮seed42|CPU通过，GPU smoke/18k待启动|仅A0 S4FFN960|
|C02|首轮seed42|CPU通过，GPU smoke/18k待启动|仅A0 S4FFN896|
|条件S4组合|至多1家族×960/896|未启动|仅首轮全齐且符合门槛才运行|
|额外seed43/44|最多2候选|未启动|完整门槛预选后执行|

## 2026-10-27 00:48 启动核验

- R0新源码全348精确CM/mIoU复现通过；官方和独立评价真实exit0。
- D00全348计算完成，但159像素舍入差使预登记等价门槛未过，已排除，不减参、不训练、不放宽门槛。
- D02 smoke20真实exit0、数据RNG通过；正式18k已于00:47:37启动，00:48:07真实日志300/18000。最终性能待完成验证，退出码运行中未知。
- 上表为初始登记快照；后续以results/heartbeat.json、各真实exit/status和EXPERIMENT_RESULTS.md更新为准。不要重复启动任务。

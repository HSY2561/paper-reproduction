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

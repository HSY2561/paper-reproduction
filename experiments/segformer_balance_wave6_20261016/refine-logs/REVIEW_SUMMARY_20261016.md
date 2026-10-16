# 消融设计独立审查记录

审查在实现前读取本地官方模块源码和历史结果，未改文件、未运行GPU训练。

1. 接受“原始width256绝对锚点 + width128母体 + width64效率控制”设计；宽度变化不得计入模块数。
2. 12个核心机制：MSCAN SpatialAttention、SFNet AlignedModule、GlobalContextBlock、CAM、Separable ASPP、PolarizedSelfAttention、CrissCrossAttention、PAM、NonLocal2D、EncModule、GloRe GruModule、UAFM_Channel。加SE、PSP组成14个真实单模块。
3. 优先检验MSCA/Aligned/GC/CAM，再做更重全局机制，SE/PSP保留实际GPU反向检查；不能把兼容失败写成性能差。
4. 对128母体的模块归因与对B0的实际精度/效率比较必须同时存在。选模仅按val mIoU精确值，不能再以边界F1二次挑checkpoint。
5. 预注册Pareto/容差门槛，避免看到结果后改成功定义；不把更多参数带来的小收益称为效率提升。
6. 推荐两组互补联合：Aligned+GC；MSCA+CAM。全局注意之间不盲目组合。各组件先单独有效，联合必须超过更强的单模块。
7. 至少14次反复访问348张val会有选择偏差，需top2做seed43/44，再冻结test；公开模型即使弱也都要评估。
8. 参数量可以精确统计，paddle.flops会漏算注意力/矩阵分解；本轮不把不完整FLOPs当作部署事实。最终候选与B0须在训练停止后同一时段复测延迟，排除桌面负载及GPU频率变化。
9. 原LightHam普通/EMA选模错误已查实，EMA更高但仍未通过；旧普通best记录的iter5000实际是15000。审计记录见provenance/historical_corrections.json，原始证据不删除。

按历史日志，14模块+2宽度控制约8GPU小时；smoke/验证/联合/多种子合计约12–15小时。阶段二公开模型可能再需5–8小时，兼容排障单独记录。

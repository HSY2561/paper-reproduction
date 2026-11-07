# SegFormer-B0+Aligned 十模块候选消融计划

本轮固定完整256维 SegFormer-B0+Aligned 作为 A0 母体，RepDW/Ghost/PDC 不再进入候选池。首轮逐个训练10个候选：MSCA、LSKA、DWR、InceptionDWConv、CG Block、CAA、SAFM、Strip Pooling、Coordinate Attention、WTConv。每组只加入一个候选，不进行候选之间堆叠。

## 固定协议

- Crack500 train/val 1896/348，禁止筛选访问test、自建数据和机器人数据。
- FP32、RTX5060、seed42、batch4、AdamW 6e-5、PolynomialDecay end_lr0、CE、EMA、18000 iter、workers0。
- 完整 MiT-B0、256维投影和 Aligned 保持不变；候选在统一融合接口内只出现一次。
- 每组普通/EMA按精确val mIoU选模，独立官方val、348图混淆矩阵、精确2px边界、参数和真实exit证据均需保存。

## 候选问题与风险

|候选|主要假设|主要风险|
|---|---|---|
|MSCA|多尺度条带卷积补长裂缝上下文|算子多、过平滑、延迟增加|
|LSKA|大核可分离空间关系改善细长结构|大核访存和边界模糊|
|DWR|分组膨胀残差覆盖不同裂缝尺度|参数与延迟增加|
|InceptionDWConv|部分通道方向处理兼顾保留与效率|方向分支对斜裂缝有限|
|CG|局部+周围+全局信息减少低对比度漏检|全局重加权压制小前景|
|CAA|上下文锚点选择对齐后有效位置|池化削弱极细裂缝|
|SAFM|分组多分辨率调制保留局部并引入上下文|插值与池化成本|
|Strip Pooling|长条上下文改善方向性裂缝|错误桥接相邻裂缝|
|Coordinate Attention|低成本方向注意力作为效率对照|收益可能很小|
|WTConv|小波频率分解改善弱裂缝和纹理干扰|后端兼容与重建成本|

## 运行顺序与门槛

按 MSCA→LSKA→DWR→InceptionDWConv→CG→CAA→SAFM→Strip Pooling→Coordinate Attention→WTConv 顺序串行运行。每组先20iter smoke，再18000iter；首轮完成后比较相对A0的Precision、Recall、F1、裂缝IoU、mIoU、BF1、细线Recall、hard clDice、参数量。只有候选在精度、细裂缝和资源上均有可信信号，才登记seed43/44和统一test；不以单seed或单一IoU涨点声称SOTA。

预计纯训练约5–7小时，加验证和报告约7–10小时；真实耗时以日志为准。当前已有 W13 结果只作历史对照，不混入本轮训练。

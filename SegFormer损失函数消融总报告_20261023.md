# SegFormer-B0 + Aligned Crack500 损失消融与公开模型对比总报告

生成时间：2026-10-23。

## 结论

本轮完成了20个预登记损失方案、Dice兼容恢复、L01/L02的seed 43/44复核，并完成冻结公开 checkpoint 的统一 Crack500 test 评估。严格三种子门槛没有通过：L01 与 L02 的区域指标和 Recall 平均提高，但 Precision 平均分别下降 0.00772 和 0.01136，超过预登记的最多 0.005 下降限制。因此当前没有可以按预注册门槛宣称为最终改进模型的候选，也不能声称 SOTA。

## 实验协议

RTX5060、FP32、PaddleSeg 2.10.0、本地 Python 环境、SegFormer-B0 width128 + Aligned、batch=4、AdamW、学习率 6e-5、PolynomialDecay end_lr=0、18,000 iter、同一 Crack500 train/val 划分。损失筛选只使用 train/val；L01/L02 的 test_used 均为 false。公开模型在冻结验证 checkpoint 后才访问 1,124 张原始 test 图。

## 损失消融结果（seed42）

|方案|状态|Precision|Recall|F1|裂缝IoU|mIoU|边界F1|参数|门槛|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|A0 SegFormer-B0 width128 + Aligned|control|0.8268|0.7910|0.8085|0.6785|0.8282|0.4493|3.470M|—|
|CE + 0.1 Dice|complete_compat_clone|0.8220|0.8029|0.8123|0.6840|0.8310|0.4564|3.470M|通过|
|CE + 0.25 Dice|complete_compat_clone|0.8240|0.8007|0.8122|0.6838|0.8309|0.4573|3.470M|通过|
|CE + 0.5 Dice|complete_compat_clone|0.8082|0.8162|0.8122|0.6837|0.8307|0.4546|3.470M|未通过|
|CE + 1.0 Dice|complete_compat_clone|0.7973|0.8291|0.8129|0.6847|0.8311|0.4504|3.470M|未通过|
|CE + 0.1 Lovasz|complete|0.8065|0.8154|0.8109|0.6820|0.8297|0.4417|3.470M|未通过|
|CE + 0.25 Lovasz|complete|0.8049|0.8234|0.8140|0.6864|0.8320|0.4472|3.470M|未通过|
|CE + 0.5 Lovasz|complete|0.7991|0.8268|0.8127|0.6846|0.8310|0.4544|3.470M|未通过|
|CE weights [1.5, 1.0]|complete|0.8278|0.7770|0.8016|0.6689|0.8231|0.4418|3.470M|未通过|
|CE weights [2.0, 1.0]|complete|0.8531|0.7380|0.7914|0.6548|0.8159|0.4359|3.470M|未通过|
|CE weights [1.0, 2.0]|complete|0.7745|0.8444|0.8079|0.6777|0.8270|0.4177|3.470M|未通过|
|CE top 0.5|complete|0.8214|0.7986|0.8099|0.6805|0.8291|0.4520|3.470M|未通过|
|CE top 0.25|complete|0.8244|0.7954|0.8096|0.6801|0.8290|0.4502|3.470M|未通过|
|OHEM CE|complete|0.8196|0.8042|0.8118|0.6832|0.8306|0.4588|3.470M|未通过|
|Softmax Focal gamma 1.0|complete|0.8268|0.7896|0.8078|0.6776|0.8277|0.4502|3.470M|未通过|
|Softmax Focal gamma 2.0|complete|0.8164|0.7951|0.8056|0.6744|0.8259|0.4377|3.470M|未通过|
|CE + 0.5 Softmax Focal|complete|0.8212|0.7969|0.8089|0.6791|0.8284|0.4491|3.470M|未通过|
|OHEM + 0.25 Dice|complete|0.8191|0.7996|0.8092|0.6796|0.8286|0.4521|3.470M|未通过|
|Weighted CE [1.5,1] + 0.25 Dice|complete_compat_clone|0.8228|0.7955|0.8089|0.6791|0.8285|0.4536|3.470M|未通过|
|CE + 0.01 DualTask|complete|0.8020|0.8125|0.8072|0.6768|0.8269|0.4363|3.470M|未通过|
|CE + 0.01 RMI|complete|0.8201|0.8001|0.8100|0.6806|0.8292|0.4526|3.470M|未通过|

说明：L01–L04、L18 原始主队列因官方 Dice 对共享 labels 原地修改导致反向版本冲突；独立兼容副本只对 labels 做 clone，公式与系数不变，合并表同时保留原始失败状态和 compat_clone 证据。

## L01/L02 三种子复核

|方案|P均值±样本std|R均值±std|F1均值±std|裂缝IoU均值±std|mIoU均值±std|BF1均值±std|严格均值门槛|
|L01_CE_Dice|0.8143±0.0067|0.8162±0.0115|0.8152±0.0025|0.6880±0.0036|0.8331±0.0018|0.4578±0.0016|未通过（Precision均值下降超过0.005）|
|L02_CE_Dice|0.8126±0.0105|0.8175±0.0146|0.8149±0.0026|0.6877±0.0036|0.8328±0.0018|0.4573±0.0033|未通过（Precision均值下降超过0.005）|

L01 相对 seed42 的三种子均值变化：Precision -0.007725、Recall +0.013318、F1 +0.002841、裂缝IoU +0.004042、mIoU +0.002039、边界F1 +0.001339。L02：Precision -0.011364、Recall +0.016758、F1 +0.002728、裂缝IoU +0.003881、mIoU +0.001901、边界F1 +0.000073。

## Crack500 原始公开模型统一 test 对比

|模型|状态|mIoU|裂缝IoU|Precision|Recall|F1|边界F1|参数量|单图耗时|
|SegFormer-B0|test_complete|0.7695|0.5707|0.7289|0.7245|0.7267|0.3508|3.715M|18.64 ms|
|OCRNet-HRNet-W18|test_complete|0.7715|0.5757|0.7038|0.7598|0.7307|0.3429|12.110M|61.79 ms|
|PP-LiteSeg-STDC1|test_complete_after_val_gate_failed|0.7663|0.5639|0.7461|0.6979|0.7212|0.3328|8.150M|17.43 ms|
|PSPNet-R18|gpu_smoke_failed|—|—|—|—|—|—|—|—|
|DeepLabV3P-ResNet50|gpu_smoke_failed_historical|—|—|—|—|—|—|—|—|
|Fast-SCNN|gpu_smoke_failed|—|—|—|—|—|—|—|—|

SegFormer、OCRNet、PP-LiteSeg 均在同一 test evaluator 上完成。PP-LiteSeg 的验证门槛此前未通过，但在验证 checkpoint 冻结后补做了 test，并在表中明确标注。PSPNet-R50 没有可用 checkpoint；本地可运行的官方 PSPNet-R18 smoke 在 RTX5060/Paddle3.4 反向阶段以真实 exit=1 失败。DeepLabV3P-R50 和 Fast-SCNN 也保留真实 GPU smoke 失败证据，没有伪造 test 数字。

## 论文结论与后续工作

1. 将 SegFormer-B0 + Aligned 作为稳定母体；L01/L02 作为“提升 Recall/裂缝IoU但牺牲 Precision 的探索性负结果”，不写成最终模型。
2. 公开 test 中 OCRNet 的 mIoU 和 F1 高于 SegFormer，PP-LiteSeg 更轻但精度更低；当前公开对比范围内没有证据支持 SOTA。
3. 下一阶段可在冻结的 A0/L01 中择一进入自建数据和机器人闭环，但应先根据任务优先级决定是优先降低假阳性，还是优先提高细裂缝召回。
4. 若要补齐六模型表，应在独立兼容环境重跑 PSPNet、DeepLabV3P、Fast-SCNN，保留同一数据清单和 test evaluator；当前 RTX5060 后端失败属于工程兼容性限制。

## 复现证据

- 损失合并结果：`D:\论文复现\experiments\segformer_loss_wave8_20261019\dice_compat_recovery_20261020\results\consolidated_comparison.json`
- 三种子结果：`D:\论文复现\experiments\segformer_loss_wave8_20261019\matched_seeds_20261022\results\three_seed_summary.json`
- 公开模型结果：`D:\论文复现\experiments\segformer_public_wave9_20261023\results\public_comparison.json`
- 公开逐图 CSV：`D:\论文复现\experiments\segformer_public_wave9_20261023\results`
- 公开配置、日志、真实 exit：`D:\论文复现\experiments\segformer_public_wave9_20261023`

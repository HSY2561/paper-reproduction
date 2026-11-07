# R0/A0 三 seed 复用审计

日期：2026-10-25。范围：只读核验旧配置、初始化骨干、训练回执、checkpoint、日志、验证复现与源码；附 CPU 随机流见证。没有 GPU、没有训练、没有修改源码。详细逐文件 SHA 与证据路径见 `R0_reuse_audit.json`。

**结论：旧 A0 三 seed 可以保留为历史性能及资源参考，不能自动成为 W11 的严格匹配训练控制。** 新四格 E0/E1/E2/E3 的共同初始化与随机流控制不受此历史缺口影响，但严格“相对 A0 非劣/超过 A0”验收需要另补 W11 同协议 R0_matched，不能拿历史行冒充已完成配对。

## 已核实的历史证据

三个 seed 的配置中数据、增强、batch4、18000iter、AdamW、lr6e-5/PolynomialDecay/end_lr0、CE coef1 完全相同；CLI 都有 EMA、FP32、num_workers0、对应 seed。真实训练 exit 均为0，日志均保留 `iter: 18000/18000`。环境日志为 Windows11 / Python3.12.10 / Paddle3.4.0 / cuDNN90.9，属于既有 RTX5060 本地 Windows 后端，不是服务器4090或后续WSL训练。

| seed | 原始运行 | 验证裂缝IoU | F1 | mIoU | 2px BF1 | val选模 |
|---|---|---:|---:|---:|---:|---|
| 42 | wave6 M13_Aligned | .6785196293 | .8084738688 | .8281842714 | .4492848242 | 普通best，8000iter |
| 43 | wave7 followup A0_s43 | .6861709725 | .8138806606 | .8319891163 | .4542095897 | 普通best，8000iter |
| 44 | wave7 followup A0_s44 | .6853680005 | .8133155492 | .8317204951 | .4514366849 | 普通best，13000iter |
| 历史均值 | n=3 | **.6833528674** | **.8118900262** | **.8306312943** | **.4516436996** | 不属于新W11匹配均值 |
| 历史样本SD | ddof=1 | .0042049179 | .0029719415 | .0021234359 | .0024689006 | 完整失败/波动不删除 |

三个原始 ImageNet 骨干初始化文件均重新计算 SHA，与旧证据一致：
`211e15d5157b8d48b1931102c40094da0b578de1c153c5706360f7ad561a774e`。

冻结 checkpoint 当前文件 SHA 已重算并分别匹配：

- s42：`000b90f9668e35d0a7cae0fd535134802c5a64a7ced2eab05f869cef726cd5ea`
- s43：`2db07c0ab3d1bc37c6aa6feaa9fe5e2686f413fb30fb7963fa8a1e27234e2248`
- s44：`c8e6dd6e8e58fe9a8d56c37bd577efd63843912838be50030a8b5fa36d7ee1d2`

wave10 recovery 已有三份348图完整复现回执，actual/expected混淆矩阵完全一致，SHA对应上列checkpoint，边界采用同一精确欧氏2px、零外延定义。这些是已存在的推理复现证据；本次文件审计没有重新运行图像推理。

数据证据引用旧4488文件hash检查回执以及 W11 的 `data_preflight.json`（4488、bad=[]、train/val清单SHA匹配9月30日manifest），manifest SHA为 `55ba5746e33c6bcc082f0129fac18cd75d9671ff1483c10d15455c915f3cc1e4`。它说明数据内容核验有保留证据，不等于已经证明各历史运行逐次数据排列/增强完全重放；本次没有访问test或重读全部原图。

## 严格匹配缺口

1. **骨干预训练 SHA 不等于完整模型初始 SHA。** 在现有保留索引中没有找到三个旧 A0 随机头完整初始 binary 的 SHA 见证。旧构造按全局seed消耗随机数，新 W11 按具名组件产生并显式共享头张量，两者不能默认逐tensor相同。
2. **旧训练与 W11 的 Paddle 随机流起点不同。** 旧官方 `tools/train.py` 在179行set_seed，212行才访问builder.model；没有模型构造后再次set_seed。W11 `train_entry.py` 在模型/优化器/数据构造后、正式train前再次调用set_seed。旧初始化已消耗Paddle RNG，Dropout/DropPath起点不能按相同seed视作相同。
3. CPU见证：相同seed42，构造原A0后抽取16个Paddle正态数与重新set_seed后抽取的数组**不相同**；其SHA均在JSON。该见证只证明CPU构造会推进Paddle随机流，不冒充历史CUDA状态重放。Python/NumPy状态在这次原A0构造中未变化；即使数据增强流可能相同，也不能由此宣称Paddle Dropout/DropPath流相同。
4. 旧运行没有与 W11 新增强的逐批随机状态/实际索引清单一一配对的证据。旧trainer/dataset/transform/backbone/CE源码与W11对应文件SHA一致，并不能补齐完整初始化和随机状态缺口。

因此：旧均值可以列为“历史三seed参考”，资源数值可同后端重新测量；不能从这些行直接宣布 W11 严格matched非劣门槛或超越门槛通过。四格内部的 M1/M2效应与交互仍按W11共同协议单独判断。

## 已确定的修复路径，当前尚无训练结果

协调主任务将登记 `R0_matched_s42/s43/s44`，安排在核心四格12项之后：保留原MiT-B0，加载原ImageNet完整骨干；原A0头/分类器使用同seed E0的对应初始tensor；与四格同用 W11 train_entry 的构造后seed重置、数据/RNG审计、18000iter/CE/EMA/val冻结规则。保留旧结果，不覆盖也不伪装旧run已具备新协议。

CPU已验证现有新类直接接受原B0：`SegFormerComplementarity(num_classes=2, backbone=MixVisionTransformer_B0(), use_m1=False, use_m2=False)`，注册量3,470,434、可训练3,470,178；**不需要再修改源码**。这仅证明构造与参数正确；三项 R0_matched 正式训练、验证、冻结证据**当前待完成**。

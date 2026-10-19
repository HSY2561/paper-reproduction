# Wave8 损失函数消融独立审查

审查日期：2026-10-19。审查范围仅包括冻结的 `SegFormer-B0 + width128 + Aligned (A0)` 母体、Crack500 train/val 和 PaddleSeg-2.10.0 本地官方损失实现。本次没有运行 GPU、没有访问 test、没有修改模型或训练代码。外部 ChatGPT 对话的最新可读信息只有“损失函数优化”这一用户方向，助手正文为工具占位符，因此不能作为方案或结果证据。

## 结论

20 个主方案整体可执行，也能回答五类明确问题：区域重叠代理、IoU 代理、类别代价、困难像素挖掘、边界/局部关系约束。正式训练前必须满足下列修正：

1. `RMILoss` 配置必须使用源码字段 `rmi_radius`、`rmi_pool_way`、`rmi_pool_size`、`rmi_pool_stride`；`radius` 或 `poolstride` 会构造失败。计划中的 `num_classes=2, rmi_radius=3, rmi_pool_way=0, rmi_pool_size=3, rmi_pool_stride=3` 与接口相符。
2. `CE + 0.25 Lovasz(classes=[1])` 不是技术兼容替补。它与普通 Lovasz 走同一源码和算子，只是把目标从“批内出现的两类平均”改为“只优化裂缝类”。它应标为禁用的预登记科学备选，不能在普通 Lovasz 验证较差后启用，也不能声称能解决普通 Lovasz 的工程失败。
3. `DiceLoss` 会原地执行 `labels[labels == ignore_index] = 0`。所有 `MixedLoss` 中 Dice 必须最后调用；当前 `CE/OHEM/WCE -> Dice` 顺序正确。正式预检还应确认训练入口没有在同一输出之后复用已被 Dice 改写的标签。
4. `MultiClassFocalLoss` 才是本轮适用的 softmax focal。其源码中 `alpha` 是乘在全部像素上的标量，虽然文档沿用了“class 1 weight”的描述；`alpha=1` 不能解释成前景权重。不要用二通道逐元素 sigmoid 的 `FocalLoss` 替代。
5. 旧复核文件中的 `0.6852607627`（A21）和 `0.6833528674`（A0）是 **crackIoU** 均值，不是 mIoU；对应 mIoU 为 `0.8316089218` 与 `0.8306312943`。A21 的 crackIoU 增益 `+0.0019078953` 低于预登记 `+0.002`，因此不把结构模块带入本轮是正确决定。

## 固定母体与可归因范围

- 母体固定为 A0；同一 ImageNet MiT-B0 初始化、数据划分、增强、batch=4、crop=400、AdamW、学习率、多项式日程、18k、FP32 和 checkpoint 选择规则均不得改变。每个方案从同一预训练骨干重新训练，不能从已训练 A0 微调。
- 每项只改变 loss 配置。若模型源码和推理图确实相同，则参数量和推理延迟理论上完全相同；这一轮只能比较验证精度与训练开销，不能把训练损失造成的精度变化写成推理效率改进。
- A0 可复用旧 checkpoint 作为 seed42 对照，但必须保存源/config/数据清单等价证据。checkpoint 仍按“普通/EMA 中 exact validation mIoU 较高者”冻结，不能为某个 loss 改用 crackIoU、F1 或阈值调优选择 checkpoint。

## 20 个主方案的接口与科学含义

| ID | 方案 | 检验的问题 | 源码行为与主要风险 |
|---|---|---|---|
| L01–L04 | CE + Dice，系数 0.1/0.25/0.5/1.0 | 区域重叠约束强度 | 官方 Dice 对 softmax 后的背景、裂缝两类等权平均，并非前景专用；强系数可能改变 Precision/Recall 工作点。Dice 必须置后。 |
| L05–L07 | CE + LovaszSoftmax，系数 0.1/0.25/0.5 | IoU 排序代理能否补充像素 CE | 默认 `classes='present'`，在整个 batch 展平后对出现类别求均值；排序操作增加训练开销，也可能放大标注噪声。 |
| L08–L10 | WCE `[1.5,1]`、`[2,1]`、`[1,2]` | 背景误检代价与前景漏检代价的方向性对照 | 类别顺序必须冻结为 0=背景、1=裂缝。前两项倾向减少 FP/提高 Precision，第三项倾向减少 FN/提高 Recall，但效果不能预设。官方 CE 按有效像素的平均类别权重归一化，绝对权重尺度不会简单放大总损失。 |
| L11–L12 | CE top-k 0.5/0.25 | 去掉易像素是否有利于细裂缝 | `CrossEntropyLoss` 在整个 batch 展平后取 top-k，不是逐图；k 按总像素数计算。ignore 像素损失为零，但仍计入 k 的基数，存在大量 ignore 时语义会改变。 |
| L13 | OHEM，`thresh=0.7,min_kept=40000` | 按真类低概率挖掘困难像素 | 作用于整个 batch；400×400×4 时 40000 仅为总像素 6.25%。0.7 阈值常会保留多于 40000 个像素，不能把它描述为固定 top-6.25%。 |
| L14–L15 | MultiClassFocal，gamma 1/2 | 连续衰减易样本是否优于硬截断 | softmax CE 上乘 `(1-p_t)^gamma`；`alpha=1` 为全局标量。单独 focal 的梯度尺度与 CE 不同，应记录早期 loss 和梯度，但不得为它单独调学习率。 |
| L16 | CE + 0.5 MultiClassFocal(gamma=2) | 保留全像素 CE 时增加困难样本强调是否更稳 | 两项高度相关，属于复合目标而非新机制；固定系数是可接受的预登记探索。 |
| L17 | OHEM + 0.25 Dice | 困难像素与全局重叠约束是否互补 | OHEM 先执行、Dice 最后；可能同时放大难例噪声和前景扩张。 |
| L18 | WCE `[1.5,1]` + 0.25 Dice | 抑制 FP 与区域重叠之间能否取得折中 | 允许作为预登记二因素组合，但不能从 L08/L02 的结果事后调系数。 |
| L19 | CE + 0.01 DualTaskLoss(`tau=0.5`) | 边界梯度一致性是否改善 BF1 | 官方实现向 logits 加 Gumbel 随机噪声并计算梯度幅值，增加随机性和卷积训练开销。必须验证固定 seed 可复现且对 logits 梯度非零、有限。 |
| L20 | CE + 0.01 RMILoss | 3×3 局部关系约束是否改善裂缝连贯性 | 官方实现对两通道分别 sigmoid，并在内部使用 float64 协方差、矩阵求逆和 Cholesky；可能慢、显存高或出现数值失败，且 RMI 值可为负。`loss_weight_lambda` 在该实现的前向路径中未被使用，不要把它当有效超参数。 |

这 20 项中，同一家族不同系数属于超参数消融，不能在论文中写成 20 种独立损失方法。L17/L18 是预登记的小组合，宜放在对应单项之后运行，但即使单项表现差也不得取消或改系数，否则队列变成结果驱动。

## 技术替补与排除项

- `BootstrappedCrossEntropyLoss(min_K=10000, loss_th=0.3)` 可作为困难像素实现的技术替补。它逐图排序，与 batch 级 top-k/OHEM 不等价；只能在目标主项构造、反向或后端算子真实失败时替换，并记录原项错误、替补映射和冻结时间。源码访问 `sorted_loss[self.K]`，因此预检必须确保每图像素数大于 10000；400×400 满足这一条件。大量 ignore 使有效像素少于 K 时仍有纳入零损失的风险。
- `CE + 0.25 Lovasz(classes=[1])` 仅可保留为禁用的科学备选。若确实要运行，应在任何该家族结果可见前把它提升为主项并说明取代哪一项；不得以验证集表现触发。
- 排除 `SemanticConnectivityLoss` 是正确的。官方实现先 `argmax`，随后转 NumPy 并调用 OpenCV 连通域，再重新构造 Tensor，不能提供可追溯到 logits 的正常梯度。也不应在本轮自写 clDice/Tversky，以免同时改变算法来源与实现质量。

## 正式训练前的放行检查

每个配置先做小张量和短 smoke，只判定技术可运行性，不读取 val 排名：

1. 构造的 loss 类型、参数、`MixedLoss` 内部顺序和系数与目录逐项一致；单输出 SegFormer 只配置一个外层 loss type，组合项由一个 `MixedLoss` 承载。
2. 在含背景、裂缝和 ignore=255 的合成标签上，逐组件前向有限、反向到 logits 的梯度存在且绝对和大于零；对 RMI 额外核验 inverse/Cholesky 的梯度有限，对 DualTask 连续两次同 seed 结果可复现。
3. 检查输入标签在组合中是否被改写；Dice 用标签副本或严格放在最后。检查 class 0/1 映射与数据标签一致。
4. 正式训练只允许真实退出码 0、18000/18000、普通与 EMA 均完成验证、按预登记 mIoU 选择并冻结 checkpoint SHA。NaN、断梯度、OOM、后端不支持才是技术失败；val 较差不是替补条件。
5. 记录每项训练 wall time 与峰值显存。RMI、Lovasz、OHEM、DualTask 的训练成本可能明显不同，但不重复测量或宣称推理参数/速度差异；模型图哈希相同即可证明 loss 未进入推理。

## 选择门槛

沿用既有 A0 parent gate，避免在看见 20 项结果后重写标准。seed42 初筛时，每项与 A0 seed42 同协议比较；候选必须同时满足：

- crackIoU `Δ >= +0.002`；
- F1 `Δ >= +0.001`；
- mIoU 不下降；
- Precision、Recall、boundary F1 各自 `Δ >= -0.005`。

若没有方案同时通过，结论应为“本轮未找到通过预登记门槛的 loss”，不能降门槛。通过者按完整门槛和 Pareto 关系在 test 之前冻结最多 top-2，补 seed43/44；最终用与 A0 相同 seed 的成对差值汇总，仍按三 seed 均值应用同一门槛，并报告逐 seed 数值和样本标准差。不要把 A0 的三 seed crackIoU 均值 `0.6833528674` 当作 seed42 基线；seed42 A0 crackIoU 是 `0.6785196293`。

20 个配置共用同一 348 图 val，家族网格和 top-2 选择都会产生重复使用验证集与 winner's curse。三 seed 只能估计训练随机性，不能消除选择偏差。最终 test 只能在候选和 checkpoint SHA 冻结后统一评估一次；若还要与六个公开模型比较，六者也必须用同一 test 和评估实现完整报告，即使其结果较弱。任何 loss 获胜都只能称为本数据划分、当前训练预算下的经验改进，不保证 SOTA，也不能单凭单 seed 的微小差异作因果结论。

## 核验源码

- `paddleseg/models/losses/cross_entropy_loss.py`
- `paddleseg/models/losses/dice_loss.py`
- `paddleseg/models/losses/lovasz_loss.py`
- `paddleseg/models/losses/ohem_cross_entropy_loss.py`
- `paddleseg/models/losses/focal_loss.py`
- `paddleseg/models/losses/mixed_loss.py`
- `paddleseg/models/losses/gscnn_dual_task_loss.py`
- `paddleseg/models/losses/rmi_loss.py`
- `paddleseg/models/losses/bootstrapped_cross_entropy.py`
- `paddleseg/models/losses/semantic_connectivity_loss.py`
- `paddleseg/core/train.py`

上述路径均相对于 `D:/论文复现/experiments/segformer_aligned_wave7_20261018/src/PaddleSeg-2.10.0`。本审查只确认本地冻结源码中的接口与行为，不把文档字符串中与实现不一致的描述当作实验事实。

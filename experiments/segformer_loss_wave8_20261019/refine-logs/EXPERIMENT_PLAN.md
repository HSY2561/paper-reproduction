# SegFormer-B0＋Aligned 损失函数优化：20方案执行计划

登记日期：2026-10-19。该计划在本轮首个损失方案训练前冻结。活动工程：`D:/论文复现/experiments/segformer_loss_wave8_20261019`。

## 1. 方向、依据与旧实验完成情况

引用对话“读取最新进度”的用户消息可读，最新方向是“重新优化损失函数”。助手正文由接口返回内容引用占位符，未能完整读取，因此下面是结合本地证据制定的执行方案，并非声称已取得网页中完整助手规划。如用户补充原文，以用户最新要求调整未运行部分并保留修订记录。

先等旧实验结束的条件已满足：wave7 的22项结构训练、后续9项归因与多种子训练均完成；9项逐阶段真实exit=0，18000/18000、冻结SHA、最高普通/EMA验证选择和独立验证已审计，错误0。后续队列在10月19日20:19生成完成文件。需要单独说明：旧计划任务末端没有最终wrapper退出JSON，系统返回0xC000013A，中止原因未知；不能把wrapper写成exit0。全局进程核验确认旧GPU训练已退出，9项训练完成的依据是各自真实回执，而非任务状态。

旧三seed验证均值如下（不是test，也不是跨论文SOTA）：

| 模型 | 裂缝IoU | F1 | mIoU | Precision | Recall | BF1 | 参数M |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 68.0389% | 80.9798% | 82.9082% | 81.9902% | 80.0075% | 45.0887% | 3.715170 |
| A0 | 68.3353% | 81.1890% | 83.0631% | 81.8156% | 80.5944% | 45.1644% | 3.470434 |
| A21 | 68.5261% | 81.3236% | 83.1609% | 81.5531% | 81.1075% | 45.3797% | 3.503330 |
| A10 | 68.1020% | 81.0246% | 82.9422% | 82.0310% | 80.0572% | 44.8862% | 3.656994 |

A21（Aligned＋Polarized）的三seed平均裂缝IoU相对A0增益0.0019078953，即0.19079个百分点，仍低于预登记0.002（0.2个百分点）门槛；不得四舍五入放行。A10（Aligned＋AFNB）平均裂缝IoU相对A0下降0.0023328659。本轮固定更小的A0：**SegFormer-B0、解码宽度128、Aligned**，不继续叠加失败结构。A0的减参来自结构，不能归因给损失函数。

旧速度交错复测仍有波动，并且480输入中位数A0约15.61ms、B0约13.93ms，尚无加速证据。新损失不进入推理图，不会进一步减参或天然提速。本文目标是保留现有参数优势，检验能否改善精度与假阳性/漏检取舍。

## 2. 执行顺序

1. 固定旧证据，确认无旧GPU任务，核验本轮代码、配置、权重、4488个train/val源文件哈希；在新源码上重现C0的完整混淆矩阵。
2. 对20方案逐组件及组合做有限值、非零有限logits梯度、含ignore标签的小张量反向预检；DualTask额外做相同seed重复。技术失败记录真实错误，不填性能。
3. 严格串行：L01至L20各自20iter官方trainer smoke，通过者从共同ImageNet初始化完整18k。不能从训练后的A0微调，不能按中间val弱取消方案。
4. 对每项成功训练，按普通/EMA精确val mIoU最高者冻结SHA，再官方val.py与全348图独立混淆矩阵、2px边界F1复核，记录逐图FP/FN、空预测与失败案例。
5. 按预登记完整门槛冻结最多top2配方，再补seed43/44；C0/B0已有相同seed合法结果复用。尚未三seed复核的单seed提升只作候选。
6. 确定最终候选后，继续六个原始公开模型：SegFormer-B0、OCRNet-HRNet-W18、PP-LiteSeg-STDC1、PSPNet-R50、DeepLabV3+-R50、Fast-SCNN。复用同协议合法结果、补缺项；全部按val冻结后统一test，不因对照val弱删行。最后整理论文报告、消融/对比表、图、逐图CSV和复现索引/报告压缩包。

本轮暂不读取test或自建数据，不进行机器人实验。历史test已被观察过，应在论文中说明，后续保留source-frame-disjoint复核，不能将当前test宣传为全项目从未查看过。

## 3. 20个预登记配置

以下是20种配置，并非20种独立算法。同一损失的不同权重明确属于超参数消融。每项修改位置仅为`configs/Lxx.yml`的`loss`；模型/encoder/decoder/推理阈值保持不变。CE为官方CrossEntropyLoss，Softmax Focal为MultiClassFocalLoss。

| 编号 | 官方损失配置 | 解决的问题及预期收益 | 训练成本及风险 |
|---|---|---|---|
| L01_CE_Dice | CE + 0.1 Dice | 区域重叠与小前景召回；单独比较Dice系数 | 可能扩张前景、降低Precision；Dice必须位于MixedLoss最后 |
| L02_CE_Dice | CE + 0.25 Dice | 区域重叠与小前景召回；单独比较Dice系数 | 可能扩张前景、降低Precision；Dice必须位于MixedLoss最后 |
| L03_CE_Dice | CE + 0.5 Dice | 区域重叠与小前景召回；单独比较Dice系数 | 可能扩张前景、降低Precision；Dice必须位于MixedLoss最后 |
| L04_CE_Dice | CE + 1.0 Dice | 区域重叠与小前景召回；单独比较Dice系数 | 可能扩张前景、降低Precision；Dice必须位于MixedLoss最后 |
| L05_CE_Lovasz | CE + 0.1 Lovasz | 像素分类与IoU代理目标结合 | 排序和更大训练开销；可能牺牲细小边界 |
| L06_CE_Lovasz | CE + 0.25 Lovasz | 像素分类与IoU代理目标结合 | 排序和更大训练开销；可能牺牲细小边界 |
| L07_CE_Lovasz | CE + 0.5 Lovasz | 像素分类与IoU代理目标结合 | 排序和更大训练开销；可能牺牲细小边界 |
| L08_WeightedCE | CE weights [1.5, 1.0] | 前两组增加背景误检代价；第三组为前景召回反向对照 | 背景加权可能漏检，前景加权可能假阳性；类别顺序0背景/1裂缝 |
| L09_WeightedCE | CE weights [2.0, 1.0] | 前两组增加背景误检代价；第三组为前景召回反向对照 | 背景加权可能漏检，前景加权可能假阳性；类别顺序0背景/1裂缝 |
| L10_WeightedCE | CE weights [1.0, 2.0] | 前两组增加背景误检代价；第三组为前景召回反向对照 | 背景加权可能漏检，前景加权可能假阳性；类别顺序0背景/1裂缝 |
| L11_TopKCE | CE top 0.5 | 降低易背景像素对梯度的支配 | 可能过拟合标注噪声；topK是CE内部机制变体 |
| L12_TopKCE | CE top 0.25 | 降低易背景像素对梯度的支配 | 可能过拟合标注噪声；topK是CE内部机制变体 |
| L13_OHEM | OHEM CE | 优先低真类概率的困难像素 | 过强难例挖掘可能放大噪声；min_kept针对整个batch |
| L14_SoftmaxFocal | Softmax Focal gamma 1.0 | 按困难程度降低易样本贡献 | 不同于双sigmoid FocalLoss；标量alpha不是前景权重 |
| L15_SoftmaxFocal | Softmax Focal gamma 2.0 | 按困难程度降低易样本贡献 | 不同于双sigmoid FocalLoss；标量alpha不是前景权重 |
| L16_CE_Focal | CE + 0.5 Softmax Focal | 保留整体CE并强调困难像素 | 复合目标改变梯度幅值；固定学习率并记录 |
| L17_OHEM_Dice | OHEM + 0.25 Dice | 难像素和区域约束是否互补 | 误检/漏检方向可能冲突；Dice置后 |
| L18_WeightedCE_Dice | Weighted CE [1.5,1] + 0.25 Dice | 背景惩罚和前景重叠的取舍 | 两项目标强度可能抵消；不改变图像或标签文件 |
| L19_CE_DualTask | CE + 0.01 DualTask | 官方梯度边界一致性，沿用同一分割头 | 额外随机采样/卷积训练开销；必须先验证对logits梯度有效 |
| L20_CE_RMI | CE + 0.01 RMI | 局部邻域互信息关系约束 | 内部float64/矩阵逆开销和数值稳定性；RMI值可为负，不伪称纯FP32损失计算 |

L01–04额外softmax/区域归约；L05–07需要像素排序；加权CE接近原CE开销；top-k/OHEM含筛选/排序；Focal增加概率权重运算；DualTask增加Gumbel采样和梯度卷积；RMI包含float64协方差、逆矩阵及Cholesky，训练时间/显存可能明显增加。全部推理图和参数相同（3,470,434），不根据短时延迟噪声编造损失导致的加速。

MixedLoss系数明确使用`[1, λ]`，并非归一化到和为1；不同系数也会改变总梯度幅值。固定学习率能比较这些训练配方，但不能把全部收益单独归因于数学机制。若最终配方有效，论文应保留这一限制，必要时另登记等总权重控制，不在本轮中途改系数。官方Dice原地改ignore标签，因此所有组合Dice必须最后；不修改原始图像或标签文件。MultiClassFocal的alpha=1是全局标量，并非前景类别权重。RMI损失可负，仅以非有限/断梯度等判工程失败。

独立审查后取消两个未运行备选：BootstrappedCE与仅前景Lovasz不作为按结果触发的替补。技术失败就保留失败，不用新配方凑完成数。已生成的未用备选配置移至provenance/prelaunch_disabled_reserves供审计。

## 4. 固定实验协议

RTX5060；同一Python/Paddle3.4环境；模型FP32（RMI内部官方float64如实记录）；seed42；batch4；AdamW(lr6e-5, beta=.9/.999, wd=.01)；PolynomialDecay end_lr=0；18000iter；EMA；每1000iter验证；workers0。相同0.75–1.25缩放、400×400裁剪、水平翻转、RandomDistort及Normalize；val只Normalize并恢复原尺寸。

Crack500 train1896、val348；只读数据入口`C:/Users/user/algolab_data`；对照4488文件原始SHA清单。共同MiT-B0 ImageNet初始化SHA为`211e15d5157b8d48b1931102c40094da0b578de1c153c5706360f7ad561a774e`。

C0不重训，复用A0 seed42冻结权重（SHA `000b90f9668e35d0a7cae0fd535134802c5a64a7ced2eab05f869cef726cd5ea`），先在本轮源码上重现混淆矩阵。C0 seed42的裂缝IoU=0.6785196293，F1=0.8084738688，mIoU=0.8281842714；不能拿三seed均值当seed42筛选基准。

边界定义：前景减3×3零外延腐蚀；欧氏精确距离≤2像素对称匹配，聚合全部348图匹配计数后算BF1；显式处理空边界。F1和前景IoU由同一二分类混淆矩阵导出，不是两份独立证据。

## 5. 放行、终止与结论

沿用结构轮A0完整门槛，所有差值与相同seed的C0比较：裂缝IoU至少+0.002、F1至少+0.001、mIoU不降，Precision/Recall/BF1各下降不超过0.005。该门槛在首项训练前冻结，结果出来后不降低。另记录Precision+0.005且区域不降的诊断轨道，但不能绕过完整门槛放行最终模型。

通过者按区域精度、Precision/Recall与边界取舍选最多top2，先冻结配置和checkpoint SHA，再匹配seed43/44；报告逐seed、均值、样本标准差和成对差值，三seed均值仍须过相同门槛且至少2/3种子IoU改善。反复20配置使用同一val存在选择偏差，多seed只能检验训练波动，不能消除验证集搜索偏差。

每组必须有真实train exit0、18k最终权重和日志、冻结选择依据、独立验证、参数核验。阶段超12h、日志30min不变、磁盘低4GiB或数据/源哈希变化停止并保留证据。NaN、断梯度、OOM、官方后端错误记技术失败；不能把val差记工程失败。旧任务结束后只清理其对应系统任务注册，正式权重、最佳权重、失败证据和原始数据保留。磁盘清理只针对有完整回执的smoke重复权重，事先保存路径/大小/SHA清单。

没有配方通过就报告“本轮没有找到达到预登记门槛的损失改进”，不继续无限堆模块。本轮即便有提升，也仅为当前数据协议和预算下的经验结果；未完成六模型统一test前不称当前对比最优，更不能称领域SOTA。

## 6. 时间、输出和复现

20项正式训练按最近同机真实耗时估计约20–40小时（资源竞争、RMI成本或兼容问题可能延长）；先做的预检/复现不计已训练完成。随后top2多seed及公开模型缺项还需额外时间，以实际日志更新，不承诺固定完成时刻。

输出：`refine-logs/RESULTS.md`、`results/loss_ablation.csv`、逐图CSV、普通/EMA选择与SHA、各阶段日志及真实exit、数据/源哈希、候选冻结说明。根目录同步更新`SegFormer损失函数消融进展_20261019.md`及`模型消融文档.md`当前章节；最终交付遵循上一轮`refine-logs/FINAL_DELIVERY.md`，汇总报告、表格/图、复现索引和不含原始数据/大权重的压缩包。

源码以本地冻结PaddleSeg-2.10.0为准，官方说明参见[MixedLoss](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/paddleseg/models/losses/mixed_loss.py)、[FocalLoss及MultiClassFocalLoss](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/paddleseg/models/losses/focal_loss.py)。独立审查见`provenance/REVIEW.md`；本轮没有修改模型或损失实现，仅使用官方配置。

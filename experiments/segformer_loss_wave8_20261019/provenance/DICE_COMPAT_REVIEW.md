# Dice / MixedLoss 兼容修复独立审查

审查日期：2026-10-20。审查对象为 `D:/论文复现/experiments/segformer_loss_wave8_20261019` 的预检证据和冻结 PaddleSeg-2.10.0 源码。本审查没有运行 GPU、没有访问 test、没有修改活动源码或队列；仅做了 CPU 小张量复现并写入本文件。

## 审查结论

建议的单行修复

```python
# before
output = loss(logits, labels)

# compatibility patch
output = loss(logits, labels.clone())
```

在当前单输出 SegFormer 和已登记的 `MixedLoss` 方案中是数学等价的输入隔离修复，可以放行到“独立兼容源码复制 -> GPU 梯度预检 -> smoke -> 18k”的流程。它不改变 CE、Dice 或 WCE 的公式、系数、模型参数、初始化、seed 和推理图，也不构成新损失或新算法。它会增加训练时标签复制和少量显存/时间开销，正式记录时必须称为 PaddleSeg-2.10.0 的兼容补丁，不能称为完全未修改的官方源码。

该结论只授权恢复 L01–L04 和 L18 的原预登记实验。主队列必须自然结束后再复制源码和冻结补丁，不能停止、重启或热修改当前进程。

## 失败原因核验

`results/gradient_preflight.json` 的真实记录为 20 项中 15 项通过，以下 5 项组合反向失败：

- L01–L04：CE + Dice，Dice 系数分别为 0.1、0.25、0.5、1.0；
- L18：Weighted CE `[1.5, 1.0]` + 0.25 Dice。

五项错误一致：`CrossEntropyWithSoftmaxGradNode` 检测到用于反向的标签 Tensor 被后续原地操作改变，`tensor_version` 比保存版本大 1。预检同时显示单独 CE/WCE 与 Dice 的前向有限、各自反向梯度有效；Dice 单项把 16 个 ignore 标签从 255 改成 0。因而这是组合容器共享可变标签造成的工程失败，不是损失公式、超参数或数据失败。

源码调用链也与错误一致：

1. `MixedLoss.forward()` 依次把同一个 `labels` 对象传给 CE 和 Dice；
2. CE 的反向节点保存标签及其版本；
3. `DiceLoss.forward()` 先构造 `mask = labels != 255`，随后原地执行 `labels[labels == 255] = 0`；
4. 总 loss 在两个组件都前向完成后统一 backward，此时 CE 保存的标签版本已经变化，触发 Paddle 的版本保护。

把 Dice 放在最后只能保证其他 loss 的**前向取值**先看到 255，不能阻止 Dice 在统一 backward 前修改 CE 保存的同一 Tensor。因此“Dice 最后”本身不足以修复该错误。

## clone 的数学等价性

标签是离散 int64 监督量，不需要梯度。`labels.clone()` 保留完全相同的 shape、dtype 和像素值，但为每个组件建立独立 Tensor/版本计数：

- CE/WCE 的 clone 始终保持原标签和 ignore=255，其损失与未发生后续污染时相同；
- Dice 在自己的 clone 上先计算 ignore mask，再把该 clone 的 255 改为 0 供 one-hot 使用；mask 仍把这些位置从 Dice 分子、分母中排除，所以 Dice 数学值不变；
- Dice 不再改变 CE/WCE 保存的标签，也不改变传入 `MixedLoss` 的基础标签；
- logits 没有被 clone，各损失对同一 logits 的梯度仍按原系数相加，因此优化目标仍是原登记的 `CE/WCE + λ Dice`。

CPU 复现使用同一冻结源码、seed42、`[1,2,32,32]` logits，并在标签中放入 16 个 ignore 像素：

| 调用方式 | CE | Dice | 基础标签改变数 | backward |
|---|---:|---:|---:|---|
| 两组件共享 labels | 0.9105772972 | 0.5786695480 | 16 | 失败，版本不匹配 |
| 每组件使用 `labels.clone()` | 0.9105772972 | 0.5786695480 | 0 | 通过，梯度有限，`abs_sum=1.0900528431` |

前向值一致而副作用消失，直接支持“公式等价、执行兼容”的判断。系数只在线性组合输出时相乘，不影响上述版本隔离，因此同一修复覆盖 L01–L04；WCE 读取 clone 中相同类别标签和权重，原理同样覆盖 L18。

## 对其他项和 ignore 标签的影响

逐组件 clone 会使所有 `MixedLoss` 子项都无法修改基础标签。这符合组合损失应相互独立读取监督信号的语义。当前登记的 CE、WCE、Lovasz、MultiClassFocal、OHEM、DualTask 和 RMI 没有任何一项需要通过修改标签向下一项传递状态，因此不存在预期通信被切断的问题。

L17（OHEM + Dice）在原预检中已经通过，尽管报告显示 Dice 最终仍改变了基础标签 16 个位置。原因是 OHEM 在调用内部 CE 前已经通过乘法、重建和 reshape 形成了新的标签 Tensor，其反向没有保存后来被 Dice 修改的基础 Tensor。当前单输出训练循环在 `loss.backward()` 之后不再用标签计算其他目标，所以已有 L17 运行无需重训。修复版若重新计算 L17，预期前向/梯度数学等价，但没有科学理由仅为统一源码重跑已经合法完成的组。

仍需注意三点：

1. clone 修复的是标签共享副作用，不是 Dice 内部所有原地操作的一般证明；GPU 预检仍必须覆盖前向、组合 backward、有限且非零 logits 梯度。
2. 每个二组件组合会保留/创建 int64 标签副本。batch4、400×400 时单个标签张量约 5.12 MB，开销只在训练期，但应由 smoke 实测峰值显存，不能假定为零。
3. 如果未来加入依赖标签对象身份或故意跨 loss 写标签的自定义实现，本结论不能自动外推；当前 20 项没有这种机制。

## 恢复运行的证据要求

主队列结束后，恢复流程应满足以下条件：

1. 从当前冻结源码复制到新的、明确命名的兼容恢复目录；禁止修改活动源码。只允许 `paddleseg/models/losses/mixed_loss.py` 上述一行发生语义变化。
2. 保存原源码树 SHA、修复源码树 SHA、单文件 before/after SHA、精确 diff、Paddle/Python/CUDA/设备版本、修复时间和审查文件 SHA。源清单中写明 `algorithm_change: false`、`compatibility_fix: per_component_label_clone`。
3. 原 5 次失败记录和 `results/gradient_preflight.json` 必须保留，不覆盖、不改写。新证据使用独立名称，例如 `gradient_preflight_dice_compat.json`、`source_hashes_dice_compat.json`。
4. 保留原 L01–L04/L18 配置文件 SHA；恢复尝试另设 attempt/source 标识，例如实验臂仍为 `L01_CE_Dice`，执行记录为 `L01_CE_Dice_compat1`。最终表可归并到原预登记实验臂，但必须链接原失败与恢复证据。
5. 先在修复源码中对 5 项全部做 GPU 组合预检：组件值有限、基础标签改变数为 0、组合 backward 成功、logits 梯度有限非零。随后逐项正式 smoke；只有真实 exit0 才能进入完整 18k。
6. 完整训练仍要求 18000/18000、真实 exit0、普通/EMA 均完成验证、按既定 exact val mIoU 选模并冻结 checkpoint SHA。不得因 val 表现决定是否启用修复、调整 Dice 系数或增加新实验。
7. 15 个已合法组不重训。最终报告按 job 明确 source snapshot：15 项使用原冻结源码，5 项使用兼容源码。模型结构、配置和 loss 数学定义相同，但训练实现来源不能隐去。

原失败发生在任何正式优化步骤之前，恢复 5 项不属于重复试验，也不能把原失败计为负精度结果。若 clone 后仍出现断梯度、非有限值、OOM 或 smoke 失败，应按新的技术失败保留，不再通过修改公式、系数或训练超参数补救。

## 审查依据

- `results/gradient_preflight.json`
- `gradient_preflight.py`
- `provenance/jobs.json`
- `provenance/config_hashes.json`
- `src/PaddleSeg-2.10.0/paddleseg/models/losses/mixed_loss.py`
- `src/PaddleSeg-2.10.0/paddleseg/models/losses/dice_loss.py`
- `src/PaddleSeg-2.10.0/paddleseg/models/losses/cross_entropy_loss.py`
- `src/PaddleSeg-2.10.0/paddleseg/models/losses/ohem_cross_entropy_loss.py`
- `src/PaddleSeg-2.10.0/paddleseg/core/train.py`

以上均为本地冻结证据；CPU 复现只验证张量版本与数学等价性，不替代后续 GPU 预检和正式 smoke。

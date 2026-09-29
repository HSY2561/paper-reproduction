# BiSeNetV2 裂缝分割优化实验计划

**日期**：2026-09-25  
**主线模型**：PaddleSeg 官方 BiSeNetV2  
**当前最佳候选**：BiSeNetV2 + OHEM + Dice + EMA  
**服务器工程**：`/hy-tmp/crack_project`  
**数据盘**：`/hy-tmp`，100 GB，当前约 65 GB 可用

## 1. 先统一模型命名

论文中的工程基线名称写作 **BiSeNetV2**。`CrossEntropyLoss` 是官方默认训练配置，不是另一个网络模型，因此不再把“BiSeNetV2 + CE”作为模型名称。

实验记录中可以写：

- 模型：BiSeNetV2；
- 默认训练配置：CE；
- 改进配置：BiSeNetV2 + OHEM、BiSeNetV2 + Dice、BiSeNetV2 + EMA 等。

当前 Crack500 固定测试结果：

| 配置 | mIoU | 裂缝 IoU | Recall | 前景 F1 |
|---|---:|---:|---:|---:|
| BiSeNetV2（默认 CE） | 0.7606 | 0.5545 | 0.7122 | 0.7134 |
| BiSeNetV2 + OHEM + Dice + EMA | 0.7694 | 0.5705 | 0.7233 | 0.7264 |
| OCRNet-HRNet-W18 参考值 | 0.7760 | 0.5819 | 0.7128 | 0.7356 |

当前改进候选距离 OCRNet 参考 mIoU 还有 0.0066，裂缝 IoU 还有 0.0114。后续目标是超过这组已完成的参考结果；是否能够称为严格文献 SOTA，要在论文中根据数据协议和训练设置谨慎表述。

## 2. 问题诊断

目前已经试过 OHEM、Dice、Focal、Lovasz、EMA、RandomRotation 以及若干 PaddleSeg 损失组件。现有结果说明：

1. 类别不平衡损失能够提升 Recall，但单独使用时对区域 IoU 的提升有限。
2. OHEM + Dice + EMA 已经提供了约 0.0088 mIoU 的提升，继续堆叠同类损失的边际收益预计较低。
3. 还缺少对裂缝边界、长条连通性和多尺度上下文的直接约束。
4. BiSeNetV2 已经包含 detail branch、semantic branch 和 BGA，新增模块必须放在明确的位置，不能把多个完整网络无依据地拼接。

验证集错误分析必须按以下类别统计：

- 细裂缝漏检；
- 断裂裂缝；
- 交叉和分支裂缝；
- 低对比度裂缝；
- 粗裂缝边界偏移；
- 沥青纹理和阴影误检。

主选模指标使用裂缝 IoU，次级指标使用裂缝 Recall 和前景 F1；mIoU 不能单独决定模型优劣。

## 3. 候选模块调研与优先级

### P1：多尺度上下文

**候选**：ASPP、PPM/SPPM、轻量 LR-ASPP。

**适用问题**：裂缝宽度变化、长裂缝与局部纹理同时存在、断裂段之间缺少上下文。

**本地可复用实现**：

- `paddleseg.models.layers.ASPPModule`
- `paddleseg.models.layers.PPModule`
- `paddleseg.models.deeplabv3p`
- `paddleseg.models.pp_liteseg`
- `paddleseg.models.lraspp`

**优先方案**：先在 BiSeNetV2 的语义分支输出或 BGA 后端加入一个轻量 PPM/SPPM；ASPP 作为第二候选。不要一开始使用完整 DeepLabV3+ 解码器，以免参数量和延迟大幅增加。

ASPP 通过不同空洞率并行获取多尺度上下文，PPM 通过不同池化尺度聚合全局上下文；二者都已有成熟论文和 PaddleSeg 实现，可作为模块级消融。ASPP 的依据是 DeepLabv3，PPM 的依据是 PSPNet。

### P2：边界监督

**候选**：Boundary Loss、GSCNN 风格边界分支、PaddleSeg 已有边界相关损失。

**适用问题**：细裂缝边缘偏移、掩膜变粗、裂缝与沥青纹理粘连。

**本地可复用实现**：

- `paddleseg.models.gscnn` 中已有边界分支和 Gated Spatial Conv 参考实现；
- `RelaxBoundaryLoss` 和 `OhemEdgeAttentionLoss` 已在前期试验中出现，应保留历史结果，不重复作为首轮候选；
- Boundary Loss 官方实现为 `LIVIAETS/boundary-loss`，需要先验证 Paddle 兼容性。

**优先方案**：先跑 GSCNN 风格边界辅助监督的最小接入；若官方实现与当前训练入口不兼容，再使用 Boundary Loss 的公开实现做适配验证。边界分支必须单独报告边界质量，不能只看 mIoU。

### P3：连通性监督

**候选**：soft-clDice。

**适用问题**：长裂缝断裂、分支断开、中心线不连续。clDice 针对管状和网络状结构，直接约束预测掩膜与骨架的连通性，和裂缝循迹任务的下游需求相符。参考：<https://arxiv.org/abs/2003.07311>。

**使用方式**：

- 第一阶段只测试 `CE + Dice + λ * soft-clDice`；
- λ 先使用小权重网格，例如 0.05、0.10、0.20；
- 只保留能提升裂缝 IoU、Recall 或轨迹连通率的配置；
- 若训练不稳定或骨架运算导致服务器耗时明显增加，则停止该方向。

clDice 的风险是它可能改善连通性却牺牲 Precision，因此必须同时记录误检率和裂缝前景 F1。

### P4：轻量注意力

**候选顺序**：Coordinate Attention → CBAM → SE。

Coordinate Attention 将空间位置信息编码进通道注意力，适合细长、方向性明显的裂缝，并以移动网络为目标设计。参考：<https://arxiv.org/abs/2103.02907>。CBAM 顺序使用通道和空间注意力，适合做通用的第二注意力对照。参考：<https://arxiv.org/abs/1807.06521>。

**插入位置**：

- 第一候选：semantic branch 的最后一级特征；
- 第二候选：BGA 输出、最终 `SegHead` 之前；
- 不在 detail branch 的每个 stage 全部插入，避免延迟和参数量失控。

注意力模块只作为补充候选，不能在 P1/P2/P3 尚未诊断清楚前大量堆叠。

### P5：不进入首轮的模块

- 大型 Transformer 编码器；
- 完整 OCR/HRNet 替换；
- 复杂 CRF 后处理；
- 多模型集成；
- 测试时增强；
- 未有官方实现或无法复现实验协议的“新模块”。

这些方向会掩盖 BiSeNetV2 改进的真实原因，也不利于 Jetson Nano 部署。

## 4. 首轮消融矩阵

所有首轮实验固定：原始 Crack500 划分、输入尺寸、训练预算、数据增强、随机种子 42 和选模规则。每轮只改变一个主要因素。

| Run | 配置 | 目的 | 保留条件 |
|---|---|---|---|
| A0 | BiSeNetV2（默认 CE） | 重新确认基线 | 指标脚本和 checkpoint 正确 |
| A1 | OHEM + Dice + EMA | 复现当前最佳候选 | 作为当前控制组 |
| A2 | A1 + 轻量 PPM/SPPM | 补充多尺度上下文 | 验证裂缝 IoU 或 mIoU 提升 |
| A3 | A1 + ASPP | 与 PPM 对照 | 提升必须覆盖额外开销 |
| A4 | A1 + 边界辅助监督 | 改善细裂缝边界 | 裂缝 IoU 或边界 F1 提升 |
| A5 | A1 + soft-clDice | 改善裂缝连通性 | 连通率和 Recall 提升，Precision 不明显下降 |
| A6 | A1 + Coordinate Attention | 轻量位置建模 | 精度提升且延迟增加不超过 15% |
| A7 | A1 + CBAM | 注意力对照 | 仅在 A6 无效时运行 |

单模块筛选门槛：

- 裂缝 IoU 至少提升 1.0 个百分点，或；
- Recall 至少提升 1.0 个百分点，或；
- 前景 F1 至少提升 0.8 个百分点；
- 参数量增加不超过 20%，目标设备延迟增加不超过 15%。

未达到门槛的模块不进入组合实验。

## 5. 组合实验与最终候选

只组合首轮通过门槛的模块，最多保留两个主改动，避免形成难以解释的堆叠模型。

推荐组合顺序：

1. `A1 + PPM/SPPM` 与 `A1 + Boundary`；
2. `A1 + PPM/SPPM + Boundary`；
3. `A1 + soft-clDice`；
4. `A1 + PPM/SPPM + soft-clDice`；
5. 若注意力单模块明显有效，再测试 `A1 + Coordinate Attention + Boundary`。

必须增加一个参数匹配对照：仅增加通道数或重复卷积层，使参数量接近最终模型，但不加入所选模块。这个对照用于排除“只是模型变大”的解释。

最终候选冻结规则：

- 先按验证集裂缝 IoU 排名；
- 若差距小于 0.5 个百分点，按 Recall、前景 F1 和延迟排序；
- 只在最终候选冻结后对测试集评估一次；
- 目标是 mIoU > 0.7760，裂缝 IoU > 0.5819，前景 F1 > 0.7356；
- 若没有达到目标，论文写“具有竞争力”或“精度与实时性的折中优势”，不强行写 SOTA。

## 6. 运行顺序

| 阶段 | 工作 | GPU 成本 | 决策 |
|---|---|---:|---|
| M0 | 数据、指标、单 batch 和短时过拟合检查 | 0.5 小时以内 | 通过后继续 |
| M1 | A0/A1 基线复现与验证集选模 | 1 小时 | 复现失败先修环境 |
| M2 | 验证集错误类型和裂缝连通性统计 | 0.5-1 小时 | 确定主短板 |
| M3 | A2-A7 单模块筛选 | 4-8 小时 | 只保留通过门槛的模块 |
| M4 | 通过模块的两两/三模块组合 | 4-8 小时 | 冻结 1-2 个最终候选 |
| M5 | 参数匹配对照和最终测试 | 2-4 小时 | 判断是否超过参考值 |
| M6 | 三随机种子和 Jetson 延迟 | 4-8 小时 | 形成论文主表 |
| M7 | 自建数据与机器人闭环 | 现场测试 | 形成系统实验 |

## 7. 服务器存储规范

服务器已核实：

- `/hy-tmp` 挂载于独立数据盘，总容量 100 GB；
- 当前 `/hy-tmp` 使用约 36 GB，可用约 65 GB；
- `/hy-tmp/crack_project` 当前约 19 GB；
- 原先占用系统盘的 `/root/.cache/pip` 约 11 GB，已迁移至 `/hy-tmp/.cache/pip`；
- `/root/.paddleseg/pretrained_model` 约 363 MB，已迁移至 `/hy-tmp/.paddleseg/pretrained_model`；
- 两个原路径均已建立软链接，现有程序路径不变。

后续统一使用：

```text
/hy-tmp/crack_project/data
/hy-tmp/crack_project/experiments
/hy-tmp/crack_project/logs
/hy-tmp/crack_project/cache
/hy-tmp/.cache/pip
/hy-tmp/.cache/huggingface
/hy-tmp/.paddleseg
```

服务器环境变量已写入：

```text
/hy-tmp/crack_project/storage_env.sh
```

启动训练前加载该文件，避免 pip、Hugging Face 和 PaddleSeg 缓存再次写入系统盘。

## 8. 论文证据链

主论文只保留以下三类结果：

1. BiSeNetV2 基线与最终优化模型；
2. 通过门槛的 2-3 个模块及其组合消融；
3. 与已完成开源模型结果的对比，以及参数量和部署速度。

附录保留失败模块、兼容性问题、参数匹配对照、详细训练配置和错误案例。SCSegamba 不再作为当前对照或消融模型。

## 9. 最终检查清单

- [ ] 基线统一命名为 BiSeNetV2，CE 只作为默认训练配置
- [ ] 每个模块先做单因素验证集消融
- [ ] 只保留 2-3 个有效模块进入论文主表
- [ ] 有参数匹配对照
- [ ] 测试集只在最终冻结后使用
- [ ] 记录裂缝 IoU、Recall、Precision、前景 F1、mIoU
- [ ] 记录 Params、FLOPs、FPS、延迟和显存
- [ ] 服务器数据、缓存、日志和 checkpoint 全部位于数据盘
- [ ] 最终模型完成自建数据和机器人闭环验证

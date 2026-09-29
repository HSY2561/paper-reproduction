# Crack500 视觉分割基线实验审核报告

**报告日期：** 2026-09-21  
**报告状态：** 2026-09-21 历史训练快照；2026-09-23 依据原始日志追加校正，后续记录见文末

## 1. 审计结论

截至 2026-09-21 23:08，终端回执显示没有正在运行的训练进程。

本轮不能认定 SegFormer-B0 和 PIDNet-S 都已完成训练：

- **BiSeNetV2：已完成 18,000 iterations，并生成最佳权重。**
- **SegFormer-B0：未完成。** 初始运行到 12,000 iter 后进入验证；从 iter_9000 恢复的续跑保存了 iter_12000，并继续训练至 14,600/18,000 后停止。没有 18,000 iter 完成标记；现存日志未记录可确认的退出原因。**
- **PIDNet-S：未完成。首次反向传播阶段出现 `CUDNN_STATUS_EXECUTION_FAILED`，没有生成有效 checkpoint。**

因此，当前只能把 SegFormer-B0 的 3,000 次验证结果作为阶段性结果，不能把它作为最终完整训练结果；PIDNet-S 暂不能参与公平的最终模型排名。

## 2. 统一实验协议

| 项目 | 设置 |
|---|---|
| 数据集 | Crack500 裁块数据 |
| 训练集 | 1,896 张 |
| 验证集 | 348 张 |
| 测试集 | 1,124 张（本轮尚未完成测试集最终评估） |
| 数据根目录 | `D:\Crack500_data_u8` |
| 输入裁剪 | 400×400 |
| 类别数 | 2（背景、裂缝） |
| 随机种子 | 42 |
| 训练迭代 | 18,000 |
| 验证间隔 | 3,000 iterations |
| GPU | NVIDIA GeForce RTX 5060，约 8 GB 显存 |
| Paddle | 3.2.2 |
| Python | 3.10.21 |
| PaddleSeg | `D:\论文复现\external\PaddleSeg-2.10.0` |

## 3. BiSeNetV2（已完成）

**日志：** `D:\论文复现\experiments\crack500_baseline\bisenetv2_s42\train.log`  
**最佳权重：** `D:\论文复现\experiments\crack500_baseline\bisenetv2_s42\best_model\model.pdparams`  
**完成迭代：** 18,000/18,000

| 指标 | 验证集结果 |
|---|---:|
| mIoU | 0.7678 |
| 背景 IoU | 0.9703 |
| 裂缝 IoU | 0.5652 |
| 背景 Precision | 0.9798 |
| 裂缝 Precision | 0.7986 |
| 背景 Recall | 0.9901 |
| 裂缝 Recall | 0.6592 |
| 宏平均 Dice（日志值） | 0.8536 |
| 参数量 | 2,328,346 |
| FLOPs | 4,934,732,616 |

BiSeNetV2 的主要短板是裂缝 Recall 偏低，说明细裂缝和弱裂缝存在漏检问题。

## 4. SegFormer-B0（部分完成）

**日志：** `D:\论文复现\experiments\crack500_baseline\segformer_b0_s42\train.log`  
**错误/标准输出：** `D:\论文复现\experiments\crack500_baseline\segformer_b0_s42\train.err`  
**当前最佳权重：** `D:\论文复现\experiments\crack500_baseline\segformer_b0_s42\best_model\model.pdparams`  
**初始运行已保存 checkpoint：** iter_3000、iter_6000、iter_9000；续跑目录另保存 iter_12000。
**续跑状态：** 日志最后到 14,600/18,000；无最终 18,000 iter checkpoint。

### 4.1 阶段性验证结果

| 运行/迭代 | mIoU | 裂缝 IoU | 裂缝 Precision | 裂缝 Recall | 宏平均 Dice（日志） | 前景 F1（按裂缝类 P/R 计算） |
|---|---:|---:|---:|---:|---:|---:|
| 初始运行 3,000 | 0.8136 | 0.6515 | 0.8094 | 0.7696 | 0.8884 | 0.7890 |
| 初始运行 6,000 | 0.8104 | 0.6447 | 0.8378 | 0.7367 | 0.8859 | 0.7840 |
| 初始运行 9,000 | 0.8109 | 0.6455 | 0.8422 | 0.7343 | 0.8863 | 0.7846 |
| 从 9,000 恢复的续跑 12,000 | 0.8212 | 0.6651 | 0.8384 | 0.7629 | 0.8937 | 0.7989 |

截至现存日志，续跑 12,000 iter 的验证 mIoU 0.8212 是当前最佳记录；该次续跑随后到 14,600 iter 后停止，未完成预定训练预算：

- Class IoU：`[0.9757, 0.6515]`
- Class Precision：`[0.9863, 0.8094]`
- Class Recall：`[0.9892, 0.7696]`
- 该次验证的宏平均 Dice：0.8884；按裂缝类 Precision/Recall 计算的前景 F1：0.7890

### 4.2 当前判断

SegFormer-B0 续跑在 12,000 iter 的阶段性验证指标高于现有 BiSeNetV2 验证指标，但未完成训练且没有统一测试回执，不能用于最终排名，也不改变用户选定的 BiSeNetV2 工程消融基线。

## 5. PIDNet-S（训练失败）

**日志：** `D:\论文复现\experiments\crack500_baseline\pidnet_s_s42\train.log`  
**配置：** `D:\论文复现\experiments\crack500_baseline\configs\pidnet_s_crack500.yml`  
**训练状态：** 未完成，未生成 checkpoint

日志中的实际错误为：

```text
OSError: (External) CUDNN error(5000), CUDNN_STATUS_EXECUTION_FAILED.
```

错误发生在第一次反向传播的卷积梯度计算阶段。该结果不能用于模型性能比较。建议下一轮先采用更保守的 GPU 设置：

- batch size 从 4 降为 2 或 1；
- 保持 400×400 输入不变；
- 必要时启用官方 Paddle AMP；
- 单独确认 PIDNet-S 的官方预训练权重加载和前向/反向测试；
- 训练成功后仍使用相同 Crack500 划分和指标。

## 6. 当前模型比较

| 模型 | 训练状态 | 当前可用结果 | 是否可作为最终排名依据 |
|---|---|---|---|
| BiSeNetV2 | 完成 18,000 次 | 验证 mIoU 0.7678，裂缝 IoU 0.5652，Recall 0.6592，宏平均 Dice 0.8536，前景 F1 0.7222 | 可核验训练/验证；无测试回执，不能最终排名 |
| SegFormer-B0 | 续跑到 14,600/18,000；未完成 | 阶段性最佳验证 mIoU 0.8212，裂缝 IoU 0.6651，Recall 0.7629，宏平均 Dice 0.8937，前景 F1 0.7989 | 不可最终排名；缺完整训练和统一测试 |
| PIDNet-S | 反向传播阶段 CUDNN 失败 | 无有效指标 | 不可以 |

## 7. 阶段性基线建议

本报告当时曾将 SegFormer-B0 列作候选基线。按用户后续决定，**BiSeNetV2 + CrossEntropyLoss 固定为工程消融基线**；SegFormer-B0 保留为未完成的公开模型对照，不改变基线选择。

论文中的最终基线结论应延后到以下条件满足后：

1. SegFormer-B0 完成 18,000 次训练并保存最终最佳权重；
2. PIDNet-S 通过降低 batch size 等方式成功完成训练；
3. 三个模型在 Crack500 测试集上使用同一评价脚本完成评估；
4. 同时记录参数量、FLOPs、显存和推理速度。

## 8. 下一步执行顺序

1. 从 SegFormer-B0 的 `iter_9000` 恢复，完成剩余训练并补齐验证结果；
2. 修复 PIDNet-S 的 CUDNN 配置，优先使用 batch size 2 重新启动；
3. 对三个模型的最佳权重统一运行验证集和测试集评估；
4. 形成最终模型比较表；
5. 再针对最终基线的主要短板开展 Dice/Focal、边界和多尺度模块的单模块消融实验。

## 9. 文件索引

- BiSeNetV2 实验目录：`D:\论文复现\experiments\crack500_baseline\bisenetv2_s42`
- SegFormer-B0 实验目录：`D:\论文复现\experiments\crack500_baseline\segformer_b0_s42`
- PIDNet-S 实验目录：`D:\论文复现\experiments\crack500_baseline\pidnet_s_s42`
- 实验配置目录：`D:\论文复现\experiments\crack500_baseline\configs`
- Crack500 数据目录：`D:\Crack500_data_u8`
- 官方 PaddleSeg 2.10.0：`D:\论文复现\external\PaddleSeg-2.10.0`

## 2026-09-23 日志校正（保留原审计记录）

本节按原始日志和 checkpoint 目录更正上文少量事实，不代表新增训练：

- BiSeNetV2 18,000 iter 的验证日志记录 Dice 0.8536，这是日志的两类宏平均 Dice；以前景裂缝类 P=0.7986、R=0.6592 计算的 F1 为 0.7222。两者不能混称。
- SegFormer-B0 初始运行到 12,000 iter 后开始验证。随后从 iter_9000 恢复的续跑目录保存了 iter_12000；该次验证为 mIoU 0.8212、裂缝 IoU 0.6651、P 0.8384、R 0.7629、宏平均 Dice 0.8937、前景 F1 0.7989。续跑日志随后到 14,600/18,000 后结束，没有完成标记；现有日志未给出可核实的退出原因。3,000 iter 的 0.8136 不是现存运行中的最佳验证 mIoU。
- 本次没有产生新模型训练、测试或消融结果；服务器 i-1.gpushare.com:64995 当前 TCP 检查失败。历史服务器汇总仍待原始产物核验，SOTA 和消融验收未完成。
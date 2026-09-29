# Source-frame-disjoint 数据划分与 SCSegamba 复核记录

- 日期：2026-09-25
- 服务器工程：`/hy-tmp/crack_project`
- 原始划分目录：`/hy-tmp/crack_project/data/Crack500_source_disjoint`
- 有效训练目录：`/hy-tmp/crack_project/data/Crack500_source_disjoint_valselect_u8_fixed_20260925`
- 独立测试目录：`/hy-tmp/crack_project/data/Crack500_source_disjoint_test_u8_fixed_20260925`
- 随机种子：`20260924`
- 标签编码：0/255（兼容官方 SCSegamba 读取器的 127 阈值）

## 划分规则

- 文件名为 `YYYYMMDD_HHMMSS_x_y` 时，以前两段 `YYYYMMDD_HHMMSS` 作为源拍摄帧组；其他命名按去除末两段裁块坐标后的 stem 分组。
- 源组总数：442
- 裁块总数：6,736
- manifest 行数：train 4,714 / val 674 / test 1,348
- 实际唯一图像文件：train 4,706 / val 674 / test 1,348
- train manifest 有 8 行重复文件名；训练实际按唯一目录文件读取 4,706 个图像文件，val/test 无重复文件名。
- 跨集合源组交集：`[]`

该划分已经在服务器生成 `train.txt`、`val.txt`、`test.txt` 和 `source_disjoint_metadata.json`。原始 1,896/348/1,124 划分用于历史基线；源帧互斥划分用于最终复核，不能把原始裁块结果直接称为 source-frame-disjoint 结果。

## 划分用途调整

SCSegamba 的 source-frame-disjoint 运行已完成，但按当前项目决策不再作为主比较模型。其结果仅保留作审计资料；本划分后续用于 BiSeNetV2 基线、BiSeNetV2 消融模型和开源对照模型的统一重跑。

## 已归档的 SCSegamba 运行

- 输出目录：`/hy-tmp/crack_project/experiments/scsegamba_source_disjoint_fixed_20260925`
- 训练日志：`/hy-tmp/crack_project/logs/scsegamba_source_disjoint_fixed_train_20260925.log`
- 评估日志：`/hy-tmp/crack_project/logs/scsegamba_source_disjoint_fixed_eval_20250925.log`
- 结果 JSON：`/hy-tmp/crack_project/logs/scsegamba_source_disjoint_fixed_result_20260925.json`
- 训练进度：50/50 epoch，已生成 50 个 checkpoint
- 验证选模：`checkpoint20.pth`
- 冻结权重：`checkpoint_best_fixed.pth`
- 验证阈值：0.5

| 指标 | 验证集 | source-frame-disjoint test |
|---|---:|---:|
| mIoU | 0.7850416 | **0.7957967** |
| 裂缝 IoU | 0.6009442 | **0.6218454** |
| Precision | 0.7082640 | 0.7327194 |
| Recall | 0.7986295 | 0.8042867 |
| 前景 F1 | 0.7507372 | 0.7668369 |

冻结后才评估独立 test，避免使用 test 选模。OCRNet-HRNet-W18 的 0.7760 来自原始固定 test 协议，与此处的 source-frame-disjoint test 不同；该结果不进入当前主模型比较。

## 无效旧运行

旧目录 `/hy-tmp/crack_project/experiments/scsegamba_source_disjoint_20260924` 的标签为 0/1，而官方读取器缩放后使用 127 阈值，导致掩码被读成全背景，训练日志中的 mIoU=0 无效。该目录仅保留作审计记录，不能进入论文结果表；以上表格只使用修正后的 0/255 标签运行。

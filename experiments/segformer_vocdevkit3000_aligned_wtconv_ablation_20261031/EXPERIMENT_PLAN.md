# VOCdevkit3000 SegFormer-B0 + Aligned + WTConv 域内消融

本轮固定最终候选为 SegFormer-B0 + Aligned + WTConv，并在用户新上传的 VOCdevkit3000 自建裂缝数据上重新训练。消融四组为：C0 原始 SegFormer-B0、C1 仅 Aligned、C2 仅 WTConv、C3 Aligned+WTConv。

## 数据

- 原始目录：`C:/Users/user/Desktop/VOCdevkit3000/VOCdevkit/VOC2007`。
- VOC 风格目录：`JPEGImages/`、`SegmentationClass/`、`ImageSets/Segmentation/`。
- 图像和标签共 2932 对，全部 448×448 RGB/L。
- train=2345，val=587，test 清单为空；本轮只训练/验证，不读取 test。
- 原标签值为 `{0,255}`，255 在 PaddleSeg 中是默认 ignore_index。因此在 D 盘建立只读派生 staging 数据，将 255 映射为前景 1，0 保持背景 0；不修改原始 Desktop 数据。
- train 前景比例约 5.1160%，val 前景比例约 5.1866%。

## 与 Crack500 协议

- RTX5060、Paddle 3.4、FP32、seed42、batch4。
- AdamW，learning rate 6e-5，PolynomialDecay `end_lr=0`，18,000 iter，CrossEntropyLoss，EMA，workers0。
- ResizeStepScaling 0.75–1.25、RandomPaddingCrop 400×400、RandomHorizontalFlip、RandomDistort、Normalize。
- 同一 W17 源码快照和 `MixVisionTransformer_B0_S4FFNControlled(hidden=1024)`，同一 ImageNet backbone 初始化。
- 普通/EMA 按 val mIoU 精确选模，冻结后保存 SHA256；不读取 test。

## 结构

| 组别 | 结构 | 参数量 |
|---|---|---:|
| C0 | SegFormerSimplify，alignment=none | 3,715,170 |
| C1 | SegFormerSimplify，alignment=full | 3,785,314 |
| C2 | SegFormerTen，alignment=none，candidate=wtconv | 3,653,346 |
| C3 | SegFormerTen，alignment=full，candidate=wtconv | 3,723,490 |

WTConv 采用当前工程的低频 AvgPool2D + 高频 depthwise 5×5 + pointwise/residual 实现，报告中称为本工程 WTConv candidate，不将其表述为原始小波论文实现。

## 完成状态（2026-11-01）

C0、C1、C2、C3 四组均完成 smoke、18,000 iter 正式训练、普通/EMA 验证选模、官方 val 和 587 张逐图精确评价，所有相关 exit_code=0。`results/workflow_state.json` 为 `stage=complete`、`test_used=false`。最终表格为 `results/voc_ablation_table.csv`，详细报告为 `VOCdevkit3000_域内消融报告_SegFormerB0_Aligned_WTConv_20261031.md`。

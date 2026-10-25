# A0 与六个公开模型：历史 Crack500 test 对照

整理日期：2026-10-25T15:52:38+08:00。本表读取已有结果，不重新训练或推理。当前基准是 **SegFormer-B0＋width128＋Aligned（A0）**；原始 SegFormer-B0 width256 是公开对照。

**本表不是尚待完成的统一后端终评。** 该表审计时统一评价记录为 `waiting_additional_B0_initialization_witness_control`，记录时间 `2026-10-25T15:52:20+08:00`，没有读取新的 test 图像。

Crack500 test 共1124张裁剪图。P、R、F1、IoU、mIoU均为百分数；IoU指裂缝前景，mIoU为背景和裂缝两类均值；参数量单位为百万。指标由全数据集混淆矩阵聚合，非逐图分数平均。

| 模型 | Precision | Recall | F1 | 裂缝IoU | mIoU | Params(M) |
|---|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | 72.8905 | 72.4478 | 72.6684 | 57.0702 | 76.9475 | 3.715170 |
| OCRNet-HRNet-W18 | 70.3819 | 75.9782 | 73.0731 | 57.5710 | 77.1509 | 12.109902 |
| PP-LiteSeg-STDC1 | 74.6053 | 69.7870 | 72.1157 | 56.3914 | 76.6260 | 8.150485 |
| PSPNet-ResNet50 | 72.1991 | 70.6710 | 71.4269 | 55.5535 | 76.1313 | 67.902052 |
| DeepLabV3P-ResNet50 | 72.2276 | 73.5601 | 72.8877 | 57.3412 | 77.0754 | 26.789874 |
| Fast-SCNN | 75.0411 | 61.4485 | 67.5680 | 51.0208 | 73.8029 | 1.441332 |
| SegFormer-B0 + width128 + Aligned (A0) | 74.5139 | 71.4484 | 72.9490 | 57.4171 | 77.1658 | 3.470434 |

完整机器可读结果：[CSV](../results/public_models_historical_test.csv)、[JSON](../results/public_models_historical_test.json)。退出码、权重SHA及来源索引：[核验记录](../evidence/public_audit.json)。

## 配方与完成证据

共同预算为18,000 iter、seed42、batch4、AdamW、初始学习率6e-5、end_lr=0、FP32及同一数据划分和增强。以下差异必须随表披露，不能描述为完全相同的训练策略。

| 模型 | 初始化 | Loss | EMA | 训练后端 |
|---|---|---|---|---|
| SegFormer-B0 | official MiT-B0 ImageNet | CE | True | Windows / Paddle 3.4 / RTX 5060 / FP32 |
| OCRNet-HRNet-W18 | random; HRNet_W18 pretrained=None | CE main/aux [1,.4] | True | Windows / Paddle 3.4 / RTX 5060 / FP32 |
| PP-LiteSeg-STDC1 | official PP_STDCNet1 pretrained | OHEM three heads [1,1,1] | True | Windows / Paddle 3.4 / RTX 5060 / FP32 |
| PSPNet-ResNet50 | random; pretrained:null | OHEM main/aux [1,.4] | False | WSL2 Ubuntu24 / Paddle 3.3.0.dev20251209 / RTX 5060 / FP32 |
| DeepLabV3P-ResNet50 | random; pretrained:null | OHEM single head | False | WSL2 Ubuntu24 / Paddle 3.3.0.dev20251209 / RTX 5060 / FP32 |
| Fast-SCNN | random; pretrained:null | OHEM main/aux [1,.4] | False | WSL2 Ubuntu24 / Paddle 3.3.0.dev20251209 / RTX 5060 / FP32 |
| SegFormer-B0 + width128 + Aligned (A0) | official MiT-B0 ImageNet | CE | True | Windows / Paddle 3.4 / RTX 5060 / FP32 |

- 六原始模型都有18,000步日志及真实test退出码0。其中五个有真实训练退出码0；PP-LiteSeg训练退出码未记录，依据保留的最终日志/checkpoint完成记录确认，不能补写0。
- 七个历史冻结checkpoint的SHA已复核。A0已有历史test统计，但本审计未找到其单独的历史test进程退出码，不以指标文件替代退出回执。
- PP-LiteSeg表中**8,150,485为完整实例化总参数（含辅助头）**：本次使用正式配置 `experiments/retrain_wave4_20261007/configs/ppliteseg_stdc1_crack500_formal.yml` 在CPU实例化，仅把预训练下载入口置None后加载正式冻结权重；checkpoint missing/extra均为空，250个参数张量、250个state keys，按 `sum(int(np.prod(p.shape)) for p in model.parameters())` 直接计数，未访问数据、GPU或推理。原日志 `experiments/retrain_wave4_20261007/logs/ppliteseg_stdc1_crack500_formal_train.stdout.log:663` 的 `Total Params: 8039122` 则是 `paddle.flops` 切换eval后、只执行seg_heads[0]得到的历史参与层计数；两者口径不同。旧汇总8,150,485曾缺直接计数凭据，现已由CPU实测补齐，不能再将8,039,122误替代为模型总参数。完整回执字段见public_audit.json。其官方原结构自带UAFM/SPPM，本地数值等价兼容修补另有记录。
- 三个WSL模型为PSPNet-ResNet50、DeepLabV3P-ResNet50、Fast-SCNN。旧Windows smoke失败不应覆盖它们后续成功训练的事实。
- A0训练使用Windows/Paddle3.4，既有test使用Paddle3.2.2兼容环境；不能将它们视作同一后端测速。
- Fast-SCNN仅1.441332M，比A0更小，因此本表不支持“所有模型中参数最少”。A0的F1和裂缝IoU也未超过OCRNet，不能宣称全指标最佳或领域SOTA。

## 开源来源

六个公开模型均来自PaddleSeg官方实现，并适配为背景/裂缝两类：

- [SegFormer官方配置与说明](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs/segformer)
- [OCRNet官方配置与说明](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs/ocrnet)
- [PP-LiteSeg官方配置与说明](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs/pp_liteseg)
- [PSPNet官方配置与说明](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs/pspnet)
- [DeepLabV3P官方配置与说明](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs/deeplabv3p)
- [Fast-SCNN官方配置与说明](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs/fastscnn)

A0采用公开SegFormer及Aligned模块构建；不声称基础模型或模块由本项目首次提出。所有本地来源路径在JSON中均为工作区相对的溯源标识，不包含个人绝对路径。PP-LiteSeg总参数口径已补CPU核验；部署有效参数、延迟和显存仍需统一profile，参数减少不等于加速。更多限制见[limitations.md](limitations.md)。

补充状态：共同初始化B0补控已于16:15完成，统一评价已启动；当前动态快照见[状态JSON](../results/current_status.json)。本页历史test数字保持原来源，不预填新终评。

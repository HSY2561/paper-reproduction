# SegFormer-B0 新基线与后续论文实验规划

日期：2026-10-06

## 1. 方向调整与当前结论

本阶段把 **SegFormer-B0** 设为精度优先的主视觉基线，编号为 S0。BiSeNetV2/C4 保留为轻量化部署对照，不再把它作为后续结构搜索的主线。这样符合论文的主线：先用统一分割模型得到裂缝概率图，再提取裂缝中心线、宽度和端点，完成四轮差速底盘的粗循迹；底盘停靠后使用同一概率图和局部重观测结果，驱动三轴机构完成精定位与修复。

现有 Crack500 本地结果只作为历史参考，不能直接替代新自建数据正式结果。RTX5060 上已经得到：SegFormer-B0 验证 mIoU 0.826676、裂缝 IoU 0.675882、F1 0.806599；OCRNet-HRNet-W18 验证 mIoU 0.818869、裂缝 IoU 0.661978、F1 0.796615；BiSeNetV2-C4 验证 mIoU 0.815506、裂缝 IoU 0.654863、F1 0.791440。原始 Crack500 test 上 SegFormer-B0 mIoU 0.772686、裂缝 IoU 0.576876、F1 0.731670，OCRNet mIoU 0.770546，BiSeNetV2-C4 mIoU 0.761899。它们显示 SegFormer 是目前合理的精度基线，但不构成领域 SOTA 证明。

旧 OCRNet 边界 F1 记录存在计算错误：旧文件把边界匹配计数直接组合成了错误值。新阶段统一使用全数据集匹配计数先计算 precision、recall，再用 harmonic mean 计算 boundary-F1；历史数字只保留为审计记录，后续报告不再引用错误值。

## 2. 数据角色与正式协议

自建数据位于 `D:\论文复现\PaddleSeg-release-2.8\dataset\custom`，通过 `C:\custom_crack` 读取，当前清单为 train 1418 对、val 177 对、test 177 对，缺失路径为 0。自建 test 在模型选择和消融阶段封存，只有最终模型冻结后才允许使用。Crack500 原始 train/val/test 用于公开数据集最终对比；source-frame-disjoint 划分用于检查同一原始帧切片泄漏造成的虚高结果。

所有候选固定：RTX5060、FP32、batch=4、AdamW、初始学习率 6e-5、PolynomialDecay end_lr=0、weight decay 0.01、EMA、seed=42 起步、18,000 iter、400×400 训练裁剪、验证原图尺寸恢复。验证选出的 checkpoint 记录普通权重和 EMA 权重的 mIoU、SHA256、训练日志和数据清单哈希。任何结构候选都不能访问 test 进行筛选。

每次评估统一报告 mIoU、裂缝 IoU、Precision、Recall、F1、2 像素边界 F1、空预测数、参数量、FLOPs 或同机延迟。边界定义为前景掩码减 3×3 腐蚀得到的内边界，采用欧氏距离 2 像素的对称匹配；空预测、空 GT 采用显式规则处理。边界 F1 必须由全局匹配计数计算，不能先逐图平均后再平均。

## 3. 新编号与实验顺序

- **S0：SegFormer-B0 原始官方结构**，沿用官方 MixVisionTransformer-B0 和 SegFormer 解码头。现有 Crack500 运行已经使用 [0.75, 1.25] 的尺度增强与 400 裁剪，因此不能把相同的 C4 作为新模块重复消融。
- **S0-custom：自建数据正式基线**，先完成 20 iter smoke，再完成 seed42 的 18,000 iter 正式训练。验证集只用 177 张自建 val。
- **D0：错误诊断与输入恢复审计**，比较 S0 与已有 BiSeNetV2/C4 的逐图错误、细裂缝、宽裂缝、阴影、纹理密集和低对比度分层。D0 不是新模型，不改变代码。
- **S1/A：细节恢复模块**，只在 D0 证明高分辨率细节或边界恢复是主要短板时运行。优先复用 PaddleSeg 官方 UAFM/PP-LiteSeg 解码器组件，接在 SegFormer 多尺度特征融合到输出头之间。目标是减少细裂缝漏检、断裂和边界偏移。
- **S2/B：上下文/多尺度模块**，只在 D0 证明长裂缝连续性、低对比度或远距离上下文不足时运行。优先复用 PaddleSeg 官方 PPContextModule 或同版本成熟上下文层，接在融合特征或解码器瓶颈，不同时改变输入增强和损失。
- **S3：A+B 联合**，只有 A、B 分别通过门槛才允许运行；联合结果必须同时超过 S0 和最佳单模块，不能因模块数量更多而直接采用。
- 每个模块先做单模块 smoke 和 seed42 正式训练；单模块通过后才补 seed43/44。若两个方向均失败，结束结构堆叠，转向数据流、标注和机器人闭环，不继续无证据地增加注意力、Transformer、Mamba、动态卷积或损失项。

## 4. 模块选择依据

### A：高分辨率细节/边界恢复

它解决的是 SegFormer 低分辨率编码特征在细裂缝、裂缝端点和窄边界上的信息损失。修改位置是 SegFormer 解码头的多尺度融合之后、最终分类头之前，使用官方可复用的特征融合/注意力层，不改主干。预期收益是裂缝 Recall、裂缝 IoU、边界 F1 和中心线连续性提升。代价是增加融合卷积、显存和推理延迟；风险是把路面纹理也当作细裂缝，造成 Precision 下降，或与现有 MLP 融合重复。

### B：多尺度上下文/目标级聚合

它解决长裂缝跨区域连续性、阴影遮挡和低对比度区域中局部纹理不够判别的问题。修改位置是融合特征的瓶颈或解码器末端，用官方上下文模块一次性引入多尺度感受野。预期收益是长裂缝 Recall、裂缝 IoU 和骨架连续性提升。代价是额外池化/卷积和显存，可能降低速度；风险是过度平滑窄裂缝、引入大范围纹理误检。A、B 都必须保持独立变量，不能在同一实验同时改变增强、输入尺寸和损失。

不优先尝试的方向：直接换 Mamba/Transformer、堆叠多个注意力、同时加入边界损失和动态卷积、把服务器 RTX4090 探索结果当作本地精度。它们会混淆原因，且不符合先单变量、再联合的论文消融逻辑。

## 5. 预注册筛选门槛

相对 S0-custom，A 通过条件：裂缝 IoU 至少增加 0.005，F1 至少增加 0.003，Recall 下降不超过 0.005，边界 F1 不下降。B 通过条件：裂缝 IoU 至少增加 0.005，F1 至少增加 0.003，Recall 下降不超过 0.005，且长裂缝连续性指标不下降。S3 只能在 A、B 均通过后启动，并要求联合模型超过 S0 和最佳单模块。所有门槛只在自建 val 的预先定义协议上判断；通过后才补多种子、公开 Crack500 对照和 test。

## 6. 误差诊断任务

在 S0-custom 完成后，保存每张 val 图的混淆矩阵、预测掩码、GT、空预测标志、边界匹配数和骨架连通指标。重点形成三类证据：边界改善但 IoU 下降；边界和 IoU 同时改善；相对 S0 完全无重叠。再按标签统计细裂缝、宽裂缝、低对比度、阴影和纹理密集样本。检查 400×400 裁剪是否删除窄裂缝，缩放与标签插值是否断裂，验证去 padding 和奇数尺寸恢复是否偏移，阈值化是否造成区域退化。只有诊断明确指向某一瓶颈，才启动 A 或 B。

## 7. 训练与产出顺序

1. 完成 S0-custom smoke，并保存真实 exit code、日志和 checkpoint。
2. 完成 S0-custom seed42 18,000 iter，验证选模、冻结、SHA256 和逐图诊断。
3. 运行 D0 误差审计，更新本计划和模型消融文档。
4. 依据 D0 只选 A 或 B 中最有证据的一项，进行单模块 smoke、18,000 iter、统一验证。
5. 单模块通过才补 seed43/44；两模块都通过才做联合。
6. 最终模型冻结后，统一重跑 Crack500 public test、SegFormer S0、BiSeNetV2/C4、OCRNet-HRNet-W18 及可复现的其他公开模型。
7. 最后做 source-frame-disjoint 复核、相机—底盘—三轴标定、四轮循迹、三轴定位、修复覆盖率/未覆盖率/溢出率/均匀性实验。

## 8. 论文叙事

论文将问题定义为“面向裂缝循迹与精细修复的统一视觉分割与几何定位”。S0 提供统一概率图；A 或 B 只保留验证有效的 1–2 个成熟模块。论文消融沿用师兄论文的逻辑：基线、单模块 A、单模块 B、联合 A+B，再做公开模型对比和真实机器人验证。师兄论文的 ASPP、重设计解码器和具体数字不复制；这里只借鉴实验组织方式。若结构没有超过强对照，论文应诚实报告数据流限制、负结果和机器人实际收益，不能把未达到的 SOTA 写成结论。

## 9. 当前文件与可追溯性

新波工程：`D:\论文复现\experiments\local_segformer_wave1_20261006`；源码通过 `C:\local_wave1src` 指向独立 PaddleSeg-2.10.0；Python 为 `C:\Users\user\算法实验\venv_bisenet\Scripts\python.exe`。自建数据清单和计数记录在 `provenance\custom_manifest.json`。S0 smoke 完成后必须写入 `results\S0_smoke_exit.json`；正式训练另写 `results\S0_seed42_exit.json`，不覆盖 smoke 证据。

## 2026-10-07 自建数据审计与 S0 正式训练状态

自建数据的 20 iter smoke 首次未产生训练步，原因是 smoke 清单仅 2 对图像，而官方训练器 `drop_last=True`、batch=4；该进程没有退出回执，已标记为 `aborted_no_progress`，不作为实验结果。修正为 8 对训练图像、2 对验证图像后，官方训练入口真实退出码为 0，完成训练、验证和 checkpoint 保存。

对 1418/177 train/val 图像进行尺寸、标签值、像素哈希和 D4 旋转/翻转审计后，发现 18 个 train/val 图像是完全相同的图像副本或 D4 等价副本。原始清单保留用于追溯；正式 S0 使用移除训练侧重复图像的 source-disjoint 清单：train=1400、val=177。测试清单没有读取，原始数据没有修改。审计证据位于 `experiments\local_segformer_wave1_20261006\provenance\custom_train_val_audit.json`、`custom_duplicate_transform_audit.json` 和 `custom_source_disjoint_manifest.json`。

S0 seed42 已在 RTX5060 上以 18,000 iter、batch4、FP32、EMA、官方 SegFormer-B0 开始正式训练。训练日志为 `experiments\local_segformer_wave1_20261006\logs\S0_seed42_train.log`，启动回执为 `results\S0_seed42_launch.json`，完成后写入 `results\S0_seed42_exit.json`。在训练未完成前不启动任何模块消融和不访问 test。


## 2026-10-07 当前显卡重跑波次

正式重跑工程见 xperiments/retrain_wave4_20261007。RTX5060 上 SegFormer、OCRNet、BiSeNetV2-C4 smoke 通过，已进入18k队列；PP-LiteSeg、PIDNet、DeepLabV3P 在当前 Paddle/cuDNN 环境的反向 smoke 失败，保留真实退出码和日志，不纳入成功比较。自建 S0 seed42 已完成验证审计，结果和证据见 wave1 工程。

## 2026-10-07 Crack500 SegFormer 主线执行结论

已按本计划在 Crack500 上完成 SegFormer-B0+UAFM_SpAtten 的 18,000 iter 正式训练和独立验证。UAFM EMA iter=11000 的 val mIoU=0.8263、裂缝 IoU=0.6752、Recall=0.7924、F1=0.8059；SegFormer-B0 基线 val mIoU=0.8280、裂缝 IoU=0.6722、Recall=0.7961、F1≈0.8030。UAFM 未达到预注册门槛，故不访问 test。

PPContextModule 在相同源码、GPU 和训练设置下的反向 smoke 真实退出码为1，错误为 `CUDNN_STATUS_INTERNAL_ERROR`；UAFM+PPContext 不启动。按补充公开模型计划复核的 PP-LiteSeg-STDC1、PIDNet-S、DeepLabV3P-ResNet50 smoke 真实退出码也均为1，分别出现 `CUDNN_STATUS_EXECUTION_FAILED` 或 `CUDNN_STATUS_INTERNAL_ERROR`，没有完整训练或 test 结果。

因此本轮停止结构堆叠：SegFormer-B0 是当前精度基线，OCRNet-HRNet-W18 是已完成的原始强对照；C4/BiSeNetV2 从主线移除。当前没有通过验证门槛的 SegFormer 优化模型，不能声称 SOTA。结果证据见 `experiments/retrain_wave4_20261007/results/crack500_segformer_ablation_summary.json`。

## 2026-10-07 主线执行更新：PPContext 已完成兼容性验证，结构搜索停止

PPContext 的 RTX5060 兼容训练已完成 18,000 iter。采用官方模块的数值等价布局修补后，独立验证仍低于 SegFormer-B0：EMA 最优 mIoU=0.824902、裂缝 IoU=0.672754、F1=0.804367、2px boundary-F1=0.577581；基线分别为 0.828021、0.678174、0.808228、0.589412。PPContext 未通过预注册门槛，test 不访问，UAFM+PPContext 不启动。当前主线不存在通过筛选的 SegFormer 结构优化模型，论文应报告该负结果及兼容性证据。PP-LiteSeg/PIDNet/DeepLab 当前环境的正式配置 smoke 仍失败，不能伪造完整对比结果。

## 2026-10-08 PP-LiteSeg-STDC1 正式训练与验证结论（覆盖此前 smoke 状态）

PP-LiteSeg-STDC1 已在 RTX5060、PaddleSeg 官方结构、Crack500 train/val 上完成 18,000 iter 正式训练。正式配置为 batch=4、FP32、seed=42、AdamW、初始学习率 6e-5、PolynomialDecay end_lr=0、训练尺度 0.75–1.25、400×400 crop；仅使用 train/val 选模，没有访问 test。训练日志确认 `iter 18000/18000`，普通 best_model 与 EMA best_model 均在 iter=8000 保存；普通模型的验证 mIoU 更高，因此选择普通 best_model。

独立官方 `tools/val.py` 对 348 张验证图复核，并用同一官方推理接口计算 2px 边界指标：

| 模型 | mIoU | 裂缝 IoU | Precision | Recall | F1 | 2px boundary-F1 |
|---|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 基线 | 0.828021 | 0.678174 | 0.828269 | 0.789134 | 0.808228 | 0.589412 |
| PP-LiteSeg-STDC1（普通 best，iter 8000） | 0.820509 | 0.664104 | 0.825001 | 0.772995 | 0.798152 | 0.564314 |
| PP-LiteSeg-STDC1（EMA best，iter 8000） | 0.819100 | 0.661400 | 0.827200 | 0.767500 | 0.796233 | 未单独计算 |

普通 best 相对 SegFormer-B0 的 mIoU、裂缝 IoU、Recall、F1 和边界 F1 分别下降 0.007511、0.014070、0.016139、0.010076 和 0.025098，未达到预注册验证门槛。因此 PP-LiteSeg 作为原始公开模型的验证负结果保留，不冻结为最终候选，也不访问 Crack500 test；当前不能声称 SOTA。模型参数量为 8,039,122，FLOPs 为 3,567,881,920，RTX5060 验证 batch_cost 约 0.0184 s。

第一次 PP-LiteSeg 训练回合因磁盘空间耗尽而中止于最后保存阶段；清理临时产物后，正式 batch4 smoke 真实退出码为 0，随后完整 18,000 iter 训练完成。训练启动器没有写出独立进程退出 JSON，因此正式训练 exit_code 保留为未知；完成依据是 `iter_18000`、最终训练日志和独立 val.py 回执。完整选择、门槛和 SHA256 记录见 `experiments/retrain_wave4_20261007/results/ppliteseg_stdc1_crack500_formal_selection.json` 与 `ppliteseg_stdc1_crack500_formal_exit.json`。

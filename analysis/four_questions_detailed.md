# 四个追问的证据化分析

本文档对应用户关于修补指标、ASPP/Decoder、HrSegNet 基线和修补任务生成算法的四个问题。页码均指 `Climbing robot.pdf` 的 PDF 页码；代码路径相对于仓库根目录 `PaddleSeg-release-2.8`。

## 1. CR、UR、OR/OFR、CUI 是否凭空捏造

这些量不是从一个总分“拍脑袋”得到的，而是由两个二值掩膜和一个 ROI 计算的面积比例：修补前裂缝掩膜 `C`、修补后胶体掩膜 `S`，以及人工确认的 ROI。设 `A_c` 为裂缝总面积，`A_cov` 为裂缝区域中被胶覆盖的面积，`A_uf` 为裂缝区域内未被胶覆盖的面积，`A_s` 为全部胶体面积，`A_of` 为落在原裂缝区域外的胶体面积，则：

```text
CR = A_cov / A_c
UR = A_uf  / A_c
OR/OFR = A_of / A_s
CUI = 1 - sigma_cov / mean_cov
```

其中 `CUI` 不是全局面积比，而是沿裂缝骨架切分局部段，计算每段覆盖率 `cov_i`，再用这些局部覆盖率的均值和标准差计算均匀性。理想情况下 `UR = 1 - CR`；表 6 的每一个任务也满足这一关系：99.1+0.9=100、98.2+1.8=100、98.8+1.2=100、99.2+0.8=100、97.4+2.6=100。

表 6 的平均值可以独立复算：CR=(99.1+98.2+98.8+99.2+97.4)/5=98.54%，UR=1.46%，OR=(54.4+53.6+58.7+46.8+50.3)/5=52.76%，CUI=(98.1+95.3+96.4+97.4+96.7)/5=96.78%。所以公式定义和平均数算术上是可验证的。

但这不等于底层实验已经被独立证明。论文公开了五次任务的汇总值和颜色编码（绿色=覆盖、红色=欠填、蓝色=裂缝外胶体，p.11–12，S014），没有随仓库提供五次任务的原始裂缝掩膜、胶体掩膜、ROI、逐像素预测和局部覆盖率序列。因此目前能确认的是“指标口径和表内算术”，不能确认“每个像素测量值是否真实”。平均 OR 52.76% 还说明过填充很明显，不能只宣传 CR 98.54%；作者用 `k=1.8` 主动增加胶量以降低内部空洞风险。

## 2. ASPP 与 Decoder 的来源和创新边界

ASPP 是成熟模块，源于 DeepLab 系列的 atrous spatial pyramid pooling；论文 p.6 的引用 [46] 是 Chen 等人的 DeepLab 论文。仓库 `paddleseg/models/deeplabv3p.py` 也有独立的 `ASPPModule` 和 `Decoder` 实现。论文的 Decoder 也遵循成熟的“高层语义上采样 + 低层特征投影 + 跳跃拼接 + 卷积细化”范式，与 DeepLabV3+ 的工程思想高度相似。

因此不能把“发明 ASPP”或“发明 Decoder”写成原创。本文较合理的贡献边界是：把这些成熟结构接到 HrSegNet 的第三个高分辨率输出和第二个 block 输出上，压缩为适合边缘部署的 48/64 通道，并与深监督和机器人闭环任务组合。论文 p.6 给出的动机是原始融合对多尺度裂缝、细边界、低对比度和复杂纹理不够有效，不是“基线在所有模型中最差所以机械加模块”。

私有数据的消融表 3 给出可检验的因果拆分：基线 F1/IoU=84.0/72.4；只加 ASPP 为 84.8/73.7；只加 Decoder 为 84.3/72.9；两者为 85.9/75.3。ASPP 更明显改善 Precision 和区域匹配，Decoder 更明显改善 Recall；组合获得最大 IoU 增益。仓库 `hrsegnet_b16_ablation.py` 用 `use_aspp` 和 `use_lite_decoder` 开关实现了同样的四格实验。

迁移到新论文时，创新应放在“问题驱动的适配、组合策略、轻量化、损失、部署约束或闭环任务”，并完整报告 baseline、+A、+D、+A+D，而不是声称模块本身全新。

## 3. HrSegNet 基线与本文架构差别

原始作者公开的架构图是 `analysis/reader/assets/hrsegnet_original_architecture.png`（作者仓库 `CHDyshli/HrSegNet4CrackSegmentation` 的 `fig/fig1.png`）。本文论文的 Fig.8 是改进网络图 `analysis/reader/assets/fig8_hrsegnet_architecture.png`，并不是单独的 baseline 图。

基线代码 `paddleseg/models/hrsegnet_b32.py` 的数据流是：RGB → 两次 stride=2 stem（到 1/4 分辨率）→ 三个 `SegBlock` → 主 `SegHead`。每个 `SegBlock` 有保持 `base` 通道和空间尺寸的高分辨率路径，以及通道数为 `2*base、4*base、8*base` 的低分辨率语义引导路径；低分支经过 `1x1` 降到 `base`、双线性上采样后与高分支相加。训练时 `hrseg1_out`、`hrseg2_out` 还各接一个辅助头，损失为 `Lmain+0.5Laux1+0.5Laux2`；推理只保留主头。

本文 AD 版本保留 stem、三个 `SegBlock` 和辅助头，但将 `hrseg3_out` 先送入 ASPP；同时从 `hrseg2_out` 经 `1x1` 投影为 48 通道。ASPP 特征上采样到低层尺寸后与 48 通道特征拼接，再经过 3x3、64 通道、Dropout=0.1 的轻量 decoder，最后进入主 head。因此 baseline 是“第三个 block 直接接 head”，AD 是“第三个 block → ASPP，与第二个 block 的低层特征融合 → decoder → head”。

原始架构图的上方橙色框表示高分辨率路径，下方青色框表示 semantic guidance；改进图增加了 ASPP、低层投影和融合 decoder。论文 Fig.8 的两个 Head1/Head2 标签略显混乱，结合代码应理解为两个训练期辅助头和一个主输出头。

## 4. 修补任务生成算法的来源，以及它是否属于循迹

论文没有给出一段完整、单独发布的“任务生成算法代码”，而是把多个成熟组件串成一条 geometry-to-trajectory 流水线。明确引用的外部方法包括 Zhang 相机标定（[48]）、Zhang–Suen thinning（[49]）和 DeepLab/ASPP（[46]）。B-spline、双边滤波和手眼标定在正文中按成熟工程步骤使用，但没有分别给出新的算法定义或单独文献。

具体顺序是：相机内参和手眼变换标定 → 操作员选 ROI → 掩膜反相、双边滤波 → Zhang–Suen 得单像素骨架 → 用骨架到边界距离估计宽度、相邻骨架点累加估计长度、像素数乘 `delta^2` 估计面积 → 用 B-spline 拟合连续中心线 → 用 `v(s)=Q/(k*w(s)*h(s))` 按局部宽度/深度调整针头速度 → 将速度分解到 X/Y，乘脉冲当量换成步进频率和步数 → 3-DOF 机构执行注胶。

把它称作“基于骨架的裂缝修补轨迹生成与跟随”是准确的；把它称为完整的自主循迹或高级路径规划则过度。它完成的是从分割几何到中心线、坐标和电机指令的前端转换，论文没有给出全局避障、在线重规划、轨迹 RMSE、闭环位置反馈或模型预测控制。机器人是沿预先生成的平滑曲线执行，因此更接近 open-loop/半自主轨迹跟随，且 ROI 选择和修补确认仍有人参与。

成熟组件都有公开实现：Zhang 标定可用 OpenCV，Zhang–Suen 可用 scikit-image 或自行实现，B-spline 可用 SciPy；但当前仓库没有完整对应的相机 GUI、ROI、骨架到步进电机控制代码。故可以复现原理和离线几何实验，不能仅凭当前仓库复现整台机器人的真实注胶过程。


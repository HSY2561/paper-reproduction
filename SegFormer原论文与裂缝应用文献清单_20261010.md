# SegFormer 原论文与裂缝应用文献清单

检索日期：2026-10-10。面向本项目的 SegFormer-B0 裂缝分割、封闭场地覆盖巡检、停车后三轴修复主线。下面的阅读建议属于对论文与本项目关系的判断；除明确注明外，尚未核验论文全部实验协议和源码。

## 1. 原始算法论文：必引

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo. **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.** NeurIPS 2021, Advances in Neural Information Processing Systems 34.

- [会议正式页面](https://papers.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)
- [会议正式 PDF](https://papers.nips.cc/paper_files/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf)
- [arXiv](https://arxiv.org/abs/2105.15203)
- [作者官方 PyTorch 代码](https://github.com/NVlabs/SegFormer)

模型核心是分层多尺度 MiT Transformer 编码器和轻量 All-MLP 解码器；B0–B5 是同一原论文中的不同规模版本，B0 没有另一篇独立原论文。原文研究的是通用语义分割，不是裂缝专用模型。本工程使用 PaddleSeg 移植实现，应同时注明原模型来源和实际实现版本，不把 PaddleSeg 写成原作者代码。阅读时关注重叠 patch embedding、efficient self-attention、Mix-FFN、四级特征融合和 B0 的配置。

## 2. 直接使用或改进 SegFormer 的裂缝研究

### A. 最直接的土木应用依据

Hubing Li et al. **Automatic crack detection on concrete and asphalt surfaces using semantic segmentation network with hierarchical Transformer.** Engineering Structures, 307, 117903, 2024.

[正式论文 / DOI](https://doi.org/10.1016/j.engstruct.2024.117903)；[出版社页面](https://www.sciencedirect.com/science/article/pii/S0141029624004656/pdf)；[作者预印本页面（2023 版本，与正式版区分）](https://doi.org/10.2139/ssrn.4370927)。

该研究直接将 SegFormer 用于混凝土和沥青裂缝，并研究 CE 与 Dice 联合训练；评价覆盖细裂缝、复杂度、噪声鲁棒性和泛化。对本项目最有用的是基线选择依据、细裂缝评价和训练目标的设计。不能将论文中的损失收益直接归因到自己的结构模块，也不能直接拿其指标与本项目 Crack500 结果相比。

### B. SegFormer 解码器结构改进

**CA-SegFormer: An improved coordinate attention-enhanced model for crack segmentation in concrete structures.** Case Studies in Construction Materials, 23, e05492, 2025.

[正式论文](https://doi.org/10.1016/j.cscm.2025.e05492)。

在 SegFormer 解码器加入改进坐标注意力，把对角线方向信息纳入建模，针对细长和不规则混凝土裂缝；在 Concrete3k 上评价。适合借鉴“针对裂缝几何性质改进解码器”的动机及单模块消融逻辑，但本次尚未确认有可复现的作者公开代码，不应直接列入训练队列。

中文相关文献：龚玉磊、张亚丽、章红梅、王爱华，**混凝土构件表面裂缝分割的改进SegFormer方法**，《结构工程师》41(4):31–40，2025。[期刊页面](https://stru-en.tongji.edu.cn/CN/10.15935/j.cnki.jggcs.202504.0005)。也使用带对角线池化的坐标注意力。两篇内容有相近之处；未核验作者和数据关系前，不当作两个完全独立的效果证明。

### C. 路面 Transformer 编码—解码设计

**Pavement crack detection based on transformer network.** Automation in Construction, 2023. DOI: 10.1016/j.autcon.2022.104646。

[正式论文](https://doi.org/10.1016/j.autcon.2022.104646)。

提出 Crack Transformer（CT），采用 Swin Transformer 编码器和 SegFormer 风格轻量解码器，研究阴影、密集裂缝和落叶等路面干扰，并提出 CrackSC 数据集。它不是原始 SegFormer-B0；适合参考路面难例、数据集构建和局部/长程特征设计。

### D. 轻量模型筛选和部署依据

**Automatic real-time crack detection using lightweight deep learning models.** Engineering Applications of Artificial Intelligence, 138, 109340, 2024.

[正式论文](https://doi.org/10.1016/j.engappai.2024.109340)。

对 14 个轻量 CNN、ViT 和混合模型进行精度、实时性和鲁棒性比较，包含 MiT-B0/SegFormer。适合借鉴基线选择、模型复杂度和移动设备测量设计。服务器/桌面 GPU 性能不等于 Jetson Nano 部署性能，本文必须实测自己的硬件。

## 3. 裂缝分割的其他结构路线

**Hybrid-Segmentor: Hybrid approach for automated fine-grained crack segmentation in civil infrastructure.** June Moh Goo et al., Automation in Construction, 170, 105960, 2025.

[正式论文](https://doi.org/10.1016/j.autcon.2024.105960)；[作者机构公开全文](https://discovery.ucl.ac.uk/id/eprint/10203150/1/1-s2.0-S0926580524006964-main.pdf)。

结合 CNN 与 Transformer，分别建模精细局部特征和全局裂缝形态。可用于相关工作中解释纯 Transformer 的局部细节问题，以及局部—全局融合方向。不要仅因它宣称 SOTA 就把不同数据和协议的数字作为本项目目标值；是否有完整官方训练代码需进一步核验。

## 4. 与当前机器人主线最相关的两篇

### E. 覆盖巡检与裂缝填充：优先精读

Vishnu Veeraraghavan, Kyle Hunte, Jingang Yi, Kaiyan Yu. **Complete and Near-Optimal Robotic Crack Coverage and Filling in Civil Infrastructure.** IEEE Transactions on Robotics, 40:2850–2867, 2024. DOI: 10.1109/TRO.2024.3392077。

[正式论文 DOI](https://doi.org/10.1109/TRO.2024.3392077)；[公开全文入口](https://arxiv.org/abs/2403.00613)；[作者代码和实验材料](https://github.com/Binghamton-ACSR-Lab/Crack-Filling-Robot)。

研究同时传感巡检与执行机构覆盖（SIFC），包含已知裂缝与在线发现未知裂缝的规划，并协调底盘与喷嘴。它与你的“巡检过程中发现裂缝并修复”主线高度相关，必须用于贡献边界比较。作者平台使用全向机器人、XY 喷嘴机构和外部运动捕捉；实验以蓝色绘制裂缝、红色涂料覆盖模拟填充，不能等同真实道路材料封缝。你的差速底盘、停车二次观测和 XYZ 修复应分别说明差异并验证。

更正：此前规划把该工作仅称为预印本；作者仓库已明确给出上述 IEEE TRO 正式发表信息，正式论文中应引用期刊版本，arXiv 只作为公开阅读入口。

### F. 视觉分割到路面封缝执行

Jianqi Zhang et al. **Vision-guided robot for automated pixel-level pavement crack sealing.** Automation in Construction, 2024. DOI: 10.1016/j.autcon.2024.105783。

[正式论文](https://doi.org/10.1016/j.autcon.2024.105783)。

通过 CrackSegRefiner 精化裂缝掩膜，再结合视觉伺服与 LQR 完成路面封缝。参考价值是从像素几何到运动执行和封缝误差评价；本项目不需要照搬其 LQR 或沿裂缝移动方式。可与师兄论文一起说明停车后三轴精定位修复的设计依据。

## 5. 建议阅读顺序与论文引用位置

1. SegFormer 原文：方法章节的基线结构和来源。
2. Li et al., Engineering Structures 2024：引言中的土木应用依据，实验中的细裂缝评价。
3. CA-SegFormer：相关工作中的解码器/注意力改进；先核验开源和完整消融再考虑复现。
4. Lightweight models 2024：基线选择与设备效率。
5. IEEE TRO 覆盖填充论文：新的覆盖巡检和修复系统主线、任务级对照。
6. Vision-guided robot 2024：像素几何、封缝轨迹和真实执行指标。
7. CT 和 Hybrid-Segmentor：补充路面干扰及局部—全局结构比较。

文献中的数据集、标签、模型规模和划分不同，不能把它们各自的 IoU/F1 合成一个排行榜。本次只完成书目信息、来源和应用相关性核验，没有启动训练、改模型或确认应用论文都有开源代码。

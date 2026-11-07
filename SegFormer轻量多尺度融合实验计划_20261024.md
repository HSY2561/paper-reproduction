# SegFormer-B0 + Aligned 轻量多尺度融合实验计划

本波以 A0（SegFormer-B0、embedding_dim=128、Aligned）为母体，针对四路特征在 C1 分辨率拼接后 `512→128` 的融合瓶颈，尝试加和、成对聚合、低秩瓶颈、逐分支压缩和深度可分离空间混合。所有模型使用相同 MiT-B0 ImageNet 初始化、Crack500 train/val 清单、增强、CE、AdamW、FP32、seed=42、batch=4、18,000 iter。只在验证集完成选模和筛选，未读取 test 或自建数据。

候选及其解决的问题、位置、收益、代价和风险见活动工程 [EXPERIMENT_PLAN.md](D:/论文复现/experiments/segformer_lightfusion_wave10_20261024/refine-logs/EXPERIMENT_PLAN.md)。每项先 smoke，随后才正式训练；失败项保留真实日志和退出码。门槛是相对 A0 的精度基本不退化并且参数量不超过 A0 的 99%。通过候选之后才进入多种子、公开模型和 test 统一对照，不能提前宣称 SOTA。

当前执行工程：[segformer_lightfusion_wave10_20261024](D:/论文复现/experiments/segformer_lightfusion_wave10_20261024)。

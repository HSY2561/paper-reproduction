# Wave8/Wave9 final delivery

总报告：`D:/论文复现/SegFormer损失函数消融总报告_20261023.md`

交付压缩包：`D:/论文复现/SegFormer_wave8_wave9_report_delivery_20261023.zip`

损失合并表：`D:/论文复现/experiments/segformer_loss_wave8_20261019/results/loss_ablation.csv`

公开模型统一 test：`D:/论文复现/experiments/segformer_public_wave9_20261023/results/public_comparison.json`

本压缩包不包含原始数据、源代码树和大 checkpoint；正式 checkpoint、逐项训练日志和 SHA 仍保留在原实验目录。

结论：L01/L02在seed42及部分多种子指标上提高裂缝IoU/Recall，但三种子平均Precision门槛失败；当前没有经过预注册严格门槛的最终改进模型，不能声称SOTA。

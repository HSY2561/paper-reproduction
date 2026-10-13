# SegFormer-B0 Crack500 结构消融波次 5

本波次围绕长裂缝连续性、复杂背景误检和机器人部署成本建立三个单变量候选，均使用 PaddleSeg-2.10 官方模块：

| 编号 | 模块 | 插入位置 | 目标 |
|---|---|---|---|
| A | EMAU | 深层 C4 特征进入 MLP 前 | 在低分辨率上形成长距离、多尺度上下文原型，改善长裂缝连续性 |
| B | DualAttentionModule | 深层 C4 特征进入 MLP 前 | 以通道和位置注意力抑制阴影、落叶和纹理背景响应 |
| C | LightHamHead（SegNeXt） | 替代原始四层 MLP 融合解码头 | 在 1/8 分辨率完成矩阵分解式多尺度融合，降低高分辨率解码开销 |

基线协议固定为 RTX5060、FP32、seed=42、batch=4、AdamW、初始学习率 6e-5、PolynomialDecay end_lr=0、18,000 iter、相同 Crack500 train/val/test 清单和增强。候选仅由验证集选择，只有通过门槛才访问 test。

验证门槛相对 SegFormer-B0：裂缝 IoU 至少 +0.005，F1 至少 +0.003，Recall 下降不超过 0.005，mIoU 和 2px boundary-F1 不下降。三个候选先通过 20 iter 反向 smoke；正式训练顺序为 A、B、C，前一候选完成后再开始下一候选。任一候选失败都保留日志和真实退出码，不把失败结果称作 SOTA。只有至少两个单模块通过验证门槛时才考虑一个联合实验。

正式训练尚未访问 test。源码是独立快照 `src/PaddleSeg-2.10.0`，运行时用 `PYTHONPATH` 显式覆盖旧 editable 安装。

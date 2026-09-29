# 实验执行跟踪表

更新时间：2026-09-22

| Run ID | 里程碑 | 目的 | 系统/变体 | 数据划分 | 主要指标 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| R001-M0 | M0 | 重选冻结基线权重 | BiSeNetV2 原版各 checkpoint | val | 裂缝 IoU、Recall、前景 F1 | MUST | TODO | 只用验证集，不重新看 test |
| R002-M1 | M1 | 裂缝专用对比 | HrSegNet-B32 | train/val/test | 裂缝 IoU、Recall、F1、Params、FPS | MUST | TODO | 官方公开实现 |
| R003-M1 | M1 | 实时部署对比 | PP-LiteSeg/STDC | train/val/test | 同上 | MUST | TODO | 选最容易迁移的官方配置 |
| R004-M1 | M1 | 经典模型对比 | U-Net 或 DeepLabV3+ | train/val/test | 同上 | MUST | TODO | 二选一进入主表 |
| R005-M1 | M1 | 裂缝专用附加对比 | DeepCrack | train/val/test | 同上 | NICE | TODO | smoke test 通过后执行 |
| R006-M2 | M2 | 误差诊断 | BiSeNetV2 | val | FN、FP、断裂、边界误差 | MUST | TODO | 形成模块选择依据 |
| R007-M3 | M3 | 类别不平衡消融 | BiSeNetV2 + OHEM/Dice-CE | train/val/test | Recall、F1、IoU | MUST | TODO | 单因素 |
| R008-M3 | M3 | 多尺度消融 | BiSeNetV2 + ASPP/PPM | train/val/test | IoU、Recall、边界指标 | MUST | TODO | 单因素 |
| R009-M3 | M3 | 边界消融 | BiSeNetV2 + 边界监督 | train/val/test | IoU、Recall、边界指标 | MUST | TODO | 单因素 |
| R010-M4 | M4 | 两模块组合 | 最优模块组合 1 | train/val/test | 主指标和部署指标 | MUST | TODO | 由 R007-R009 决定 |
| R011-M4 | M4 | 三模块组合 | 最优模块组合 2 | train/val/test | 主指标和部署指标 | MUST | TODO | 由验证集决定 |
| R012-M4 | M4 | 参数量反事实 | 加宽 BiSeNetV2 | train/val/test | 主指标、Params、FPS | MUST | TODO | 排除“只是变大” |
| R013-M5 | M5 | 最终多种子 | 原版 BiSeNetV2 | train/val/test | 均值±标准差 | MUST | TODO | seeds 42/43/44 |
| R014-M5 | M5 | 最终多种子 | 最终改进模型 | train/val/test | 均值±标准差 | MUST | TODO | seeds 42/43/44 |
| R015-M5 | M5 | 最终强对比 | 最强 2-3 个对比模型 | train/val/test | 同一主表指标 | MUST | TODO | 明确 seed 和协议 |
| R016-M6 | M6 | 底盘循迹 | 冻结最终模型 | robot test | 跟踪误差、停靠误差 | MUST | TODO | 相机/机器人标定后 |
| R017-M6 | M6 | 三轴精定位 | 冻结最终模型 | robot test | 像素-物理误差、定位误差 | MUST | TODO | 记录重复性 |
| R018-M6 | M6 | 修复质量 | 冻结最终模型 | robot test | CR、UR、OR、CUI、AE、MSE、SD、CI | MUST | TODO | 形成系统闭环表 |

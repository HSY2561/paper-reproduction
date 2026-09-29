# 执行记录：视觉模型消融与 SOTA 阶段继续执行

日期：2026-09-23  
主工作区：`D:\论文复现`  
GitHub 同步目录：`D:\github\1`  
服务器工程：`/hy-tmp/crack_project`

## 已核验结果

服务器真实测试日志已经取回并核验：

| 模型 | mIoU | 裂缝 IoU | 裂缝 Precision | 裂缝 Recall |
|---|---:|---:|---:|---:|
| BiSeNetV2 + CE | 0.7606 | 0.5545 | 0.7146 | 0.7122 |
| BiSeNetV2 + OHEM + Dice + EMA | 0.7694 | 0.5705 | 0.7296 | 0.7233 |
| PP-LiteSeg-STDC1 | 0.7661 | 0.5634 | 0.7527 | 0.6915 |
| OCRNet-HRNet-W18 | **0.7760** | **0.5819** | **0.7601** | 0.7128 |
| DeepLabV3P-ResNet50 | 0.7257 | 0.4955 | 0.5995 | 0.7406 |
| SegFormer-B0 | 0.7570 | 0.5466 | 0.7388 | 0.6776 |
| PIDNet-S | 0.5005 | 0.0547 | 0.8036 | 0.0554 |

结论：BiSeNetV2+OHEM+Dice+EMA 优于 BiSeNetV2+CE，但低于 OCRNet-HRNet-W18；当前没有 SOTA 证据。

## 本轮实际执行

1. 通过仍可用的 SSH 交互会话进入 `/hy-tmp/crack_project`，确认服务器、GPU、配置和日志存在。
2. 核对 Crack500 三个 split 的掩码，train/val/test 均无空掩码。
3. 发现 SCSegamba 官方脚本每个 epoch 使用 `test` 目录做模型选择；为防止测试集选择偏差，建立：

   ```text
   /hy-tmp/crack_project/data/Crack500_scseg_valselect
   train_img/train_lab -> Crack500_data_u8/train_img/train_lab
   test_img/test_lab   -> Crack500_data_u8/val_img/val_lab
   ```

4. 先停止直接把原始 test 当验证的 SCSegamba 运行。
5. 按官方源码、官方环境和官方入口重新启动 SCSegamba 验证集选权重训练：

   ```text
   /usr/local/miniconda3/envs/scsegamba/bin/python main.py
   --dataset_path /hy-tmp/crack_project/data/Crack500_scseg_valselect
   --epochs 50 --batch_size_train 1 --batch_size_test 1
   --load_width 512 --load_height 512
   --output_dir /hy-tmp/crack_project/experiments/scsegamba_valselect_20260924
   ```

   日志：`/hy-tmp/crack_project/logs/scsegamba_valselect_20260924.log`。

## 已知失败和限制

- SCSegamba 1 epoch smoke 产生全零预测，不能作为正式结果；官方评估器还对空掩码缺少稳健处理。当前三个 split 已检查无空掩码。
- MixerCSeg 官方 selective-scan 扩展编译失败：服务器 nvcc 12.4，PyTorch 2.1.0+cu118。
- PIDNet-S 已完成测试但裂缝 Recall 仅 0.0554，表现为几乎不检出裂缝；不能据此断言其理论能力，需记录为当前配置失败。
- 当前裁块数据存在源拍摄编号跨集合的同源相关风险；正式论文仍需 source-frame-disjoint split。

## 未完成事项

- 等待 SCSegamba 50 epoch 训练完成，并用验证集最佳 checkpoint 对原始 test 1124 做独立评估。
- 若服务器有 CUDA 11.8 toolkit，按 MixerCSeg 官方说明重编译扩展；否则保留环境阻塞。
- 在冻结方案后做多随机种子、源帧互斥 split 和 Jetson Nano/机器人端延迟评估。

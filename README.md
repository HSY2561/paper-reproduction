# SegFormer Crack500 修复视觉模型

当前交付版本是 **SegFormer-B0 + Aligned + WTConv**，在 Crack500 上完成了四组同协议消融、验证集选模、冻结 checkpoint 和 test 评估。

## 当前最终模型

- 模型：SegFormer-B0 + Aligned + WTConv（C3）
- 数据集：Crack500；train 1896、val 348、test 1124
- 训练：RTX 5060、PaddlePaddle 3.4.0、FP32、seed 42、batch size 4、AdamW、18,000 iter
- test Precision：70.2578%
- test Recall：78.4197%
- test F1：74.1147%
- test crack IoU：58.8748%
- test mIoU：77.8361%
- 参数量：3.723490M

最终权重位于 `experiments/segformer_crack500_aligned_wtconv_ablation_wave17_20261030/checkpoints/C3_Aligned_WTConv/model.pdparams`。

## 复现实验

数据集不会上传。将 Crack500 放在本机数据目录后，按 `configs/C3_Aligned_WTConv.yml` 修改 `dataset_root`、`train_path` 和 `val_path`。使用 `runtime/ENVIRONMENT.md` 中的环境，设置 `PYTHONPATH` 指向 `src/PaddleSeg-2.10.0`，再运行：

```powershell
$env:PYTHONPATH = "experiments/segformer_crack500_aligned_wtconv_ablation_wave17_20261030/src/PaddleSeg-2.10.0"
python src/PaddleSeg-2.10.0/tools/train.py `
  --config experiments/segformer_crack500_aligned_wtconv_ablation_wave17_20261030/configs/C3_Aligned_WTConv.yml `
  --device gpu --save_dir training/C3_Aligned_WTConv `
  --iters 18000 --batch_size 4 --seed 42 --precision fp32 --use_ema
```

`results/` 保存四组有效模型的 val/test JSON、逐图边界指标和选择记录；`provenance/wave17_final_manifest.json` 保存数据清单、源码和 checkpoint SHA256。

## 结果说明

WTConv-only 使用 `alignment: none`，联合模型 C3 才使用 `alignment: full`。当前 WTConv 是工程中的低频 AvgPool2D、高分支 depthwise 5×5 和 pointwise/residual 实现，报告中将其称为 WTConv-only candidate，不将它表述为原论文的小波分解实现。当前结果属于本实验统一协议内的比较，尚未构成跨论文领域 SOTA 结论。

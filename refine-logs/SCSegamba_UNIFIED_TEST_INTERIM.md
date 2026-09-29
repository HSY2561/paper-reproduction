# SCSegamba 统一测试阶段性回执

日期：2026-09-23  
服务器工程：`/hy-tmp/crack_project`  
训练状态：50 epoch 训练尚未结束，当前日志已完成 epoch 11。以下结果是中途 checkpoint 的独立 test，不是最终模型选择结果。

## 测试协议

- 数据：原始 `Crack500_scseg_test_u8`，1,124 张图像。
- 输入：RGB，缩放到 512×512，归一化 `(x-0.5)/0.5`。
- 输出：sigmoid 后固定阈值 0.5。
- 统计：全局 TP/FP/FN/TN，计算裂缝 IoU、mIoU、Precision、Recall、前景 F1。
- 权重选择：尚未冻结；正式选择必须先在 348 张验证集上统一评估，再对原始 test 独立评估。

## 真实服务器回执

```text
checkpoint6.pth n 1124 tp 11986775 fp 5305736 fn 4487096 tn 272870249 mIoU 0.757861 fgIoU 0.550367 P 0.693177 R 0.727623 F1 0.709983
checkpoint9.pth n 1124 tp 11586925 fp 4554666 fn 4886946 tn 273621319 mIoU 0.758827 fgIoU 0.551010 P 0.717830 R 0.703352 F1 0.710517
checkpoint_best.pth n 1124 tp 11986775 fp 5305736 fn 4487096 tn 272870249 mIoU 0.757861 fgIoU 0.550367 P 0.693177 R 0.727623 F1 0.709983
```

## 阶段性判断

checkpoint9 的 mIoU 0.758827 低于当前 OCRNet-HRNet-W18 的 0.7760，也低于 BiSeNetV2 + OHEM CE + DiceLoss + EMA 的 0.7694。该结论只描述中途 checkpoint；待 50 epoch 训练完成并按验证集冻结模型后，再更新正式排名。

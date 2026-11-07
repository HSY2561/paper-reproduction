# SegFormer-B0 + Aligned + WTConv 四组消融表

四组均在 Crack500 上使用 RTX5060、FP32、seed42、batch4、AdamW、18,000 iter 和相同训练清单；选模只使用验证集，冻结后访问 test。

| Variant | Precision (%) | Recall (%) | F1-score (%) | IoU (%) | mIoU (%) | Boundary F1@2px (%) | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 SegFormer-B0 | 72.8613 | 72.4660 | 72.6631 | 57.0637 | 76.9434 | 37.9749 | 3.7152 | 4.1350 |
| C1 SegFormer-B0 + Aligned | 70.9480 | 77.1986 | 73.9414 | 58.6564 | 77.7387 | 37.9813 | 3.7853 | 4.5140 |
| C2 SegFormer-B0 + WTConv-only | 72.6574 | 73.6516 | 73.1511 | 57.6679 | 77.2579 | 37.9876 | 3.6533 | 3.3920 |
| C3 SegFormer-B0 + Aligned + WTConv | 70.2578 | 78.4197 | 74.1147 | 58.8748 | 77.8361 | 36.8405 | 3.7235 | 3.7710 |

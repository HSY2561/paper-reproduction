# AW+bgCE 主线交付索引

更新时间：2026-11-13（Asia/Shanghai）

## 主报告

- [AW公开对照与自建集全部消融汇总_20261112.md](AW公开对照与自建集全部消融汇总_20261112.md)
- [模型消融文档.md](模型消融文档.md)
- [SegFormer_AW精度诊断与单变量修正报告_20261112.md](SegFormer_AW精度诊断与单变量修正报告_20261112.md)
- [WTConv标准Haar实现与兼容说明_20261111.md](WTConv标准Haar实现与兼容说明_20261111.md)
- [cleanup_manifest_aw_bgce_20261113.md](cleanup_manifest_aw_bgce_20261113.md)

## 公开对照

- `experiments/segformer_crack500_aw_wavelet_20261112/results/AW_CEbg110_vs_six_public_test.csv`
- `experiments/segformer_crack500_aw_wavelet_20261112/results/AW_CEbg110_s42_test_exact.json`
- `experiments/segformer_crack500_aw_wavelet_20261112/results/AW_CEbg110_s42_selection.json`

## 自建集与跨数据集消融

- VOCdevkit3000：`experiments/segformer_voc_wtconv_bgce_20261112`
- VOCdevkit3000 旧四格历史：`experiments/segformer_vocdevkit3000_aligned_wtconv_ablation_20261031`
- DeepCrack：`experiments/segformer_deepcrack_aw_bgce_20261112`
- Crack500 当前 WTConv-only：`experiments/segformer_crack500_wtconv_bgce_20261112`

## 复现约定

所有实验目录仅提交配置、评估脚本、汇总结果、provenance 和报告；不提交原始数据、权重、训练缓存或本机凭据。主线配置固定 seed=42、FP32、batch=4、18,000 iter、AdamW/6e-5、EMA；VOC 的 test 清单为空，DeepCrack 的 test 为237张。

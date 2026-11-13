# AW+bgCE 主线冻结与仓库整理说明

## 冻结主线

主模型：SegFormer-B0 + Aligned + 标准 Haar WTConv（levels=1, kernel=3）+ background-weighted CE `[1.10, 1.0]`。

Crack500 选模规则：普通/EMA 精确 val mIoU 选最高；当前 AW+bgCE 选 ordinary，val mIoU=82.7069%，test Precision=73.1589%、Recall=74.7343%、F1=73.9382%、IoU=58.6524%、mIoU=77.7894%，checkpoint SHA256=`8b5ed4b1cef32e9e9bdc9dd2f657176a566cc106242e3800fde39692add26a09`。

## 保留的验证范围

- Crack500：AW+bgCE、阈值诊断、六个历史公开模型对照、当前标准 Haar WTConv-only。
- VOCdevkit3000：旧四格历史消融，以及当前标准 Haar WTConv-only + bgCE 的 587 张 val 重训。test 清单为空。
- DeepCrack：当前 Haar+bgCE 的 C0/C1/C2/C3 四组，train/val/test=255/45/237。

旧 AvgPool+Depthwise `candidate: wtconv` 与当前标准 Haar `wtconv_wavelet` 分开保存；普通 CE 与 bgCE 不混写。历史候选搜索、机器人实验、Bisenet/PaddleSeg下载缓存和原始数据不属于当前主线。

## 仓库规则

提交配置、评估脚本、结果 JSON/CSV、报告、provenance 和必要的模型源码；不提交原始数据、权重、训练缓存、日志、私有凭据或本机进程文件。完整保留清单见 [MANIFEST.md](MANIFEST.md)。

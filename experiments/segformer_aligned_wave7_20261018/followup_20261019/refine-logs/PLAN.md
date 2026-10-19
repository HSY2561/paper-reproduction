# 候选冻结后的归因、多seed与速度复核

2026年10月19日，所有新训练仅用Crack500 train/val，不访问test或自建数据。

## 冻结候选

22个完整组的真实exit0、最终iter18000、普通/EMA精确val最高选择、冻结SHA和独立val一致均已复核。只选同时满足原登记Aligned母体贡献门槛的两项：

- A21：SegFormer-B0 width128＋Aligned＋Polarized。满足A0贡献和原B0严格精度/参数数值门槛；稳定速度未知。
- A10：SegFormer-B0 width128＋Aligned＋AFNB。满足A0贡献门槛，总参数小于B0；不满足减参5%，属于平衡备选。

更高IoU的PAM/CCA有Precision或边界退化，APNB/Hamburger未通过A0的Precision下降上限，不能降低原门槛把它们写成通过。候选选择在新seed和test之前冻结，详见provenance/frozen_candidates.json。

## 新训练9项

1. W128＋AFNB，seed42，补缺失的单独模块行。
2. 原始B0、A0、A21、A10各seed43和44，共8项。

复用四模型合法seed42；Polarized单独模块复用旧波M02_PSA冻结结果，W128及Aligned复用已有冻结结果。最终归因表含W128、＋Aligned、＋X、＋Aligned＋X，原始B0另作总基线，不将压缩收益误算模块贡献。

所有训练仍是FP32、batch4、18k、ImageNet MiT-B0共同初始化、AdamW 6e-5、end_lr0、既有数据/增强/CE/EMA。独立选模、冻结SHA、官方验证和精确零外延2px边界验证不变。未齐3seed时不写均值±标准差。

## 在新seed开始前登记的判定

各候选分别与同seed的A0/B0作差；报告3个独立结果、均值和样本标准差，不能把三seed称为统计显著证据。

- 第二模块稳定贡献：对A0的平均IoU≥+0.002，F1≥+0.001，mIoU不降，平均P/R/BF1下降各≤0.005；至少2/3 seed的IoU差值为正。
- 严格方案继续核对B0均值的IoU≥+0.002、F1≥+0.001、mIoU不降、R/BF1下降≤0.005，Params≤0.95倍，稳定配对延迟≤1.10倍。
- 平衡备选：第二模块稳定贡献成立，Params≤原B0，稳定配对延迟≤1.10倍；标签不与严格减参5%混淆。
- 任一速度证据不稳定，最终速度/完整目标通过留空；不得仅凭参数量宣称已加速。若均不稳定，报告单seed与多seed结论差异，不能事后挑seed或降低门槛。

速度复测在本轮GPU训练之前执行：四个冻结checkpoint核SHA、同进程FP32/batch1、400/480、15轮平衡顺序，每轮30次同步前向。预先登记每模型P90/P10≤1.15，配对比值IQR/median≤0.10为稳定条件。桌面应用继续共享资源，当前CPU满载；若仍不稳定只记录条件测量，不伪造独占性能。

完成这9项后审核候选，再进入六个原始公开模型的缺项、冻结统一test和最终交付。预估本阶段约6–8小时，机器资源和后端异常可能延长。全部交付清单仍以主工程refine-logs/FINAL_DELIVERY.md为准。

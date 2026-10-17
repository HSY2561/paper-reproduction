# SegFormer-B0 Crack500 精度效率消融进度

更新时间：2026-10-17T23:13:52+08:00

以下都是验证集结果。待训练/失败条目不填写性能；普通/EMA以精确验证mIoU选择。

| 编号 | 模块 | 状态 | Precision | Recall | F1 | IoU | mIoU | Params M | 480延迟ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | SegFormer-B0 | 冻结复用 | 0.828269 | 0.789134 | 0.808228 | 0.678174 | 0.828021 | 3.7152 | 15.585 |
| W128 | MLP decoder width128 | complete | 0.802747 | 0.811416 | 0.807058 | 0.676528 | 0.826784 | 3.4517 | 13.089 |
| W64 | MLP decoder width64 | complete | 0.811151 | 0.794618 | 0.802799 | 0.670564 | 0.823742 | 3.3692 | 15.709 |
| M08_MSCA | MSCAN SpatialAttention | complete | 0.811014 | 0.798482 | 0.804699 | 0.673219 | 0.825150 | 3.5154 | 10.013 |
| M13_Aligned | SFNet AlignedModule | complete | 0.826766 | 0.790974 | 0.808474 | 0.678520 | 0.828184 | 3.4704 | 17.940 |
| M03_GC | Global Context Block | complete | 0.820801 | 0.791426 | 0.805846 | 0.674826 | 0.826142 | 3.4605 | 16.156 |
| M06_CAM | Channel Attention Module | complete | 0.807763 | 0.806565 | 0.807164 | 0.676676 | 0.826938 | 3.4517 | 15.218 |
| M11_ASPPsep | Separable ASPP | complete | 0.809164 | 0.806430 | 0.807795 | 0.677564 | 0.827430 | 3.5562 | 9.738 |
| M02_PSA | Polarized Self Attention | complete | 0.801853 | 0.807231 | 0.804533 | 0.672986 | 0.824889 | 3.4846 | 13.818 |
| M04_CCA | Criss Cross Attention | complete | 0.816508 | 0.796821 | 0.806544 | 0.675806 | 0.826602 | 3.5340 | 14.707 |
| M05_PAM | Position Attention Module | complete | 0.816568 | 0.795738 | 0.806019 | 0.675068 | 0.826211 | 3.5340 | 14.271 |
| M07_NonLocal | NonLocal2D | complete | 0.797710 | 0.819002 | 0.808216 | 0.678156 | 0.827572 | 3.5185 | 9.883 |
| M09_Encoding | EncModule | complete | 0.823205 | 0.791703 | 0.807147 | 0.676652 | 0.827145 | 3.5885 | 9.482 |
| M10_GloRe | GloRe GruModule | complete | 0.810472 | 0.805128 | 0.807791 | 0.677559 | 0.827446 | 3.4990 | 10.673 |
| M14_UAFMchannel | UAFM Channel Attention | complete | 0.818604 | 0.796510 | 0.807406 | 0.677017 | 0.827275 | 3.7894 | 16.321 |
| M01_SE | MobileNetV3 SE | needs_review | — | — | — | — | — | — | — |
| M12_PSP | PSP PPModule | smoke_failed | — | — | — | — | — | — | — |

14个不同模块机制与2个宽度控制分开计数。默认每个完成18k；smoke失败不等于性能差。
精度/效率通过与最终目标通过分开记录，详见comparison.json和预注册计划。六个公开模型由候选确定后统一执行，均保留原始结构。
不能从筛选验证集的偶然最大值推断SOTA；最终需要多种子、冻结test与源帧互斥复核。

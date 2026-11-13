import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
test = json.loads((ROOT / "results/AW_s42_test_exact.json").read_text(encoding="utf-8-sig"))
val = json.loads((ROOT / "results/AW_s42_val_exact.json").read_text(encoding="utf-8-sig"))
ema = json.loads((ROOT / "results/AW_s42_val_ema_exact.json").read_text(encoding="utf-8-sig"))

rows = [
    {"Model": "SegFormer-B0 + Aligned + Wavelet WTConv (AW, ours)", "Precision (%)": test["precision"] * 100, "Recall (%)": test["recall"] * 100, "F1-score (%)": test["f1"] * 100, "IoU (%)": test["crackIoU"] * 100, "mIoU (%)": test["mIoU"] * 100, "Params (M)": 3.710818, "Status": "new AW seed42 test"},
    {"Model": "SegFormer-B0", "Precision (%)": 72.8905, "Recall (%)": 72.4478, "F1-score (%)": 72.6684, "IoU (%)": 57.0702, "mIoU (%)": 76.9475, "Params (M)": 3.715170, "Status": "historical frozen public comparison"},
    {"Model": "OCRNet-HRNet-W18", "Precision (%)": 70.3819, "Recall (%)": 75.9782, "F1-score (%)": 73.0731, "IoU (%)": 57.5710, "mIoU (%)": 77.1509, "Params (M)": 12.109902, "Status": "historical frozen public comparison"},
    {"Model": "PP-LiteSeg-STDC1", "Precision (%)": 74.6053, "Recall (%)": 69.7870, "F1-score (%)": 72.1157, "IoU (%)": 56.3914, "mIoU (%)": 76.6260, "Params (M)": 8.039122, "Status": "historical frozen public comparison"},
    {"Model": "PSPNet-ResNet50", "Precision (%)": 72.1991, "Recall (%)": 70.6710, "F1-score (%)": 71.4269, "IoU (%)": 55.5535, "mIoU (%)": 76.1313, "Params (M)": 67.902052, "Status": "historical frozen public comparison"},
    {"Model": "DeepLabV3P-ResNet50", "Precision (%)": 72.2276, "Recall (%)": 73.5601, "F1-score (%)": 72.8877, "IoU (%)": 57.3412, "mIoU (%)": 77.0754, "Params (M)": 26.789874, "Status": "historical frozen public comparison"},
    {"Model": "Fast-SCNN", "Precision (%)": 75.0411, "Recall (%)": 61.4485, "F1-score (%)": 67.5680, "IoU (%)": 51.0208, "mIoU (%)": 73.8029, "Params (M)": 1.441332, "Status": "historical frozen public comparison"},
]

with (ROOT / "results/AW_vs_six_public_test.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Crack500 SegFormer-B0 + Aligned + Wavelet WTConv（AW）重新训练与六模型对照报告",
    "",
    "本轮重新训练标准一层正交 Haar DWT/IWT 版本 `candidate: wtconv_wavelet`。训练与选模只使用 Crack500 train/val，普通 checkpoint 和 EMA checkpoint 均独立验证，冻结后才访问 test。",
    "",
    "## 1. 固定协议",
    "",
    "RTX5060、PaddlePaddle 3.4.0、FP32、seed=42、batch=4、AdamW、初始学习率 6e-5、PolynomialDecay end_lr=0、CE、EMA、18,000 iter、400×400 裁剪与既有增强。train/val/test 数量为 1896/348/1124。",
    "",
    "AW 结构为完整 Aligned 融合后接 WTConvWavelet(levels=1, kernel=3)，包含 LL/LH/HL/HH Haar 分解、逐子带 depthwise filtering、逆小波重建和残差局部细节分支。",
    "",
    "## 2. ordinary/EMA 验证选模",
    "",
    "| checkpoint | val mIoU | crack IoU | Precision | Recall | F1 | BF1@2px | hard clDice |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    f"| ordinary（冻结） | {val['mIoU']*100:.4f} | {val['crackIoU']*100:.4f} | {val['precision']*100:.4f} | {val['recall']*100:.4f} | {val['f1']*100:.4f} | {val['boundary_f1_2px']*100:.4f} | {val['hard_cldice']*100:.4f} |",
    f"| EMA | {ema['mIoU']*100:.4f} | {ema['crackIoU']*100:.4f} | {ema['precision']*100:.4f} | {ema['recall']*100:.4f} | {ema['f1']*100:.4f} | {ema['boundary_f1_2px']*100:.4f} | {ema['hard_cldice']*100:.4f} |",
    "",
    f"ordinary 精确 val mIoU={val['mIoU']:.10f}，EMA={ema['mIoU']:.10f}，因此冻结 ordinary。",
    "",
    "## 3. AW 的 Crack500 test 结果",
    "",
    "| Model | Precision (%) | Recall (%) | F1-score (%) | IoU (%) | mIoU (%) | Params (M) | FLOPs (G) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    f"| **AW** | **{test['precision']*100:.4f}** | **{test['recall']*100:.4f}** | **{test['f1']*100:.4f}** | **{test['crackIoU']*100:.4f}** | **{test['mIoU']*100:.4f}** | 3.710818 | 3.732400 |",
    "",
    f"AW test BF1@2px={test['boundary_f1_2px']*100:.4f}%，hard clDice={test['hard_cldice']*100:.4f}%，最细裂缝 Recall(r≤1)={test['gt_radius_strata']['r_le1']['recall']*100:.4f}%。",
    "",
    "## 4. 与六个公开模型的 Crack500 对照",
    "",
    "这里按当前主线将“六个公开数据”解释为六个公开模型在 Crack500 上的冻结 test 记录；它们的历史后端、初始化、损失和 EMA 与 AW 不完全相同，因此只能作为当前复现范围内的对照。",
    "",
    "| Model | Precision (%) | Recall (%) | F1-score (%) | IoU (%) | mIoU (%) | Params (M) | 数据状态 |",
    "|---|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    lines.append(f"| {row['Model']} | {row['Precision (%)']:.4f} | {row['Recall (%)']:.4f} | {row['F1-score (%)']:.4f} | {row['IoU (%)']:.4f} | {row['mIoU (%)']:.4f} | {row['Params (M)']:.6f} | {row['Status']} |")

lines += [
    "",
    "## 5. AW 相对历史 SegFormer-B0",
    "",
    f"Precision {(test['precision']*100)-72.8905:+.4f} pp，Recall {(test['recall']*100)-72.4478:+.4f} pp，F1 {(test['f1']*100)-72.6684:+.4f} pp，裂缝 IoU {(test['crackIoU']*100)-57.0702:+.4f} pp，mIoU {(test['mIoU']*100)-76.9475:+.4f} pp，参数量 {3.710818-3.715170:+.6f} M。",
    "",
    "AW 在当前六模型表中裂缝 IoU、mIoU 和 F1 均高于六个历史公开模型；Recall 高于 OCRNet、DeepLabV3P、PP-LiteSeg、PSPNet 和 Fast-SCNN，但 Precision 低于 PP-LiteSeg、Fast-SCNN 和原始 SegFormer-B0。",
    "",
    "## 6. 与旧 AvgPool WTConv 结果的关系",
    "",
    "旧 Wave17 C3 的 WTConv 是 AvgPool + depthwise candidate，不能当作标准 Haar WTConv。新 AW 必须使用本轮新 checkpoint；两者实现和 checkpoint 不同。当前单 seed AW test 指标低于旧 C3 历史记录，说明标准化小波实现不等于自动提升精度，后续应通过多种子和逐图分析确认稳定性。",
    "",
    "## 7. 证据索引",
    "",
    f"- 训练 exit：`{ROOT / 'results/AW_s42_train_exit.json'}`",
    f"- 选模记录：`{ROOT / 'results/AW_s42_selection.json'}`",
    f"- 精确 val：`{ROOT / 'results/AW_s42_val_exact.json'}`",
    f"- 精确 test：`{ROOT / 'results/AW_s42_test_exact.json'}`",
    f"- 六模型 CSV：`{ROOT / 'results/AW_vs_six_public_test.csv'}`",
    f"- 协议：`{ROOT / 'EXPERIMENT_PLAN.md'}`",
    "",
    "当前结论限定为 Crack500 本实验协议内的比较，不宣称领域 SOTA。",
]
(ROOT / "AW重新训练与六模型对照报告_20261112.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(ROOT / "AW重新训练与六模型对照报告_20261112.md")

from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.__version__ = "3.7.0"
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = Path(r"D:\论文复现")
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

public_rows = [
    ["SegFormer-B0 + Aligned + WTConv (Ours, C3, Wave17)", 70.2578, 78.4197, 74.1147, 58.8748, 77.8361, 3.723490, "本次Wave17新训练"],
    ["SegFormer-B0", 72.8905, 72.4478, 72.6684, 57.0702, 76.9475, 3.715170, "公开模型历史冻结记录"],
    ["OCRNet-HRNet-W18", 70.3819, 75.9782, 73.0731, 57.5710, 77.1509, 12.109902, "公开模型历史冻结记录"],
    ["PP-LiteSeg-STDC1", 74.6053, 69.7870, 72.1157, 56.3914, 76.6260, 8.039122, "公开模型历史冻结记录"],
    ["PSPNet-ResNet50", 72.1991, 70.6710, 71.4269, 55.5535, 76.1313, 67.902052, "公开模型历史冻结记录"],
    ["DeepLabV3P-ResNet50", 72.2276, 73.5601, 72.8877, 57.3412, 77.0754, 26.789874, "公开模型历史冻结记录"],
    ["Fast-SCNN", 75.0411, 61.4485, 67.5680, 51.0208, 73.8029, 1.441332, "公开模型历史冻结记录"],
]
public_header = ["Model", "Precision (%)", "Recall (%)", "F1-score (%)", "IoU (%)", "mIoU (%)", "Params (M)", "Status"]
with (RESULTS / "Crack500_六模型对照表_20261031.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(public_header)
    writer.writerows(public_rows)

abl_rows = [
    ["C0 SegFormer-B0", 72.8613, 72.4660, 72.6631, 57.0637, 76.9434, 37.9749, 3.715170, 4.135],
    ["C1 SegFormer-B0 + Aligned", 70.9480, 77.1986, 73.9414, 58.6564, 77.7387, 37.9813, 3.785314, 4.514],
    ["C2 SegFormer-B0 + WTConv-only", 72.6574, 73.6516, 73.1511, 57.6679, 77.2579, 37.9876, 3.653346, 3.392],
    ["C3 SegFormer-B0 + Aligned + WTConv", 70.2578, 78.4197, 74.1147, 58.8748, 77.8361, 36.8405, 3.723490, 3.771],
]
abl_header = ["Variant", "Precision (%)", "Recall (%)", "F1-score (%)", "IoU (%)", "mIoU (%)", "Boundary F1@2px (%)", "Params (M)", "FLOPs (G)"]
with (RESULTS / "SegFormer_Aligned_WTConv_消融表_20261031.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(abl_header)
    writer.writerows(abl_rows)

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers)-1)) + "|"]
    for row in rows:
        vals = [row[0]] + [f"{x:.4f}" if isinstance(x, float) else str(x) for x in row[1:]]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)

(RESULTS / "Crack500_六模型对照表_20261031.md").write_text(
    "# Crack500 六模型对照表\n\n" +
    "当前模型为 Wave17 新训练结果；六个公开模型沿用已冻结的历史公开模型记录，严格公平结论仍需统一后端、初始化、损失、EMA和训练预算复核。\n\n" +
    md_table(public_header, public_rows) + "\n",
    encoding="utf-8"
)
(RESULTS / "SegFormer_Aligned_WTConv_消融表_20261031.md").write_text(
    "# SegFormer-B0 + Aligned + WTConv 四组消融表\n\n" +
    "四组均在 Crack500 上使用 RTX5060、FP32、seed42、batch4、AdamW、18,000 iter 和相同训练清单；选模只使用验证集，冻结后访问 test。\n\n" +
    md_table(abl_header, abl_rows) + "\n",
    encoding="utf-8"
)

# Paper-like architecture diagram.
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(18, 10), dpi=220)
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")

colors = {
    "stage": "#f8e3a5", "feature": "#d8e4ee", "aligned": "#85c7be",
    "wtconv": "#f2c5a5", "head": "#9bc9df", "detail": "#f4f5f6",
    "text": "#1f2933", "line": "#34495e", "accent": "#2f7e78"
}

def box(x, y, w, h, text, face, fontsize=10, edge="#52616b", radius=0.08, weight="normal"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                           linewidth=1.2, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize,
            color=colors["text"], weight=weight, wrap=True)
    return patch

def arrow(x1, y1, x2, y2, text=None, dashed=False, color=colors["line"], lw=1.4):
    style = "->"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle="--" if dashed else "-"))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2+0.18, text, ha="center", va="bottom", fontsize=8, color=color)

ax.text(0.4, 9.55, "SegFormer-B0 + Aligned + WTConv", fontsize=19, weight="bold", color=colors["text"])
ax.text(0.4, 9.17, "MiT-B0 encoder, spatial feature alignment, low-high frequency detail enhancement and pixel head", fontsize=10, color="#52616b")

# Encoder lane.
ax.text(0.5, 8.65, "Encoder (MiT-B0)", fontsize=12, weight="bold", color=colors["text"])
box(0.5, 7.5, 1.0, 0.7, "Input\nH×W×3", colors["feature"], 9, weight="bold")
arrow(1.5, 7.85, 2.0, 7.85)
for i, (label, ch) in enumerate([("Stage 1", "H/4×W/4\n32 ch"), ("Stage 2", "H/8×W/8\n64 ch"), ("Stage 3", "H/16×W/16\n160 ch"), ("Stage 4", "H/32×W/32\n256 ch")]):
    x = 2.0 + i*1.55
    box(x, 7.4, 1.1, 0.9, f"{label}\n{ch}", colors["stage"], 8.5, weight="bold")
    if i < 3: arrow(x+1.1, 7.85, x+1.55, 7.85)

ax.text(8.25, 8.65, "Multi-scale projection and alignment", fontsize=12, weight="bold", color=colors["text"])
for i, label in enumerate(["P1\n256 ch", "P2\n256 ch", "P3\n256 ch", "P4\n256 ch"]):
    x = 8.2 + i*1.35
    box(x, 7.4, 0.9, 0.8, label, colors["feature"], 8.5, weight="bold")
    arrow(7.5, 7.85, x, 7.85) if i == 0 else None

# Aligned module branch.
ax.text(7.85, 6.75, "Aligned feature restoration", fontsize=10.5, weight="bold", color=colors["accent"])
box(8.1, 5.85, 1.35, 0.65, "Flow\nestimation", colors["aligned"], 8.8, weight="bold")
box(10.0, 5.85, 1.35, 0.65, "Bilinear\nwarp", colors["aligned"], 8.8, weight="bold")
arrow(9.45, 6.18, 10.0, 6.18, "Δ")
arrow(8.65, 7.4, 8.65, 6.5, "P1")
arrow(11.1, 7.4, 11.1, 6.5, "P4")
box(11.85, 5.85, 1.1, 0.65, "A4\n256 ch", colors["aligned"], 8.8, weight="bold")
arrow(11.35, 6.18, 11.85, 6.18)

# Fusion and WTConv branch.
ax.text(12.65, 8.65, "Detail/context enhancement", fontsize=12, weight="bold", color=colors["text"])
box(12.5, 4.65, 1.5, 0.75, "Concat\nP1 + ↑P2 + ↑P3 + A4", colors["head"], 8.5, weight="bold")
for x, label in [(8.65, "P1"), (10.0, "↑P2"), (11.35, "↑P3"), (12.4, "A4")]:
    arrow(x, 7.35, 13.1, 5.4, label, dashed=True, color="#627d98", lw=1.0)
box(14.35, 4.65, 1.05, 0.75, "1×1\n1024→128", colors["stage"], 8.3, weight="bold")
arrow(14.0, 5.02, 14.35, 5.02)
box(15.8, 4.65, 1.15, 0.75, "WTConv\nlow + high", colors["wtconv"], 8.3, weight="bold")
arrow(15.4, 5.02, 15.8, 5.02)
box(17.25, 4.65, 0.6, 0.75, "1×1\nhead", colors["head"], 7.6, weight="bold")
arrow(16.95, 5.02, 17.25, 5.02)

# Output label.
ax.text(17.72, 4.25, "2-class\nmask", ha="center", va="top", fontsize=8.5, color=colors["text"], weight="bold")

# Detail panels.
box(0.5, 1.0, 5.55, 2.5, "", "white", edge="#9aa7b1", radius=0.12)
ax.text(0.75, 3.25, "Aligned module detail", fontsize=11, weight="bold", color=colors["text"])
box(0.9, 2.15, 1.05, 0.55, "P1", colors["feature"], 9, weight="bold")
box(2.35, 2.15, 1.35, 0.55, "1×1 + resize", colors["stage"], 8.5)
box(4.2, 2.15, 1.15, 0.55, "Concat + 3×3", colors["stage"], 8.5)
arrow(1.95, 2.42, 2.35, 2.42)
arrow(3.7, 2.42, 4.2, 2.42)
box(1.15, 1.25, 1.05, 0.55, "P4", colors["feature"], 9, weight="bold")
box(3.1, 1.25, 1.35, 0.55, "Flow Δ", colors["aligned"], 8.5)
box(4.95, 1.25, 0.8, 0.55, "Warp", colors["aligned"], 8.5)
arrow(2.2, 1.52, 3.1, 1.52)
arrow(4.45, 1.52, 4.95, 1.52)
arrow(4.75, 2.15, 4.75, 1.8, dashed=True, color=colors["accent"])

box(6.35, 1.0, 5.55, 2.5, "", "white", edge="#9aa7b1", radius=0.12)
ax.text(6.6, 3.25, "WTConv candidate detail", fontsize=11, weight="bold", color=colors["text"])
box(6.8, 2.15, 1.25, 0.55, "Bottleneck\n128 ch", colors["stage"], 8.5)
box(8.7, 2.15, 1.1, 0.55, "High\nDW 5×5", colors["wtconv"], 8.5)
box(10.35, 2.15, 1.15, 0.55, "Low\nAvgPool2", colors["wtconv"], 8.5)
arrow(8.05, 2.42, 8.7, 2.42)
arrow(8.05, 2.42, 10.35, 2.42)
box(8.7, 1.25, 1.1, 0.55, "Upsample", colors["aligned"], 8.5)
box(10.35, 1.25, 1.15, 0.55, "BN + ReLU\n+ residual", colors["aligned"], 8.5)
arrow(10.92, 2.15, 9.25, 1.8, dashed=True, color=colors["accent"])
arrow(9.8, 1.52, 10.35, 1.52)
ax.text(6.8, 1.08, "Low branch captures coarse context; high branch preserves local crack detail.", fontsize=8.3, color="#52616b")

# Legend.
box(12.45, 1.0, 5.45, 2.5, "", "white", edge="#9aa7b1", radius=0.12)
ax.text(12.7, 3.25, "Legend and output", fontsize=11, weight="bold", color=colors["text"])
legend = [("Encoder stage / projection", colors["stage"]), ("Feature map", colors["feature"]), ("Alignment / warp", colors["aligned"]), ("WTConv branch", colors["wtconv"]), ("Fusion / prediction head", colors["head"])]
for i, (label, face) in enumerate(legend):
    y = 2.75 - i*0.34
    ax.add_patch(Rectangle((12.75, y-0.12), 0.24, 0.24, facecolor=face, edgecolor="#52616b", linewidth=0.8))
    ax.text(13.12, y, label, va="center", fontsize=8.5, color=colors["text"])
ax.text(12.75, 1.18, "Output: 1×1 classification head → resize → binary crack mask", fontsize=8.3, color="#52616b")

plt.tight_layout(pad=0.5)
fig.savefig(FIGURES / "SegFormer_Aligned_WTConv_算法原理图_20261031.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "SegFormer_Aligned_WTConv_算法原理图_20261031.svg", bbox_inches="tight")
plt.close(fig)
print("generated tables and figures")

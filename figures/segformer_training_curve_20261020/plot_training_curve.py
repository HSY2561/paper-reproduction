"""Render observed SegFormer-B0 training loss and validation mIoU; no inference."""
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\论文复现")
RUN = ROOT / "experiments/retrain_wave4_20261007"
OUT = Path(__file__).resolve().parent
LOG = RUN / "logs/segformer_b0_train.log"
SEL = RUN / "checkpoints/segformer_b0/selection.json"
text = LOG.read_text(encoding="utf-8-sig")
selection = json.loads(SEL.read_text(encoding="utf-8-sig"))
train, events = [], []
iteration = None
for line in text.splitlines():
    m = re.search(r"\[TRAIN\] epoch: (\d+), iter: (\d+)/18000, loss: ([0-9.eE+-]+)", line)
    if m:
        epoch, iteration, loss = int(m[1]), int(m[2]), float(m[3])
        train.append(dict(epoch=epoch, iteration=iteration, training_loss=loss))
    m = re.search(r"\[EVAL\] #Images: 348 mIoU: ([0-9.]+)", line)
    if m:
        assert iteration is not None
        events.append((iteration, float(m[1])))
assert [r["iteration"] for r in train] == list(range(10, 18001, 10))
assert len(events) == 36
validation = []
for i in range(0, len(events), 2):
    assert events[i][0] == events[i + 1][0]
    validation.append(dict(iteration=events[i][0], ordinary_miou=events[i][1],
                           ema_miou=events[i + 1][1]))
best_iter = selection["selected"]["state"]["iter"]
best_miou = selection["selected"]["state"]["mIoU"]
assert selection["selected"]["variant"] == "best_model"
assert max(validation, key=lambda r: r["ordinary_miou"])["iteration"] == best_iter
assert abs(max(r["ordinary_miou"] for r in validation) - best_miou) < 0.00005
blocks = []
for end in range(1000, 18001, 1000):
    rows = [r for r in train if end - 1000 < r["iteration"] <= end]
    blocks.append(dict(iteration=end, start_iteration=end - 999,
                       logged_windows=len(rows),
                       mean_training_loss=float(np.mean([r["training_loss"] for r in rows]))))

def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

write_csv("training_loss_raw.csv", train)
write_csv("training_loss_1000iter_mean.csv", blocks)
write_csv("validation_miou.csv", validation)
metadata = dict(
    model="SegFormer-B0 (original, width256)", dataset="Crack500", seed=42,
    log=str(LOG), log_sha256=hashlib.sha256(LOG.read_bytes()).hexdigest(),
    selection=str(SEL), selection_sha256=hashlib.sha256(SEL.read_bytes()).hexdigest(),
    frozen_checkpoint_sha256=selection["sha256"],
    selected_iteration=best_iter, selected_validation_miou=best_miou,
    train_log_points=len(train), validation_events=len(validation),
    validation_loss_available=False, test_accessed=False, inference_run=False,
    loss_definition="CE; each raw log value is the mean of 10 training iterations, rounded to 4 decimals.",
    aggregation="Red squares: non-overlapping 1000-iteration means (100 logged windows); plotted at interval ends.",
    validation_definition="Blue circles: ordinary-model validation mIoU recorded every 1000 iterations, rounded to 4 decimals.",
    selection_rule="Maximum validation mIoU across ordinary and EMA; ordinary iter 11000 selected.",
)
(OUT / "source_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "xtick.direction": "in", "ytick.direction": "in",
})
x = np.array([r["iteration"] for r in blocks]) / 1000
raw_x = np.array([r["iteration"] for r in train]) / 1000
raw_y = np.array([r["training_loss"] for r in train])
loss_y = [r["mean_training_loss"] for r in blocks]
val_y = [r["ordinary_miou"] for r in validation]

def save(fig, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(6.0, 3.9))
right = ax.twinx()
ax.plot(raw_x, raw_y, color="#efb6b6", lw=.5, alpha=.65, label="_nolegend_")
a, = ax.plot(x, loss_y, "-s", color="red", ms=4.2, lw=1.5, label="Training loss (1000-iter mean)")
b, = right.plot(x, val_y, "-o", color="blue", ms=4.3, lw=1.5, label="Validation mIoU")
ax.set(xlim=(0, 18.5), ylim=(0, max(raw_y) * 1.06),
       xlabel=r"Iteration ($\times 10^3$)", ylabel="Training loss (cross-entropy)")
right.set(ylim=(.78, .85), ylabel="Validation mIoU")
ax.tick_params(top=True)
ax.set_xticks([0, 3, 6, 9, 12, 15, 18])
ax.axvline(best_iter / 1000, color="black", linestyle=(0, (5, 4)), linewidth=1.2)
right.annotate(f"Frozen: {best_iter:,} iter\nVal. mIoU = {best_miou:.4f}",
               xy=(best_iter / 1000, best_miou), xytext=(12, .800),
               fontsize=9, ha="left", va="top",
               bbox=dict(facecolor="white", edgecolor="none", pad=2),
               arrowprops=dict(arrowstyle="-", color="black", lw=.8))
ax.legend(handles=[a, b], loc="upper right", frameon=False)
fig.tight_layout()
save(fig, "SegFormer_B0_training_validation_curve")

# Separate axes are useful for manuscripts where a dual-axis plot is unsuitable.
fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.25))
axes[0].plot(raw_x, raw_y, color="#efb6b6", lw=.5)
axes[0].plot(x, loss_y, "-s", color="red", ms=3.5, lw=1.3)
axes[0].set(ylabel="Training loss (cross-entropy)", ylim=(0, max(raw_y)*1.06),
            title="(a) Training loss")
axes[1].plot(x, val_y, "-o", color="blue", ms=3.5, lw=1.3, label="Ordinary")
axes[1].plot(x, [r["ema_miou"] for r in validation], "--^", color="#777777",
             ms=3, lw=1.1, label="EMA")
axes[1].set(ylabel="Validation mIoU", ylim=(.785, .835), title="(b) Validation and selection")
axes[1].text(10.7, .787, "Frozen: 11k", rotation=90, va="bottom", ha="right", fontsize=9)
axes[1].legend(frameon=False, loc="lower right")
for axis in axes:
    axis.set(xlabel=r"Iteration ($\times 10^3$)", xlim=(0,18.5))
    axis.set_xticks([0, 3, 6, 9, 12, 15, 18])
    axis.tick_params(top=True, right=True)
    axis.axvline(best_iter/1000, color="black", ls="--", lw=1)
fig.tight_layout()
save(fig, "SegFormer_B0_training_validation_two_panels")
print(json.dumps(metadata, ensure_ascii=False, indent=2))

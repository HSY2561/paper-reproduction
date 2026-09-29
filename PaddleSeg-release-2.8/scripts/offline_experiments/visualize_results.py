import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import ensure_dir, load_json


METRIC_COLUMNS = ["Precision", "Recall", "F1", "IoU", "mIoU"]
EFFICIENCY_COLUMNS = ["Params_M", "FPS"]
PALETTE = {
    "Precision": "#2f4858",
    "Recall": "#33658a",
    "F1": "#86bbd8",
    "IoU": "#f6ae2d",
    "mIoU": "#f26419",
    "Params_M": "#5c7cfa",
    "FPS": "#2b9348",
}


def load_summary_rows(result_root):
    rows = []
    result_root = Path(result_root)
    for summary_path in sorted(result_root.rglob("summary_row.json")):
        row = load_json(summary_path, default={})
        if row:
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["Experiment", *METRIC_COLUMNS, *EFFICIENCY_COLUMNS])
    for column in METRIC_COLUMNS + EFFICIENCY_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _sorted_frame(frame):
    if frame.empty:
        return frame
    return frame.sort_values(by=["F1", "IoU", "FPS"], ascending=[False, False, False]).reset_index(drop=True)


def plot_metrics_figure(frame, title, output_path):
    frame = _sorted_frame(frame)
    fig_height = max(6, 0.45 * max(len(frame), 1) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    if frame.empty:
        ax.text(0.5, 0.5, "No completed results", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    else:
        y = np.arange(len(frame))
        total_width = 0.8
        bar_height = total_width / len(METRIC_COLUMNS)
        offsets = np.linspace(-total_width / 2 + bar_height / 2, total_width / 2 - bar_height / 2, len(METRIC_COLUMNS))
        for offset, metric in zip(offsets, METRIC_COLUMNS):
            ax.barh(y + offset, frame[metric].fillna(0), height=bar_height, color=PALETTE[metric], label=metric)
        ax.set_yticks(y)
        ax.set_yticklabels(frame["Experiment"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score")
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.legend(loc="lower right", ncol=len(METRIC_COLUMNS))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def plot_efficiency_figure(frame, title, output_path):
    frame = _sorted_frame(frame)
    fig_height = max(6, 0.45 * max(len(frame), 1) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_height), sharey=True)
    if frame.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No completed results", ha="center", va="center", fontsize=14)
            ax.set_axis_off()
    else:
        y = np.arange(len(frame))
        for ax, metric in zip(axes, EFFICIENCY_COLUMNS):
            ax.barh(y, frame[metric].fillna(0), color=PALETTE[metric])
            ax.set_title(metric)
            ax.grid(axis="x", linestyle="--", alpha=0.25)
            ax.set_xlabel(metric)
            ax.set_yticks(y)
            ax.set_yticklabels(frame["Experiment"])
            ax.invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def generate_all_figures(ablation_root, crack500_root, output_dir):
    output_dir = ensure_dir(output_dir)
    ablation_frame = load_summary_rows(ablation_root)
    crack500_frame = load_summary_rows(crack500_root)
    paths = {
        "ablation_metrics": plot_metrics_figure(
            ablation_frame,
            "Ablation Metrics Comparison",
            output_dir / "ablation_metrics.png",
        ),
        "ablation_efficiency": plot_efficiency_figure(
            ablation_frame,
            "Ablation Efficiency Comparison",
            output_dir / "ablation_efficiency.png",
        ),
        "crack500_metrics": plot_metrics_figure(
            crack500_frame,
            "Crack500 Metrics Comparison",
            output_dir / "crack500_metrics.png",
        ),
        "crack500_efficiency": plot_efficiency_figure(
            crack500_frame,
            "Crack500 Efficiency Comparison",
            output_dir / "crack500_efficiency.png",
        ),
    }
    return {key: str(path) for key, path in paths.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize offline experiment results.")
    parser.add_argument("--ablation_root", default="output/offline_experiments/ablation")
    parser.add_argument("--crack500_root", default="output/offline_experiments/crack500")
    parser.add_argument("--output_dir", default="output/offline_experiments/figures")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = generate_all_figures(args.ablation_root, args.crack500_root, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

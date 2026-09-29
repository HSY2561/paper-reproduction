import argparse
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import dump_json, ensure_dir, load_json


SUMMARY_COLUMNS = [
    "Experiment",
    "Dataset",
    "Config",
    "BestModelPath",
    "Precision",
    "Recall",
    "F1",
    "IoU",
    "mIoU",
    "Params_M",
    "FPS",
]


def build_summary_row(experiment_dir):
    experiment_dir = Path(experiment_dir)
    metrics = load_json(experiment_dir / "metrics.json")
    stats = load_json(experiment_dir / "model_stats.json")
    fps = load_json(experiment_dir / "fps.json")
    metric_block = metrics.get("metrics", {})

    precision = metric_block.get("foreground_precision")
    recall = metric_block.get("foreground_recall")
    f1 = metric_block.get("foreground_f1")
    if f1 is None and precision is not None and recall is not None:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    row = {
        "Experiment": metrics.get("experiment_name", experiment_dir.name),
        "Dataset": metrics.get("dataset", experiment_dir.parent.name),
        "Config": metrics.get("config_path", ""),
        "BestModelPath": metrics.get("best_model_path", ""),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": metric_block.get("foreground_iou"),
        "mIoU": metrics.get("miou"),
        "Params_M": stats.get("params_m"),
        "FPS": fps.get("fps"),
    }
    dump_json(row, experiment_dir / "summary_row.json")
    return row


def summarize_result_roots(result_roots, output_dir, table_prefix):
    rows = []
    for root in result_roots:
        root = Path(root)
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob("metrics.json")):
            rows.append(build_summary_row(metrics_path.parent))

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        frame = frame[SUMMARY_COLUMNS]

    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / f"{table_prefix}.csv"
    xlsx_path = output_dir / f"{table_prefix}.xlsx"
    json_path = output_dir / f"{table_prefix}.json"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)
    dump_json(frame.to_dict(orient="records"), json_path)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize offline experiment results.")
    parser.add_argument("--result_roots", nargs="+", required=True, type=str)
    parser.add_argument("--output_dir", default="output/offline_experiments/tables", type=str)
    parser.add_argument("--table_prefix", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = summarize_result_roots(args.result_roots, args.output_dir, args.table_prefix)
    print(frame)


if __name__ == "__main__":
    main()

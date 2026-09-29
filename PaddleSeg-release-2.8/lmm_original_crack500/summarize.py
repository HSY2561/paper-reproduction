from pathlib import Path

from lmm_original_crack500.common import dump_json, load_json


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

    row = {
        "Experiment": metrics.get("experiment_name", experiment_dir.name),
        "Dataset": metrics.get("dataset", experiment_dir.parent.name),
        "Config": metrics.get("config_path", ""),
        "BestModelPath": metrics.get("best_model_path", ""),
        "Precision": metric_block.get("foreground_precision"),
        "Recall": metric_block.get("foreground_recall"),
        "F1": metric_block.get("foreground_f1"),
        "IoU": metric_block.get("foreground_iou"),
        "mIoU": metrics.get("miou"),
        "Params_M": stats.get("params_m"),
        "FPS": fps.get("fps"),
    }
    dump_json(row, experiment_dir / "summary_row.json")
    return row

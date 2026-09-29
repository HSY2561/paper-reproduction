import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.lmm_torch.common import dump_json
from experiments.lmm_torch.config import load_config
from experiments.lmm_torch.dataset import Crack500Dataset
from experiments.lmm_torch.metrics import confusion_matrix_from_masks, metrics_from_hist
from experiments.lmm_torch.model import LMMNet


def evaluate_model(model, dataloader, device, num_classes=2):
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["mask"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            hist += confusion_matrix_from_masks(
                preds.cpu().numpy(),
                labels.cpu().numpy(),
                num_classes=num_classes,
            )
    result = metrics_from_hist(hist, foreground_class=1)
    result["num_images"] = len(dataloader.dataset)
    return result


def build_eval_dataloader(cfg, list_path):
    dataset = Crack500Dataset(cfg, list_path=list_path, mode="val")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["num_workers"])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch LMM on Crack500.")
    parser.add_argument("--config", default="experiments/lmm_torch/configs/crack500_lmnet.yml")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--list_path", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.dataset_root:
        overrides["dataset_root"] = args.dataset_root
    if args.device:
        overrides["device"] = args.device
    cfg = load_config(args.config, overrides=overrides)
    device = torch.device(cfg["device"])

    model = LMMNet(num_classes=cfg["num_classes"]).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)

    list_path = args.list_path or cfg["test_list"]
    dataloader = build_eval_dataloader(cfg, list_path)
    results = evaluate_model(model, dataloader, device=device, num_classes=cfg["num_classes"])
    results["config_path"] = str(Path(args.config).as_posix())
    results["best_model_path"] = str(Path(args.model_path).as_posix())
    results["dataset"] = cfg["dataset_name"]
    results["experiment_name"] = cfg["experiment_name"]
    results["dataset_root"] = cfg["dataset_root"]
    results["list_path"] = str(Path(list_path).as_posix())
    dump_json(results, args.save_path)
    print(results)


if __name__ == "__main__":
    main()

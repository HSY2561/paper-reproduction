import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from lmm_original_crack500.common import dump_json
from lmm_original_crack500.config import load_config
from lmm_original_crack500.dataset import Crack500Dataset
from lmm_original_crack500.metrics import confusion_matrix_from_masks, metrics_from_hist
from lmm_original_crack500.model import build_model


def evaluate_model(model, dataloader, device, num_classes=2):
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["mask"].to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
            hist += confusion_matrix_from_masks(
                preds.cpu().numpy(),
                labels.long().cpu().numpy(),
                num_classes=num_classes,
            )
    result = metrics_from_hist(hist, foreground_class=1)
    result["num_images"] = len(dataloader.dataset)
    return result


def build_eval_dataloader(cfg, list_path):
    dataset = Crack500Dataset(cfg, list_path=list_path, mode="val")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["num_workers"])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate original-source LM_Net on Crack500.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--list_path", default=None)
    parser.add_argument("--dataset_name", default="Crack500")
    parser.add_argument("--experiment_name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg["device"])
    model = build_model(cfg["model_type"]).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)

    list_path = args.list_path or cfg["test_list"]
    dataloader = build_eval_dataloader(cfg, list_path)
    result = evaluate_model(model, dataloader, device=device, num_classes=cfg["num_classes"])
    result.update(
        {
            "dataset": args.dataset_name,
            "dataset_root": cfg["dataset_root"],
            "list_path": list_path,
            "experiment_name": args.experiment_name or cfg["experiment_name"],
            "config_path": args.config,
            "best_model_path": args.model_path,
        }
    )
    dump_json(result, args.save_path)
    print(result)


if __name__ == "__main__":
    main()

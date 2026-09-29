import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.lmm_torch.common import ensure_dir, set_random_seed
from experiments.lmm_torch.config import load_config
from experiments.lmm_torch.dataset import Crack500Dataset
from experiments.lmm_torch.evaluate import evaluate_model
from experiments.lmm_torch.model import LMMNet


def _build_optimizer(model, cfg):
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr_scheduler"]["learning_rate"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )


def _build_train_loader(cfg):
    dataset = Crack500Dataset(cfg, list_path=cfg["train_list"], mode="train")
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=False,
    )


def _build_val_loader(cfg):
    dataset = Crack500Dataset(cfg, list_path=cfg["val_list"], mode="val")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["num_workers"])


def _current_lr(cfg, current_iter):
    lr_cfg = cfg["lr_scheduler"]
    base_lr = lr_cfg["learning_rate"]
    warmup_iters = lr_cfg.get("warmup_iters", 0)
    if current_iter <= warmup_iters and warmup_iters > 0:
        start_lr = lr_cfg.get("warmup_start_lr", 1.0e-5)
        alpha = current_iter / max(1, warmup_iters)
        return start_lr + alpha * (base_lr - start_lr)

    end_lr = lr_cfg.get("end_lr", 0.0)
    power = lr_cfg.get("power", 0.9)
    progress = (current_iter - warmup_iters) / max(1, cfg["iters"] - warmup_iters)
    progress = min(max(progress, 0.0), 1.0)
    return (base_lr - end_lr) * math.pow(1.0 - progress, power) + end_lr


def _save_checkpoint(output_dir, iteration, model, optimizer, best_miou=None):
    save_dir = ensure_dir(Path(output_dir) / f"iter_{iteration}")
    torch.save(
        {
            "iter": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
        },
        save_dir / "model.pt",
    )


def _save_best_model(output_dir, iteration, model, optimizer, best_miou):
    output_dir = ensure_dir(output_dir)
    torch.save(
        {
            "iter": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
        },
        Path(output_dir) / "best_model.pt",
    )


def train(cfg):
    set_random_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    output_dir = ensure_dir(cfg["output_dir"])
    do_eval_during_train = cfg.get("do_eval_during_train", True)
    accumulation_steps = max(1, int(cfg.get("accumulation_steps", 1)))

    model = LMMNet(num_classes=cfg["num_classes"]).to(device)
    optimizer = _build_optimizer(model, cfg)
    train_loader = _build_train_loader(cfg)
    val_loader = _build_val_loader(cfg) if do_eval_during_train else None
    train_iter = iter(train_loader)

    best_miou = -1.0
    start_time = time.perf_counter()

    for current_iter in range(1, cfg["iters"] + 1):
        lr = _current_lr(cfg, current_iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        optimizer.zero_grad()
        accumulated_loss = 0.0
        for _ in range(accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images = batch["image"].to(device)
            labels = batch["mask"].to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels.long())
            (loss / accumulation_steps).backward()
            accumulated_loss += loss.item()
        optimizer.step()

        if current_iter % cfg["log_interval"] == 0 or current_iter == 1:
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / current_iter
            eta_seconds = avg_time * max(0, cfg["iters"] - current_iter)
            print(
                f"[TRAIN] iter: {current_iter}/{cfg['iters']}, "
                f"loss: {accumulated_loss / accumulation_steps:.4f}, "
                f"lr: {lr:.6f}, accum: {accumulation_steps}, ETA {eta_seconds:.0f}s",
                flush=True,
            )

        if current_iter % cfg["save_interval"] == 0 or current_iter == cfg["iters"]:
            _save_checkpoint(output_dir, current_iter, model, optimizer, best_miou=best_miou)
            if do_eval_during_train:
                eval_results = evaluate_model(model, val_loader, device=device, num_classes=cfg["num_classes"])
                miou = eval_results["miou"]
                print(
                    f"[EVAL] iter: {current_iter}/{cfg['iters']} "
                    f"mIoU: {miou:.4f} IoU: {eval_results['metrics']['foreground_iou']:.4f}",
                    flush=True,
                )
                if miou >= best_miou:
                    best_miou = miou
                    _save_best_model(output_dir, current_iter, model, optimizer, best_miou=best_miou)
                    print(
                        f"[EVAL] The model with the best validation mIoU ({best_miou:.4f}) "
                        f"was saved at iter {current_iter}.",
                        flush=True,
                    )
            else:
                _save_best_model(output_dir, current_iter, model, optimizer, best_miou=best_miou)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch LMM on Crack500.")
    parser.add_argument("--config", default="experiments/lmm_torch/configs/crack500_lmnet.yml")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_val", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.iters is not None:
        overrides["iters"] = args.iters
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.accumulation_steps is not None:
        overrides["accumulation_steps"] = args.accumulation_steps
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.skip_val:
        overrides["do_eval_during_train"] = False
    cfg = load_config(args.config, overrides=overrides)
    train(cfg)


if __name__ == "__main__":
    main()

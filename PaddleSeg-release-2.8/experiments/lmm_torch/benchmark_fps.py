import argparse
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.lmm_torch.common import dump_json
from experiments.lmm_torch.config import load_config
from experiments.lmm_torch.model import LMMNet


def benchmark_model(model, device, input_shape, warmup=10, repeats=50):
    model.eval()
    sample = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / repeats) * 1000.0
    fps = 0.0 if elapsed == 0 else repeats / elapsed
    return {"fps": float(fps), "avg_latency_ms": float(avg_latency_ms)}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LMM Torch FPS.")
    parser.add_argument("--config", default="experiments/lmm_torch/configs/crack500_lmnet.yml")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input_shape", nargs=4, type=int, default=[1, 3, 720, 720])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {"device": args.device} if args.device else None
    cfg = load_config(args.config, overrides=overrides)
    device = torch.device(cfg["device"])

    model = LMMNet(num_classes=cfg["num_classes"]).to(device)
    if args.model_path:
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)

    result = benchmark_model(
        model,
        device=device,
        input_shape=args.input_shape,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result.update(
        {
            "experiment_name": cfg["experiment_name"],
            "dataset": cfg["dataset_name"],
            "config_path": str(Path(args.config).as_posix()),
            "input_shape": args.input_shape,
        }
    )
    dump_json(result, args.save_path)
    print(result)


if __name__ == "__main__":
    main()

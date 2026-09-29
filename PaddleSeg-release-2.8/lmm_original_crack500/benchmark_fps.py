import argparse
import time

import torch

from lmm_original_crack500.common import dump_json
from lmm_original_crack500.config import load_config
from lmm_original_crack500.model import build_model


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
    parser = argparse.ArgumentParser(description="Benchmark original-source LM_Net FPS.")
    parser.add_argument("--config", required=True)
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
    model = build_model(cfg["model_type"]).to(device)
    if args.model_path:
        state = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
    result = benchmark_model(model, device=device, input_shape=args.input_shape, warmup=args.warmup, repeats=args.repeats)
    result.update(
        {
            "experiment_name": cfg["experiment_name"],
            "dataset": cfg["dataset_name"],
            "config_path": args.config,
            "input_shape": args.input_shape,
        }
    )
    dump_json(result, args.save_path)
    print(result)


if __name__ == "__main__":
    main()

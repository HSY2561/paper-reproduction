import argparse
import time
from pathlib import Path

import paddle

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import utils

from scripts.offline_experiments.common import dump_json


def _sync_if_needed(device):
    if device == "gpu" and paddle.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()


def benchmark_model(model, device, input_shape, warmup=10, repeats=50, precision="fp32"):
    model.eval()
    sample = paddle.randn(input_shape)
    with paddle.no_grad():
        for _ in range(warmup):
            if precision == "fp16":
                with paddle.amp.auto_cast(enable=True):
                    model(sample)
            else:
                model(sample)
        _sync_if_needed(device)

        start = time.perf_counter()
        for _ in range(repeats):
            if precision == "fp16":
                with paddle.amp.auto_cast(enable=True):
                    model(sample)
            else:
                model(sample)
        _sync_if_needed(device)
        elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / repeats) * 1000.0
    fps = 0.0 if elapsed == 0 else repeats / elapsed
    return {"fps": float(fps), "avg_latency_ms": float(avg_latency_ms)}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FPS for offline experiments.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"], type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--input_shape", nargs=4, type=int, default=[1, 3, 720, 720])
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--repeats", default=50, type=int)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"], type=str)
    parser.add_argument("--experiment_name", default="", type=str)
    parser.add_argument("--dataset_name", default="", type=str)
    parser.add_argument("--opts", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args.config, opts=args.opts)
    utils.set_device(args.device)
    builder = SegBuilder(cfg)
    model = builder.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)

    result = benchmark_model(
        model,
        device=args.device,
        input_shape=args.input_shape,
        warmup=args.warmup,
        repeats=args.repeats,
        precision=args.precision,
    )
    result.update(
        {
            "experiment_name": args.experiment_name,
            "dataset": args.dataset_name,
            "config_path": str(Path(args.config).as_posix()),
            "input_shape": args.input_shape,
            "precision": args.precision,
        }
    )
    dump_json(result, args.save_path)
    print(result)


if __name__ == "__main__":
    main()

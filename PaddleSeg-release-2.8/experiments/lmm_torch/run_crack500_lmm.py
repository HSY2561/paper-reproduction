import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.lmm_torch.common import dump_yaml, find_best_model_path, load_yaml, run_command
from experiments.lmm_torch.config import load_config
from experiments.lmm_torch.summarize import build_summary_row


def materialize_runtime_config(config_path, output_dir, overrides):
    cfg = load_config(config_path, overrides=overrides)
    runtime_path = Path(output_dir) / "runtime_config.yml"
    dump_yaml(cfg, runtime_path)
    return runtime_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyTorch LMM Crack500 experiment.")
    parser.add_argument("--config", default="experiments/lmm_torch/configs/crack500_lmnet.yml")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_fps", action="store_true")
    parser.add_argument("--resume_failed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    output_dir = args.output_dir or base_cfg["output_dir"]
    overrides = {"output_dir": output_dir}
    if args.iters is not None:
        overrides["iters"] = args.iters
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.accumulation_steps is not None:
        overrides["accumulation_steps"] = args.accumulation_steps
    if args.device is not None:
        overrides["device"] = args.device

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    runtime_config = materialize_runtime_config(args.config, output_dir, overrides)

    best_model_path = Path(output_dir) / "best_model.pt"
    if not args.skip_train and not (args.resume_failed and best_model_path.exists()):
        run_command(
            [sys.executable, "experiments/lmm_torch/train.py", "--config", str(runtime_config)],
            Path(output_dir) / "train.log",
            Path.cwd(),
        )

    model_path = Path(args.model_path) if args.model_path else None
    if model_path is None and (not args.skip_eval or not args.skip_fps):
        model_path = find_best_model_path(output_dir)

    metrics_path = Path(output_dir) / "metrics.json"
    if not args.skip_eval and not (args.resume_failed and metrics_path.exists()):
        if model_path is None:
            raise FileNotFoundError("Evaluation requires --model_path or a generated best_model.pt checkpoint.")
        run_command(
            [
                sys.executable,
                "experiments/lmm_torch/evaluate.py",
                "--config",
                str(runtime_config),
                "--model_path",
                str(model_path),
                "--save_path",
                str(metrics_path),
            ],
            Path(output_dir) / "eval_test.log",
            Path.cwd(),
        )

    stats_path = Path(output_dir) / "model_stats.json"
    if not (args.resume_failed and stats_path.exists()):
        run_command(
            [
                sys.executable,
                "experiments/lmm_torch/model_stats.py",
                "--config",
                str(runtime_config),
                "--save_path",
                str(stats_path),
            ],
            Path(output_dir) / "model_stats.log",
            Path.cwd(),
        )

    fps_path = Path(output_dir) / "fps.json"
    if not args.skip_fps and not (args.resume_failed and fps_path.exists()):
        if model_path is None:
            raise FileNotFoundError("FPS benchmark requires --model_path or a generated best_model.pt checkpoint.")
        run_command(
            [
                sys.executable,
                "experiments/lmm_torch/benchmark_fps.py",
                "--config",
                str(runtime_config),
                "--model_path",
                str(model_path),
                "--save_path",
                str(fps_path),
            ],
            Path(output_dir) / "fps.log",
            Path.cwd(),
        )

    if metrics_path.exists() and stats_path.exists():
        build_summary_row(output_dir)


if __name__ == "__main__":
    main()

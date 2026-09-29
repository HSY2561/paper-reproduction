import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.run_public_benchmarks import (
    build_public_benchmark_specs,
    materialize_public_benchmark_configs,
)
from torch_crack_benchmark.common import run_command
from torch_crack_benchmark.summarize import summarize_result_roots


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyTorch crack benchmarks on prepared datasets.")
    parser.add_argument("--dataset", required=True, choices=["crackseg9k", "crack500_kaggle"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["crackformer_ii", "deepcrack", "efficientnet", "mobilenetv3"],
    )
    parser.add_argument("--resume_failed", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--table_prefix", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    materialize_public_benchmark_configs()
    specs = build_public_benchmark_specs()[args.dataset]["torch"]
    selected_roots = []
    for model_name in args.models:
        if model_name not in specs:
            raise ValueError(f"Unknown benchmark model: {model_name}")
        spec = specs[model_name]
        output_dir = spec["output_dir"]
        command = [sys.executable, "-m", "torch_crack_benchmark.run", "--config", spec["config_path"], "--output_dir", output_dir]
        if args.resume_failed:
            command.append("--resume_failed")
        if args.device:
            command.extend(["--device", args.device])
        run_command(command, Path(output_dir) / "run.log", Path.cwd())
        selected_roots.append(Path(output_dir))

    summarize_result_roots(
        selected_roots,
        Path("output/offline_experiments_torch/tables"),
        args.table_prefix or f"{args.dataset}_torch_benchmarks",
    )


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import dump_yaml, find_best_model_path, load_yaml, run_command
from scripts.offline_experiments.run_public_benchmarks import (
    build_public_benchmark_specs,
    materialize_public_benchmark_configs,
)
from scripts.offline_experiments.summarize_results import build_summary_row, summarize_result_roots


DEFAULT_MODELS = ["pspnet", "unet", "shufflenetv2", "deepcrack"]


def _materialize_runtime_config(config_path, output_dir):
    cfg = load_yaml(config_path)
    cfg["save_dir"] = str(Path(output_dir).as_posix())
    runtime_path = Path(output_dir) / "runtime_config.yml"
    dump_yaml(cfg, runtime_path)
    return runtime_path


def _run_paddle_experiment(model_name, spec, dataset_name, args):
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = _materialize_runtime_config(spec["config_path"], output_dir)
    experiment_name = f"{model_name}_crack500_kaggle"

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "tools/train.py",
            "--config",
            str(runtime_config),
            "--save_dir",
            str(output_dir),
            "--device",
            args.device,
            "--do_eval",
            "--num_workers",
            str(args.num_workers),
            "--save_interval",
            str(args.save_interval),
            "--log_iters",
            str(args.log_iters),
        ]
        if args.iters is not None:
            train_cmd.extend(["--iters", str(args.iters)])
        if args.batch_size is not None:
            train_cmd.extend(["--batch_size", str(args.batch_size)])
        run_command(train_cmd, output_dir / "train.log", Path.cwd())

    model_path = find_best_model_path(output_dir)

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "scripts/offline_experiments/evaluate_experiment.py",
            "--config",
            str(runtime_config),
            "--model_path",
            str(model_path),
            "--save_path",
            str(output_dir / "metrics.json"),
            "--device",
            args.device,
            "--dataset_root",
            spec["dataset_root"],
            "--list_path",
            f"{spec['dataset_root']}/test.txt",
            "--dataset_name",
            dataset_name,
            "--experiment_name",
            experiment_name,
            "--num_workers",
            str(args.num_workers),
        ]
        run_command(eval_cmd, output_dir / "eval_test.log", Path.cwd())

    stats_cmd = [
        sys.executable,
        "scripts/offline_experiments/model_stats.py",
        "--config",
        str(runtime_config),
        "--save_path",
        str(output_dir / "model_stats.json"),
        "--experiment_name",
        experiment_name,
        "--dataset_name",
        dataset_name,
    ]
    run_command(stats_cmd, output_dir / "model_stats.log", Path.cwd())

    if not args.skip_fps:
        fps_cmd = [
            sys.executable,
            "scripts/offline_experiments/benchmark_fps.py",
            "--config",
            str(runtime_config),
            "--model_path",
            str(model_path),
            "--save_path",
            str(output_dir / "fps.json"),
            "--device",
            args.device,
            "--experiment_name",
            experiment_name,
            "--dataset_name",
            dataset_name,
        ]
        run_command(fps_cmd, output_dir / "fps.log", Path.cwd())

    if (output_dir / "metrics.json").exists() and (output_dir / "model_stats.json").exists() and (output_dir / "fps.json").exists():
        build_summary_row(output_dir)


def _run_torch_experiment(model_name, spec, args):
    command = [
        sys.executable,
        "-m",
        "torch_crack_benchmark.run",
        "--config",
        spec["config_path"],
        "--output_dir",
        spec["output_dir"],
    ]
    if args.skip_train:
        command.append("--skip_train")
    if args.skip_eval:
        command.append("--skip_eval")
    if args.skip_fps:
        command.append("--skip_fps")
    if args.iters is not None:
        command.extend(["--iters", str(args.iters)])
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    command.extend(["--device", "cuda" if args.device == "gpu" else "cpu"])
    run_command(command, Path(spec["output_dir"]) / "run.log", Path.cwd())


def parse_args():
    parser = argparse.ArgumentParser(description="Run Crack500-Kaggle hybrid benchmarks.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"], type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--iters", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--log_iters", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_fps", action="store_true")
    parser.add_argument("--table_prefix", default="crack500_kaggle_hybrid_benchmarks", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    materialize_public_benchmark_configs()
    specs = build_public_benchmark_specs()["crack500_kaggle"]
    selected_roots = []

    for model_name in args.models:
        if model_name in specs["paddle"]:
            spec = specs["paddle"][model_name]
            _run_paddle_experiment(model_name, spec, specs["dataset_name"], args)
            selected_roots.append(Path(spec["output_dir"]))
            continue
        if model_name in specs["torch"]:
            spec = specs["torch"][model_name]
            _run_torch_experiment(model_name, spec, args)
            selected_roots.append(Path(spec["output_dir"]))
            continue
        raise ValueError(f"Unknown Crack500-Kaggle benchmark model: {model_name}")

    summarize_result_roots(
        selected_roots,
        Path("output/offline_experiments_torch/tables"),
        args.table_prefix,
    )


if __name__ == "__main__":
    main()

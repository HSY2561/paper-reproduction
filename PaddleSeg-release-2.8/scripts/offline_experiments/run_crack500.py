import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import (
    dump_yaml,
    find_best_model_path,
    load_json,
    load_yaml,
    run_command,
)
from scripts.offline_experiments.generate_ablation_configs import generate_all_configs
from scripts.offline_experiments.prepare_datasets import copy_custom_dataset, prepare_crack500_lists
from scripts.offline_experiments.summarize_results import build_summary_row, summarize_result_roots


EXPERIMENTS = {
    "hrsegnet_b16": "configs/offline_experiments/crack500/hrsegnet_b16.yml",
    "hrsegnet_b16_a": "configs/offline_experiments/crack500/hrsegnet_b16_a.yml",
    "hrsegnet_b32": "configs/offline_experiments/crack500/hrsegnet_b32.yml",
    "hrsegnet_b32_a": "configs/offline_experiments/crack500/hrsegnet_b32_a.yml",
    "hrsegnet_b64": "configs/offline_experiments/crack500/hrsegnet_b64.yml",
    "hrsegnet_b64_a": "configs/offline_experiments/crack500/hrsegnet_b64_a.yml",
    "hrsegnet_b16_a_aspp32": "configs/offline_experiments/crack500/hrsegnet_b16_a_aspp32.yml",
    "hrsegnet_b16_a_aspp64": "configs/offline_experiments/crack500/hrsegnet_b16_a_aspp64.yml",
    "hrsegnet_b16_a_aspp96": "configs/offline_experiments/crack500/hrsegnet_b16_a_aspp96.yml",
    "hrsegnet_b16_a_aspp128": "configs/offline_experiments/crack500/hrsegnet_b16_a_aspp128.yml",
    "unet": "configs/offline_experiments/crack500/unet.yml",
    "deeplabv3p": "configs/offline_experiments/crack500/deeplabv3p.yml",
    "pspnet": "configs/offline_experiments/crack500/pspnet.yml",
    "segformer_b0": "configs/offline_experiments/crack500/segformer_b0.yml",
    "lmnet": "configs/offline_experiments/crack500/lmnet.yml",
}

DEFAULT_MODELS = [name for name in EXPERIMENTS.keys() if name != "lmnet"]


def _locate_model_path(experiment_dir):
    try:
        return find_best_model_path(experiment_dir)
    except FileNotFoundError:
        return None


def _stages_completed(experiment_dir):
    experiment_dir = Path(experiment_dir)
    return {
        "model": _locate_model_path(experiment_dir) is not None,
        "eval": (experiment_dir / "metrics.json").exists(),
        "stats": (experiment_dir / "model_stats.json").exists(),
        "fps": (experiment_dir / "fps.json").exists(),
        "summary": (experiment_dir / "summary_row.json").exists(),
    }


def _normalized_path_text(path_text):
    return str(Path(path_text).as_posix())


def _metrics_match_eval_target(experiment_dir, dataset_root, list_path):
    metrics_path = Path(experiment_dir) / "metrics.json"
    if not metrics_path.exists():
        return False
    metrics = load_json(metrics_path, default={})
    expected_root = _normalized_path_text(dataset_root)
    expected_list = _normalized_path_text(list_path)
    actual_root = _normalized_path_text(metrics.get("dataset_root", ""))
    actual_list = _normalized_path_text(metrics.get("list_path", ""))
    if actual_root != expected_root:
        return False
    if actual_list != expected_list:
        return False
    return int(metrics.get("num_images", 0) or 0) > 0


def build_train_override_args(dataset_root, train_list=None, val_list=None):
    opts = []
    if dataset_root:
        opts.extend(
            [
                f"train_dataset.dataset_root={dataset_root}",
                f"val_dataset.dataset_root={dataset_root}",
            ]
        )
    if train_list:
        opts.append(f"train_dataset.train_path={train_list}")
    if val_list:
        opts.append(f"val_dataset.val_path={val_list}")
    return opts


def crack500_lists_need_refresh(crack500_root):
    crack500_root = Path(crack500_root)
    train_txt = crack500_root / "train.txt"
    if not train_txt.exists():
        return True

    first_line = ""
    for line in train_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            first_line = line
            break
    if not first_line:
        return True
    parts = first_line.replace("\\", "/").split()
    if len(parts) != 2:
        return True
    image_rel, mask_rel = parts
    image_path = crack500_root / image_rel
    mask_path = crack500_root / mask_rel
    if not image_path.exists() or not mask_path.exists():
        return True
    normalized_mask_path = mask_rel.replace("\\", "/")
    return ("/masks_bin/" not in normalized_mask_path) and ("/Annotations_bin/" not in normalized_mask_path)


def ensure_datasets(prepare_data, custom_src, custom_root, crack500_root):
    custom_train = Path(custom_root) / "train.txt"
    if prepare_data or not custom_train.exists():
        copy_custom_dataset(custom_src, custom_root, dry_run=False)
    if prepare_data or crack500_lists_need_refresh(crack500_root):
        prepare_crack500_lists(
            crack500_root,
            dry_run=False,
            normalize_masks=True,
            normalized_mask_dir_name="masks_bin",
        )


def ensure_configs(custom_root, crack500_root):
    generate_all_configs(
        baseline_config_path="configs/aa/hrsegnetb16.yml",
        final_config_path="configs/aa/hrsegnetb16_asppdelite.yml",
        unet_config_path="configs/aa/unet.yml",
        deeplab_config_path="configs/aa/deeplabv3p.yml",
        pspnet_config_path="configs/aa/pspnet.yml",
        lmm_config_path="configs/aa/lmnet.yml",
        segformer_config_path="configs/aa/segformer_b0_720.yml",
        custom_root=custom_root,
        crack500_root=crack500_root,
        output_root="configs/offline_experiments",
    )


def materialize_runtime_config(config_path, experiment_dir, train_crop_size=None):
    config = load_yaml(config_path)
    config["save_dir"] = str(experiment_dir).replace("\\", "/")
    if train_crop_size is not None:
        for transform in config.get("train_dataset", {}).get("transforms", []):
            if transform.get("type") == "RandomPaddingCrop":
                transform["crop_size"] = [train_crop_size, train_crop_size]
    runtime_config_path = Path(experiment_dir) / "runtime_config.yml"
    dump_yaml(config, runtime_config_path)
    return runtime_config_path


def run_single_experiment(name, config_path, args):
    experiment_dir = Path(args.output_root) / "crack500" / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = materialize_runtime_config(
        config_path,
        experiment_dir,
        train_crop_size=args.train_crop_size,
    )
    eval_list_path = args.test_list or f"{args.crack500_root}/test.txt"
    stages = _stages_completed(experiment_dir)
    summary_needs_refresh = False

    if args.resume_failed and not args.skip_train and stages["model"]:
        print(f"[RESUME] Skip train for {name}: model checkpoint already exists.")
    elif not args.skip_train:
        train_cmd = [
            sys.executable,
            "tools/train.py",
            "--config",
            str(runtime_config),
            "--save_dir",
            str(experiment_dir),
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
        train_opts = build_train_override_args(
            dataset_root=args.crack500_root,
            train_list=args.train_list,
            val_list=args.val_list,
        )
        if args.train_opts:
            train_opts.extend(args.train_opts)
        if train_opts:
            train_cmd.extend(["--opts", *train_opts])
        run_command(train_cmd, experiment_dir / "train.log", Path.cwd())
        stages = _stages_completed(experiment_dir)

    model_path = _locate_model_path(experiment_dir)
    if model_path is None:
        raise FileNotFoundError(
            f"No model checkpoint found for {name}. "
            "Run without --skip_train or remove --resume_failed for a fresh run."
        )

    skip_eval_stage = False
    if args.resume_failed and not args.skip_eval and stages["eval"]:
        if _metrics_match_eval_target(experiment_dir, args.crack500_root, eval_list_path):
            print(f"[RESUME] Skip eval for {name}: metrics.json already exists.")
            skip_eval_stage = True
        else:
            print(
                f"[RESUME] Re-run eval for {name}: existing metrics.json does not match "
                f"dataset_root={args.crack500_root}, list_path={eval_list_path}."
            )
    if not args.skip_eval and not skip_eval_stage:
        eval_cmd = [
            sys.executable,
            "scripts/offline_experiments/evaluate_experiment.py",
            "--config",
            str(runtime_config),
            "--model_path",
            str(model_path),
            "--save_path",
            str(experiment_dir / "metrics.json"),
            "--device",
            args.device,
            "--dataset_root",
            args.crack500_root,
            "--list_path",
            eval_list_path,
            "--dataset_name",
            "Crack500",
            "--experiment_name",
            name,
            "--num_workers",
            str(args.num_workers),
        ]
        run_command(eval_cmd, experiment_dir / "eval_test.log", Path.cwd())
        stages = _stages_completed(experiment_dir)
        summary_needs_refresh = True

    if args.resume_failed and stages["stats"]:
        print(f"[RESUME] Skip stats for {name}: model_stats.json already exists.")
    else:
        stats_cmd = [
            sys.executable,
            "scripts/offline_experiments/model_stats.py",
            "--config",
            str(runtime_config),
            "--save_path",
            str(experiment_dir / "model_stats.json"),
            "--dataset_name",
            "Crack500",
            "--experiment_name",
            name,
        ]
        run_command(stats_cmd, experiment_dir / "model_stats.log", Path.cwd())
        stages = _stages_completed(experiment_dir)
        summary_needs_refresh = True

    if args.resume_failed and not args.skip_fps and stages["fps"]:
        print(f"[RESUME] Skip fps for {name}: fps.json already exists.")
    elif not args.skip_fps:
        fps_cmd = [
            sys.executable,
            "scripts/offline_experiments/benchmark_fps.py",
            "--config",
            str(runtime_config),
            "--model_path",
            str(model_path),
            "--save_path",
            str(experiment_dir / "fps.json"),
            "--device",
            args.device,
            "--experiment_name",
            name,
            "--dataset_name",
            "Crack500",
        ]
        run_command(fps_cmd, experiment_dir / "fps.log", Path.cwd())
        stages = _stages_completed(experiment_dir)
        summary_needs_refresh = True

    if (
        args.resume_failed
        and stages["summary"]
        and (args.skip_eval or stages["eval"])
        and stages["stats"]
        and not summary_needs_refresh
    ):
        print(f"[RESUME] Keep existing summary for {name}.")
    else:
        build_summary_row(experiment_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Crack500 generalization experiments.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"], type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--iters", default=12000, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--log_iters", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_fps", action="store_true")
    parser.add_argument("--resume_failed", action="store_true")
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--custom_src", default="data", type=str)
    parser.add_argument("--custom_root", default="dataset/custom", type=str)
    parser.add_argument("--crack500_root", default="dataset/Crack500", type=str)
    parser.add_argument("--output_root", default="output/offline_experiments", type=str)
    parser.add_argument("--train_list", default=None, type=str)
    parser.add_argument("--val_list", default=None, type=str)
    parser.add_argument("--test_list", default=None, type=str)
    parser.add_argument("--train_opts", nargs="*", default=None)
    parser.add_argument("--train_crop_size", default=720, type=int)
    parser.add_argument("--summary_prefix", default="crack500_summary", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_datasets(args.prepare_data, args.custom_src, args.custom_root, args.crack500_root)
    ensure_configs(args.custom_root, args.crack500_root)

    for name in args.models:
        if name not in EXPERIMENTS:
            raise ValueError(f"Unknown Crack500 experiment: {name}")
        run_single_experiment(name, EXPERIMENTS[name], args)

    selected_result_roots = [Path(args.output_root) / "crack500" / name for name in args.models]
    summarize_result_roots(
        selected_result_roots,
        Path(args.output_root) / "tables",
        table_prefix=args.summary_prefix,
    )


if __name__ == "__main__":
    main()

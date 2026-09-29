import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import dump_yaml, find_best_model_path, load_yaml, run_command
from scripts.offline_experiments.generate_ablation_configs import generate_all_configs
from scripts.offline_experiments.prepare_datasets import copy_custom_dataset, prepare_crack500_lists
from scripts.offline_experiments.summarize_results import build_summary_row, summarize_result_roots


EXPERIMENTS = {
    "b16_baseline": "configs/offline_experiments/ablation/b16_baseline.yml",
    "b16_aspp": "configs/offline_experiments/ablation/b16_aspp.yml",
    "b16_decoder": "configs/offline_experiments/ablation/b16_decoder.yml",
    "b16_final": "configs/offline_experiments/ablation/b16_final.yml",
    "b32_baseline": "configs/offline_experiments/ablation/b32_baseline.yml",
    "b32_aspp": "configs/offline_experiments/ablation/b32_aspp.yml",
    "b32_decoder": "configs/offline_experiments/ablation/b32_decoder.yml",
    "b32_final": "configs/offline_experiments/ablation/b32_final.yml",
    "b64_baseline": "configs/offline_experiments/ablation/b64_baseline.yml",
    "b64_aspp": "configs/offline_experiments/ablation/b64_aspp.yml",
    "b64_decoder": "configs/offline_experiments/ablation/b64_decoder.yml",
    "b64_final": "configs/offline_experiments/ablation/b64_final.yml",
}


def _metric_text(value, float_fmt):
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return format(float(value), float_fmt)


def parse_aspp_rate_sets(raw_rate_sets):
    if not raw_rate_sets:
        return None
    parsed = []
    for rate_set_text in raw_rate_sets:
        tokens = [token.strip() for token in str(rate_set_text).split(",") if token.strip()]
        if not tokens:
            raise ValueError(f"Invalid aspp rate set: {rate_set_text}")
        try:
            rate_set = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(f"Invalid aspp rate set: {rate_set_text}") from exc
        if any(rate <= 0 for rate in rate_set):
            raise ValueError(f"ASPP rates must be positive integers: {rate_set_text}")
        parsed.append(rate_set)
    return parsed


def _aspp_experiment_name(base_name, rate_set, out_channels):
    rate_text = "-".join(str(rate) for rate in rate_set)
    return f"{base_name}_r{rate_text}_c{out_channels}"


def create_aspp_search_experiments(
    base_name,
    base_config_path,
    rate_sets,
    out_channels,
    generated_config_root="configs/offline_experiments/ablation_search",
):
    base_config = load_yaml(base_config_path)
    generated_config_root = Path(generated_config_root)
    generated_config_root.mkdir(parents=True, exist_ok=True)

    experiments = {}
    for rate_set in rate_sets:
        for channel in out_channels:
            if channel <= 0:
                raise ValueError(f"ASPP output channels must be positive: {channel}")
            experiment_name = _aspp_experiment_name(base_name, rate_set, channel)
            config = deepcopy(base_config)
            model_config = deepcopy(config.get("model", {}))
            model_config["aspp_rates"] = list(rate_set)
            model_config["aspp_out_channels"] = int(channel)
            config["model"] = model_config

            generated_path = generated_config_root / f"{experiment_name}.yml"
            dump_yaml(config, generated_path)
            experiments[experiment_name] = str(generated_path)
    return experiments


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


def ensure_datasets(prepare_data, custom_src, custom_root, crack500_root):
    custom_train = Path(custom_root) / "train.txt"
    if prepare_data or not custom_train.exists():
        copy_custom_dataset(custom_src, custom_root, dry_run=False)
    crack500_train = Path(crack500_root) / "train.txt"
    if prepare_data or not crack500_train.exists():
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
        b32_baseline_config_path="configs/aa/hrsegnetb32.yml",
        b32_final_config_path="configs/aa/hrsegnetb32_asppdelite.yml",
        b64_baseline_config_path="configs/aa/hrsegnetb64.yml",
        b64_final_config_path="configs/aa/hrsegnetb64_asppde.yml",
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
    experiment_dir = Path(args.output_root) / "ablation" / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = materialize_runtime_config(
        config_path,
        experiment_dir,
        train_crop_size=args.train_crop_size,
    )
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
            dataset_root=args.custom_root,
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

    if args.resume_failed and not args.skip_eval and stages["eval"]:
        print(f"[RESUME] Skip eval for {name}: metrics.json already exists.")
    elif not args.skip_eval:
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
            args.custom_root,
            "--list_path",
            args.test_list or f"{args.custom_root}/test.txt",
            "--dataset_name",
            "custom",
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
            "custom",
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
            "custom",
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
    parser = argparse.ArgumentParser(description="Run offline ablation experiments.")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"], type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--iters", default=None, type=int)
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
    parser.add_argument(
        "--aspp_rates",
        nargs="*",
        default=None,
        help="ASPP rate sets, each item is comma-separated. Example: 1,6,12,18 1,8,16,24",
    )
    parser.add_argument(
        "--aspp_out_channels",
        nargs="*",
        type=int,
        default=None,
        help="ASPP output channels candidates. Example: 32 64 96 128",
    )
    parser.add_argument(
        "--aspp_search_base",
        default="b16_final",
        choices=["b16_aspp", "b16_final", "b32_aspp", "b32_final"],
        type=str,
    )
    parser.add_argument("--aspp_search_include_default", action="store_true")
    parser.add_argument("--aspp_summary_prefix", default="ablation_aspp_search", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_datasets(args.prepare_data, args.custom_src, args.custom_root, args.crack500_root)
    ensure_configs(args.custom_root, args.crack500_root)

    experiments = dict(EXPERIMENTS)
    generated_search_experiments = {}
    if args.aspp_rates or args.aspp_out_channels:
        base_name = args.aspp_search_base
        base_config_path = experiments[base_name]
        base_model = load_yaml(base_config_path).get("model", {})
        rate_sets = parse_aspp_rate_sets(args.aspp_rates) or [base_model.get("aspp_rates", [1, 6, 12, 18])]
        out_channels = args.aspp_out_channels or [int(base_model.get("aspp_out_channels", 64))]
        generated_search_experiments = create_aspp_search_experiments(
            base_name=base_name,
            base_config_path=base_config_path,
            rate_sets=rate_sets,
            out_channels=out_channels,
        )
        if args.aspp_search_include_default:
            experiments.update(generated_search_experiments)
        else:
            experiments = generated_search_experiments

        print(f"[ASPP-SEARCH] Generated {len(generated_search_experiments)} experiments.")
        for name in generated_search_experiments:
            print(f"[ASPP-SEARCH] {name}")

    selected_models = args.models or list(experiments.keys())
    for name in selected_models:
        if name not in experiments:
            raise ValueError(f"Unknown ablation experiment: {name}")
        run_single_experiment(name, experiments[name], args)

    if generated_search_experiments:
        selected_result_roots = [Path(args.output_root) / "ablation" / name for name in selected_models]
        frame = summarize_result_roots(
            selected_result_roots,
            Path(args.output_root) / "tables",
            table_prefix=args.aspp_summary_prefix,
        )
        if frame.empty:
            print("[ASPP-SEARCH] No completed metrics found for summary.")
        else:
            ranked = frame.sort_values(by=["F1", "IoU", "FPS"], ascending=[False, False, False]).reset_index(
                drop=True
            )
            print("[ASPP-SEARCH] Top candidates:")
            for idx, row in ranked.head(5).iterrows():
                print(
                    f"  {idx + 1}. {row['Experiment']} | "
                    f"F1={_metric_text(row['F1'], '.6f')} "
                    f"IoU={_metric_text(row['IoU'], '.6f')} "
                    f"FPS={_metric_text(row['FPS'], '.3f')}"
                )


if __name__ == "__main__":
    main()

import argparse
import sys
from copy import deepcopy
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import dump_yaml, load_yaml


DATASET_SPECS = {
    "crackseg9k": {"dataset_name": "Crackseg9k", "dataset_root": "dataset/Crackseg9k"},
    "crack500_kaggle": {"dataset_name": "Crack500_kaggle", "dataset_root": "dataset/Crack500_kaggle"},
}

PADDLE_MODELS = {
    "crackseg9k": {
        "segformer_b0": {"template_path": "configs/aa/segformer_b0_720.yml"},
    },
    "crack500_kaggle": {
        "segformer_b0": {"template_path": "configs/aa/segformer_b0_720.yml"},
        "pspnet": {"template_path": "configs/aa/pspnet.yml"},
        "unet": {"template_path": "configs/aa/unet.yml"},
    },
}

PYTORCH_MODELS = {
    "crackseg9k": {
        "crackformer_ii": {"batch_size": 2, "accumulation_steps": 4},
        "deepcrack": {"batch_size": 2, "accumulation_steps": 4},
        "efficientnet": {"batch_size": 4, "accumulation_steps": 2},
        "mobilenetv3": {"batch_size": 4, "accumulation_steps": 1},
    },
    "crack500_kaggle": {
        "crackformer_ii": {"batch_size": 2, "accumulation_steps": 4},
        "deepcrack": {"batch_size": 2, "accumulation_steps": 4},
        "efficientnet": {"batch_size": 4, "accumulation_steps": 2},
        "mobilenetv3": {"batch_size": 4, "accumulation_steps": 1},
        "shufflenetv2": {"batch_size": 8, "accumulation_steps": 1, "iters": 12000},
    },
}


def build_public_benchmark_specs():
    specs = {}
    for dataset_key, dataset_spec in DATASET_SPECS.items():
        dataset_root = dataset_spec["dataset_root"]
        paddle_models = {}
        for model_name in PADDLE_MODELS.get(dataset_key, {}):
            paddle_output_root = "output/offline_experiments_torch" if dataset_key == "crack500_kaggle" else "output/offline_experiments"
            paddle_models[model_name] = {
                "dataset_root": dataset_root,
                "config_path": f"configs/offline_experiments/{dataset_key}/{model_name}.yml",
                "output_dir": f"{paddle_output_root}/{dataset_key}/{model_name}",
            }
        specs[dataset_key] = {
            "dataset_name": dataset_spec["dataset_name"],
            "dataset_root": dataset_root,
            "paddle": paddle_models,
            "torch": {
                model_name: {
                    "dataset_root": dataset_root,
                    "config_path": f"configs/torch_crack_benchmark/{dataset_key}/{model_name}.yml",
                    "output_dir": f"output/offline_experiments_torch/{dataset_key}/{model_name}",
                }
                for model_name in PYTORCH_MODELS.get(dataset_key, {})
            },
        }
    return specs


def _dataset_list_paths(dataset_root):
    return {
        "train_list": f"{dataset_root}/train.txt",
        "val_list": f"{dataset_root}/val.txt",
        "test_list": f"{dataset_root}/test.txt",
    }


def _build_paddle_config(base_config, output_dir, dataset_root):
    config = deepcopy(base_config)
    config["save_dir"] = f"./{Path(output_dir).as_posix()}"
    backbone = config.get("model", {}).get("backbone", {})
    pretrained = backbone.get("pretrained")
    if pretrained:
        pretrained_text = str(pretrained).replace("\\", "/")
        marker = "/pretrained/"
        if marker in pretrained_text:
            backbone["pretrained"] = pretrained_text.split(marker, 1)[1]
            backbone["pretrained"] = f"pretrained/{backbone['pretrained']}"
        else:
            backbone["pretrained"] = str(Path(pretrained).as_posix())
    config["train_dataset"]["dataset_root"] = dataset_root
    config["train_dataset"]["train_path"] = f"{dataset_root}/train.txt"
    config["val_dataset"]["dataset_root"] = dataset_root
    config["val_dataset"]["val_path"] = f"{dataset_root}/val.txt"
    return config


def _build_torch_config(dataset_key, dataset_root, model_name):
    model_cfg = PYTORCH_MODELS[dataset_key][model_name]
    return {
        "experiment_name": f"{model_name}_{dataset_key}",
        "model_type": model_name,
        "dataset_name": DATASET_SPECS[dataset_key]["dataset_name"],
        "dataset_root": dataset_root,
        **_dataset_list_paths(dataset_root),
        "num_classes": 2,
        "batch_size": model_cfg["batch_size"],
        "accumulation_steps": model_cfg["accumulation_steps"],
        "iters": model_cfg.get("iters", 18000),
        "save_interval": 1000,
        "log_interval": 10,
        "do_eval_during_train": True,
        "use_amp": True,
        "train_crop_size": 720,
        "min_scale_factor": 0.5,
        "max_scale_factor": 2.0,
        "scale_step_size": 0.25,
        "brightness_range": 0.5,
        "contrast_range": 0.5,
        "saturation_range": 0.5,
        "optimizer": {"type": "SGD", "momentum": 0.9, "weight_decay": 0.0005},
        "lr_scheduler": {
            "type": "PolynomialDecay",
            "learning_rate": 0.01,
            "end_lr": 0.0,
            "power": 0.9,
            "warmup_iters": 1000,
            "warmup_start_lr": 1.0e-5,
        },
        "early_stop": {"enabled": True, "metric": "miou", "patience": 8, "min_delta": 0.001},
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "num_workers": 0,
        "device": "cuda",
        "seed": 42,
        "output_dir": f"output/offline_experiments_torch/{dataset_key}/{model_name}",
    }


def materialize_public_benchmark_configs():
    specs = build_public_benchmark_specs()
    generated = []
    for dataset_key, dataset_spec in specs.items():
        dataset_root = dataset_spec["dataset_root"]
        for model_name, model_spec in dataset_spec["paddle"].items():
            base_config = load_yaml(PADDLE_MODELS[dataset_key][model_name]["template_path"])
            paddle_cfg = _build_paddle_config(base_config, model_spec["output_dir"], dataset_root)
            paddle_path = Path(model_spec["config_path"])
            dump_yaml(paddle_cfg, paddle_path)
            generated.append(paddle_path)
        for model_name, model_spec in dataset_spec["torch"].items():
            torch_cfg = _build_torch_config(dataset_key, dataset_root, model_name)
            torch_path = Path(model_spec["config_path"])
            dump_yaml(torch_cfg, torch_path)
            generated.append(torch_path)
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Generate relative-path public benchmark configs.")
    parser.add_argument("--generate_only", action="store_true")
    return parser.parse_args()


def main():
    parse_args()
    generated = materialize_public_benchmark_configs()
    print(f"Generated {len(generated)} benchmark configs.")


if __name__ == "__main__":
    main()

from copy import deepcopy

from lmm_original_crack500.common import load_yaml


def default_config():
    return {
        "experiment_name": "lmnet_original",
        "model_type": "lmnet",
        "dataset_name": "Crack500",
        "dataset_root": "dataset/Crack500",
        "train_list": "dataset/Crack500/train.txt",
        "val_list": "dataset/Crack500/val.txt",
        "test_list": "dataset/Crack500/test.txt",
        "num_classes": 2,
        "batch_size": 1,
        "accumulation_steps": 8,
        "iters": 12000,
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
        "optimizer": {
            "type": "SGD",
            "momentum": 0.9,
            "weight_decay": 0.0005,
        },
        "lr_scheduler": {
            "type": "PolynomialDecay",
            "learning_rate": 0.01,
            "end_lr": 0.0,
            "power": 0.9,
            "warmup_iters": 1000,
            "warmup_start_lr": 1.0e-5,
        },
        "normalize": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "num_workers": 0,
        "device": "cuda",
        "seed": 42,
        "output_dir": "output/offline_experiments/crack500/lmnet_original",
    }


def load_config(config_path=None, overrides=None):
    cfg = deepcopy(default_config())
    if config_path:
        loaded = load_yaml(config_path) or {}
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg

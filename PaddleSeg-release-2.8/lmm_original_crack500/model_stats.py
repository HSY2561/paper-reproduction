import argparse

from lmm_original_crack500.common import dump_json
from lmm_original_crack500.config import load_config
from lmm_original_crack500.model import build_model


def count_params_m(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    return total / 1e6


def parse_args():
    parser = argparse.ArgumentParser(description="Collect original-source LM_Net model stats.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--save_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg["model_type"])
    result = {
        "experiment_name": cfg["experiment_name"],
        "dataset": cfg["dataset_name"],
        "config_path": args.config,
        "params_m": float(count_params_m(model)),
    }
    dump_json(result, args.save_path)
    print(result)


if __name__ == "__main__":
    main()

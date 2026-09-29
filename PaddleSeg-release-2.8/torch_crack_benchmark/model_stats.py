import argparse

from torch_crack_benchmark.common import dump_json
from torch_crack_benchmark.config import load_config
from torch_crack_benchmark.model import build_model


def count_params_m(model):
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def parse_args():
    parser = argparse.ArgumentParser(description="Collect benchmark model parameter stats.")
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

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.lmm_torch.common import dump_json
from experiments.lmm_torch.config import load_config
from experiments.lmm_torch.model import LMMNet


def count_params_m(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    return total / 1e6


def parse_args():
    parser = argparse.ArgumentParser(description="Collect LMM Torch model stats.")
    parser.add_argument("--config", default="experiments/lmm_torch/configs/crack500_lmnet.yml")
    parser.add_argument("--save_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = LMMNet(num_classes=cfg["num_classes"])
    result = {
        "experiment_name": cfg["experiment_name"],
        "dataset": cfg["dataset_name"],
        "config_path": str(Path(args.config).as_posix()),
        "params_m": float(count_params_m(model)),
    }
    dump_json(result, args.save_path)
    print(result)


if __name__ == "__main__":
    main()

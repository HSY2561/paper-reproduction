import argparse
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paddleseg.cvlibs import Config, SegBuilder

from scripts.offline_experiments.common import dump_json


def count_params_m(model):
    total = 0
    for parameter in model.parameters():
        total += int(np.prod(parameter.shape))
    return total / 1e6


def parse_args():
    parser = argparse.ArgumentParser(description="Collect model stats for offline experiments.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--experiment_name", default="", type=str)
    parser.add_argument("--dataset_name", default="", type=str)
    parser.add_argument("--opts", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args.config, opts=args.opts)
    builder = SegBuilder(cfg)
    model = builder.model
    result = {
        "experiment_name": args.experiment_name,
        "dataset": args.dataset_name,
        "config_path": str(Path(args.config).as_posix()),
        "params_m": float(count_params_m(model)),
    }
    dump_json(result, args.save_path)
    print(result)


if __name__ == "__main__":
    main()

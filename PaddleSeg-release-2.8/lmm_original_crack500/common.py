import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml


def repo_root():
    return Path(__file__).resolve().parents[1]


def default_config_path():
    return repo_root() / "Lightweight-Modular-Model-main" / "experiments" / "crack500" / "configs" / "lmnet_crack500.yml"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return path


def load_yaml(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    return path


def find_best_model_path(output_dir):
    output_dir = Path(output_dir)
    best_model = output_dir / "best_model.pt"
    if best_model.exists():
        return best_model
    candidates = sorted(output_dir.glob("iter_*/model.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint found under {output_dir}")


def set_random_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def run_command(command, log_path, cwd):
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    env = os.environ.copy()
    root_text = str(Path(cwd).resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root_text if not existing_pythonpath else root_text + os.pathsep + existing_pythonpath
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code

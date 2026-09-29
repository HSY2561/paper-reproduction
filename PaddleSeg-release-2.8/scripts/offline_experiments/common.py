import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
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


def run_command(command, log_path, cwd):
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    env = os.environ.copy()
    repo_root = str(Path(cwd).resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        repo_root if not existing_pythonpath else repo_root + os.pathsep + existing_pythonpath
    )
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


def find_best_model_path(save_dir):
    save_dir = Path(save_dir)
    best_model = save_dir / "best_model" / "model.pdparams"
    if best_model.exists():
        return best_model

    iter_dirs = sorted(
        (
            path
            for path in save_dir.glob("iter_*")
            if path.is_dir() and path.name.split("_")[-1].isdigit()
        ),
        key=lambda item: int(item.name.split("_")[-1]),
        reverse=True,
    )
    for path in iter_dirs:
        candidate = path / "model.pdparams"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model weights found under {save_dir}")


def to_posix(path):
    return Path(path).as_posix()

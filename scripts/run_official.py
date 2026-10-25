"""Print (or explicitly execute) the pinned official train/val or exact-eval command."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def check_recipe(actual, expected, portable_paths, evaluation, path=()):
    """Only path materialization and explicit evaluation initialization may differ."""
    field = ".".join(path)
    if field in portable_paths:
        if evaluation and path[-1] == "pretrained" and actual is None:
            return
        if not isinstance(actual, str) or not actual or "${" in actual:
            raise ValueError(f"Expected a resolved local path at {field}.")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Recipe fields differ at {field or '<root>'}.")
        for key in expected:
            check_recipe(actual[key], expected[key], portable_paths, evaluation, path + (key,))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Recipe list differs at {field}.")
        for index, item in enumerate(expected):
            check_recipe(actual[index], item, portable_paths, evaluation, path + (str(index),))
    elif actual != expected:
        raise ValueError(f"Recipe value differs at {field}.")


def main():
    recipes = json.loads((ROOT / "configs" / "recipes.json").read_text(encoding="utf-8"))["recipes"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "val", "exact"))
    parser.add_argument("--recipe", required=True, choices=recipes)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime" / "PaddleSeg-2.10.0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="gpu", choices=("cpu", "gpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, help="Exact evaluator output prefix (not a filename extension).")
    parser.add_argument("--execute", action="store_true", help="Actually run the printed command; otherwise dry run only.")
    args = parser.parse_args()
    recipe = recipes[args.recipe]
    config_text = args.config.read_text(encoding="utf-8-sig")
    if "${" in config_text:
        parser.error("Materialize the configuration before running it.")
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict) or config.get("val_dataset", {}).get("mode") != "val":
        parser.error("This runner requires val_dataset.mode=val.")
    if config.get("model", {}).get("type") != recipe["model_type"]:
        parser.error("The configuration model type does not match the selected recipe.")
    expected = yaml.safe_load((ROOT / recipe["template"]).read_text(encoding="utf-8"))
    try:
        check_recipe(config, expected, set(recipe["path_only_changes"]), args.action != "train")
    except ValueError as exc:
        parser.error(str(exc))
    runtime = args.runtime.resolve()
    entry = (ROOT / "scripts" / "evaluate_exact.py" if args.action == "exact"
             else runtime / "tools" / ("train.py" if args.action == "train" else "val.py"))
    if not (runtime / "BOOTSTRAP_PROVENANCE.json").is_file() or not entry.is_file():
        parser.error("Bootstrap the pinned runtime first.")
    command = [args.python, "-B", str(entry), "--config", str(args.config.resolve()), "--device", args.device]
    if args.action == "train":
        if args.save_dir is None:
            parser.error("train requires --save-dir.")
        if config.get("train_dataset", {}).get("mode") != "train":
            parser.error("train_dataset.mode must be train.")
        command += ["--save_dir", str(args.save_dir.resolve()), "--do_eval",
                    "--save_interval", "1000", "--seed", str(args.seed),
                    "--log_iters", "100", "--num_workers", "0",
                    "--precision", "fp32"]
        if recipe["keep_checkpoint_max"] is not None:
            command += ["--keep_checkpoint_max", str(recipe["keep_checkpoint_max"])]
        if recipe["use_ema"]:
            command.append("--use_ema")
    else:
        if args.model_path is None:
            parser.error("val and exact require --model-path.")
        command += ["--model_path", str(args.model_path.resolve())]
        if args.action == "val":
            command += ["--num_workers", "0"]
        else:
            if args.output is None:
                parser.error("exact requires --output.")
            command += ["--output", str(args.output.resolve())]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(runtime)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    print(json.dumps({"execute": args.execute, "recipe": args.recipe,
                      "historical_backend": recipe["historical_backend"],
                      "argv": command, "PYTHONPATH": str(runtime)}, indent=2))
    if args.execute:
        if args.action == "exact":
            args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        return subprocess.call(command, env=environment, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

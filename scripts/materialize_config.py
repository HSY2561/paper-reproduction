"""Resolve portable path placeholders into a new local PaddleSeg YAML.

Only the configuration is read or written. No dataset or weight file is opened.
"""

import argparse
from pathlib import Path
import re

import yaml


PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


def resolve(value, substitutions):
    if isinstance(value, dict):
        return {key: resolve(item, substitutions) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, substitutions) for item in value]
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in substitutions:
                raise ValueError(f"Unknown placeholder: {name}")
            return substitutions[name]
        return PLACEHOLDER.sub(replace, value)
    return value


def clear_pretrained(value):
    """Disable initialization only when the caller will load a full checkpoint."""
    if isinstance(value, dict):
        return {key: None if key == "pretrained" else clear_pretrained(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [clear_pretrained(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--for-evaluation", action="store_true",
                        help="Set existing pretrained entries to null; a full checkpoint is required for evaluation.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.config.resolve() == args.output.resolve():
        parser.error("Input template and output must be different files.")
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists; choose another path or explicitly use --overwrite.")
    data = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("val_dataset", {}).get("mode") != "val":
        parser.error("A full configuration with val_dataset.mode=val is required.")
    if "_base_" in data:
        parser.error("Templates must be self-contained; _base_ is not accepted.")
    try:
        data = resolve(data, {
            "DATA_ROOT": args.data_root.resolve().as_posix(),
            "ARTIFACT_ROOT": args.artifact_root.resolve().as_posix(),
        })
    except ValueError as exc:
        parser.error(str(exc))
    if args.for_evaluation:
        data["model"] = clear_pretrained(data["model"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

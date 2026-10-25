"""Check syntax, pinned source hashes, and all portable recipes without Paddle or data."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    "overlays/paddleseg/models/segformer.py": "ad292bca4d0171028360ec00bbd1f7b87214c03a494ed2d13a02871e9c57d362",
    "scripts/evaluate_exact.py": "97ac482974536c1cd9be268b126313741eedbb6697b5f9af65adc873be7f53a6",
    "third_party/PaddleSeg-LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compiled = []
    for folder in ("scripts", "overlays"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            compile(path.read_bytes(), str(path.relative_to(ROOT)), "exec")
            compiled.append(path.relative_to(ROOT).as_posix())
    verified = []
    for relative, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        require(actual == expected, f"Source hash mismatch: {relative}")
        verified.append(relative)
    manifest = json.loads((ROOT / "configs" / "recipes.json").read_text(encoding="utf-8"))
    require(len(manifest["recipes"]) == 10, "Expected ten recipes.")
    families = {"A0": (128, ["aligned"], "base"), "B0": (256, [], "base"),
                "W96": (96, ["aligned"], "base"), "F08": (128, ["aligned"], "branch_bottleneck")}
    checked = []
    for name, recipe in manifest["recipes"].items():
        path = ROOT / recipe["template"]
        text = path.read_text(encoding="utf-8")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == recipe["template_sha256"],
                f"Template hash mismatch: {name}")
        require(not re.search(r"[A-Za-z]:[\\/]|/mnt/[a-z]/|/Users/|/home/", text),
                f"Nonportable absolute path in {name}")
        tokens = set(re.findall(r"\$\{([^}]+)\}", text))
        require(tokens <= {"DATA_ROOT", "ARTIFACT_ROOT"} and "DATA_ROOT" in tokens,
                f"Unexpected placeholder in {name}")
        data = yaml.safe_load(text)
        require(data["iters"] == 18000 and data["batch_size"] == 4, f"Budget differs: {name}")
        require(data["optimizer"] == {"type": "AdamW", "beta1": 0.9, "beta2": 0.999,
                                       "weight_decay": 0.01}, f"Optimizer differs: {name}")
        require(data["lr_scheduler"] == {"type": "PolynomialDecay", "learning_rate": 6e-5,
                                          "end_lr": 0.0, "power": 1}, f"LR differs: {name}")
        require(data["model"]["type"] == recipe["model_type"], f"Model type differs: {name}")
        require(data["train_dataset"]["mode"] == "train", f"Training mode differs: {name}")
        require(data["val_dataset"]["mode"] == "val", f"Validation mode differs: {name}")
        require(data["train_dataset"]["train_path"] == "${DATA_ROOT}/train.txt", f"Train path: {name}")
        require(data["val_dataset"]["val_path"] == "${DATA_ROOT}/val.txt", f"Val path: {name}")
        require("test_dataset" not in data, f"Unexpected test split: {name}")
        require(data["val_dataset"]["transforms"] == [{"type": "Normalize"}], f"Val transform: {name}")
        require(len(data["loss"]["types"]) == len(data["loss"]["coef"]), f"Loss arity: {name}")
        if name in families:
            model = data["model"]
            require((model["embedding_dim"], model["balance_modules"], model["fusion_variant"])
                    == families[name], f"A0-family architecture differs: {name}")
            require(model["backbone"]["pretrained"] == "${ARTIFACT_ROOT}/pretrained/mit_b0_imagenet.pdparams",
                    f"Common initialization differs: {name}")
        checked.append(name)
    result = {"passed": True, "compiled_without_import": compiled,
              "source_hashes_verified": verified, "recipes_checked": checked,
              "paddle_imported": False, "datasets_read": False, "training_started": False,
              "scope": "Static checks only; model forward and full training are separate validations."}
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()

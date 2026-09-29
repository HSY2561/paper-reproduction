import argparse
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import dump_json, ensure_dir, load_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ExperimentSpec:
    name: str
    family: str
    experiment_dir: Path
    config_path: Path
    model_path: Path


def repo_root():
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path_text, root=None):
    if not path_text:
        return None
    root = Path(root or repo_root())
    candidate = Path(str(path_text).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _metadata_candidates(experiment_dir):
    return [
        experiment_dir / "metrics.json",
        experiment_dir / "summary_row.json",
        experiment_dir / "model_stats.json",
    ]


def _read_metadata(experiment_dir):
    for path in _metadata_candidates(experiment_dir):
        if path.exists():
            yield load_json(path, default={})


def _resolve_config_path(experiment_dir, root):
    local_runtime = experiment_dir / "runtime_config.yml"
    if local_runtime.exists():
        return local_runtime

    for metadata in _read_metadata(experiment_dir):
        for key in ("config_path", "Config"):
            resolved = resolve_repo_path(metadata.get(key), root=root)
            if resolved and resolved.exists():
                return resolved

    for candidate in [
        root / "configs" / "torch_crack_benchmark" / "crack500_kaggle" / f"{experiment_dir.name}.yml",
        root / "configs" / "offline_experiments" / "crack500_kaggle" / f"{experiment_dir.name}.yml",
    ]:
        if candidate.exists():
            return candidate
    return None


def _latest_matching_file(experiment_dir, pattern):
    candidates = sorted(experiment_dir.glob(pattern))
    if candidates:
        return candidates[-1]
    return None


def _resolve_model_path(experiment_dir, root):
    for metadata in _read_metadata(experiment_dir):
        for key in ("best_model_path", "BestModelPath"):
            resolved = resolve_repo_path(metadata.get(key), root=root)
            if resolved and resolved.exists():
                return resolved

    for candidate in [
        experiment_dir / "best_model" / "model.pdparams",
        experiment_dir / "best_model.pt",
        _latest_matching_file(experiment_dir, "iter_*/model.pdparams"),
        _latest_matching_file(experiment_dir, "iter_*/model.pt"),
    ]:
        if candidate and candidate.exists():
            return candidate
    return None


def _infer_family(model_path):
    suffix = model_path.suffix.lower()
    if suffix == ".pdparams":
        return "paddle"
    if suffix == ".pt":
        return "torch"
    return None


def discover_experiments(result_root, repo_root=None):
    result_root = Path(result_root)
    root = Path(repo_root or globals()["repo_root"]())
    experiments = []
    if not result_root.exists():
        return experiments

    for experiment_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        config_path = _resolve_config_path(experiment_dir, root)
        model_path = _resolve_model_path(experiment_dir, root)
        if config_path is None or model_path is None:
            continue
        family = _infer_family(model_path)
        if family is None:
            continue
        experiments.append(
            ExperimentSpec(
                name=experiment_dir.name,
                family=family,
                experiment_dir=experiment_dir,
                config_path=config_path,
                model_path=model_path,
            )
        )
    return experiments


def save_binary_mask(mask, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    mask_array = np.asarray(mask)
    if mask_array.dtype != np.uint8:
        mask_array = mask_array.astype(bool).astype(np.uint8) * 255
    else:
        mask_array = (mask_array > 0).astype(np.uint8) * 255
    Image.fromarray(mask_array, mode="L").save(output_path)
    return output_path


def collect_image_paths(entries):
    image_paths = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            image_paths.extend(sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths.append(path)
    return image_paths


def _auto_device():
    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            return "gpu"
    except Exception:
        pass
    return "cpu"


class PaddlePredictor:
    def __init__(self, config_path, model_path, device):
        import paddle
        from paddleseg.core import infer
        from paddleseg.cvlibs import Config, SegBuilder
        from paddleseg.transforms import Compose
        from paddleseg.utils import utils

        utils.set_device(device)
        self.paddle = paddle
        self.infer = infer
        self.utils = utils
        self.cfg = Config(str(config_path))
        self.builder = SegBuilder(self.cfg)
        self.model = self.builder.model
        self.utils.load_entire_model(self.model, str(model_path))
        self.model.eval()
        self.transforms = Compose(self.builder.val_transforms)
        self.test_config = getattr(self.cfg, "test_config", {}) or {}

    def predict(self, image_path):
        data = {"img": str(image_path), "trans_info": []}
        data = self.transforms(data)
        tensor = self.paddle.to_tensor(data["img"][np.newaxis, ...])
        pred, _ = self.infer.inference(
            self.model,
            tensor,
            trans_info=data["trans_info"],
            is_slide=self.test_config.get("is_slide", False),
            stride=self.test_config.get("stride"),
            crop_size=self.test_config.get("crop_size"),
        )
        pred = np.squeeze(pred.numpy()).astype(np.uint8)
        return pred == 1


class TorchPredictor:
    def __init__(self, config_path, model_path, device):
        import torch

        from torch_crack_benchmark.config import load_config
        from torch_crack_benchmark.model import build_model, extract_logits

        self.torch = torch
        self.extract_logits = extract_logits
        self.cfg = load_config(str(config_path))
        self.device = torch.device("cuda" if device == "gpu" else "cpu")
        self.model = build_model(self.cfg["model_type"]).to(self.device)
        state = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"] if "model_state" in state else state)
        self.model.eval()
        self.mean = np.array(self.cfg["normalize"]["mean"], dtype="float32").reshape(1, 1, 3)
        self.std = np.array(self.cfg["normalize"]["std"], dtype="float32").reshape(1, 1, 3)

    def predict(self, image_path):
        with Image.open(image_path) as image_obj:
            image = image_obj.convert("RGB")
        image_np = np.asarray(image).astype("float32") / 255.0
        image_np = (image_np - self.mean) / self.std
        image_np = np.transpose(image_np, (2, 0, 1))[np.newaxis, ...]
        tensor = self.torch.from_numpy(image_np).to(self.device)
        with self.torch.no_grad():
            logits = self.extract_logits(self.model(tensor))
            probs = self.torch.sigmoid(logits)
            mask = (probs > 0.5).squeeze().detach().cpu().numpy().astype(bool)
        if mask.shape != image_np.shape[-2:]:
            mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(image.size, resample=Image.Resampling.NEAREST)) > 0
        return mask


def build_predictor(spec, device):
    if spec.family == "paddle":
        return PaddlePredictor(spec.config_path, spec.model_path, device=device)
    if spec.family == "torch":
        return TorchPredictor(spec.config_path, spec.model_path, device=device)
    raise ValueError(f"Unsupported experiment family: {spec.family}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict crack masks with all usable Crack500-Kaggle models.")
    parser.add_argument("--images", nargs="+", required=True, help="Image file(s) or directories to predict.")
    parser.add_argument(
        "--model_root",
        default="output/offline_experiments_torch/crack500_kaggle",
        help="Directory containing experiment result folders.",
    )
    parser.add_argument(
        "--output_root",
        default="output/offline_experiments_torch/crack500_kaggle_predictions",
        help="Directory to save binary masks.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--_venv_reentered", action="store_true")
    parser.add_argument("--_single_result_path", default=None)
    parser.add_argument("--_single_name", default=None)
    parser.add_argument("--_single_family", default=None)
    parser.add_argument("--_single_config_path", default=None)
    parser.add_argument("--_single_model_path", default=None)
    parser.add_argument("--_single_output_dir", default=None)
    return parser.parse_args()


def maybe_reexec_in_venv(args):
    if args._single_result_path:
        return
    if args._venv_reentered:
        return
    preferred_python = repo_root() / ".venv" / "Scripts" / "python.exe"
    current_python = Path(sys.executable).resolve()
    if not preferred_python.exists():
        return
    if current_python == preferred_python.resolve():
        return
    command = [str(preferred_python), str(Path(__file__).resolve()), *sys.argv[1:], "--_venv_reentered"]
    raise SystemExit(subprocess.call(command, cwd=str(repo_root())))


def run_single_experiment(args, device):
    image_paths = collect_image_paths(args.images)
    if not image_paths:
        raise FileNotFoundError("No input images found.")

    spec = ExperimentSpec(
        name=args._single_name,
        family=args._single_family,
        experiment_dir=Path(args.model_root) / args._single_name,
        config_path=Path(args._single_config_path),
        model_path=Path(args._single_model_path),
    )
    result = {
        "name": spec.name,
        "family": spec.family,
        "config_path": str(spec.config_path.as_posix()),
        "model_path": str(spec.model_path.as_posix()),
        "outputs": [],
    }
    predictor = build_predictor(spec, device=device)
    model_output_dir = ensure_dir(args._single_output_dir)
    for image_path in image_paths:
        mask = predictor.predict(image_path)
        output_path = Path(model_output_dir) / f"{image_path.stem}.png"
        save_binary_mask(mask, output_path)
        result["outputs"].append(str(output_path.as_posix()))
    result["status"] = "ok"
    dump_json(result, args._single_result_path)


def run_all_experiments(args, device):
    root = repo_root()
    image_paths = collect_image_paths(args.images)
    if not image_paths:
        raise FileNotFoundError("No input images found.")

    experiments = discover_experiments(args.model_root, repo_root=root)
    if not experiments:
        raise FileNotFoundError(f"No usable experiments found under {args.model_root}.")

    output_root = ensure_dir(args.output_root)
    manifest = {
        "device": device,
        "model_root": str(Path(args.model_root).as_posix()),
        "output_root": str(Path(output_root).as_posix()),
        "images": [str(path.as_posix()) for path in image_paths],
        "experiments": [],
    }

    for spec in experiments:
        result_path = Path(output_root) / f".{spec.name}_result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--images",
            *args.images,
            "--model_root",
            args.model_root,
            "--output_root",
            args.output_root,
            "--device",
            device,
            "--_single_result_path",
            str(result_path),
            "--_single_name",
            spec.name,
            "--_single_family",
            spec.family,
            "--_single_config_path",
            str(spec.config_path),
            "--_single_model_path",
            str(spec.model_path),
            "--_single_output_dir",
            str(Path(output_root) / spec.name),
        ]
        if args._venv_reentered:
            command.append("--_venv_reentered")

        completed = subprocess.run(
            command,
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
        )
        if result_path.exists():
            entry = load_json(result_path, default={})
            result_path.unlink(missing_ok=True)
        else:
            entry = {
                "name": spec.name,
                "family": spec.family,
                "config_path": str(spec.config_path.as_posix()),
                "model_path": str(spec.model_path.as_posix()),
                "outputs": [],
                "status": "error",
            }
        if completed.returncode != 0 and entry.get("status") != "ok":
            stderr = (completed.stderr or completed.stdout or "").strip()
            entry["error"] = stderr or f"subprocess failed with exit code {completed.returncode}"
        manifest["experiments"].append(entry)

    manifest_path = Path(output_root) / "manifest.json"
    dump_json(manifest, manifest_path)
    print(f"Saved manifest to {manifest_path}")


def main():
    args = parse_args()
    maybe_reexec_in_venv(args)
    device = _auto_device() if args.device == "auto" else args.device
    if args._single_result_path:
        run_single_experiment(args, device)
        return
    run_all_experiments(args, device)


if __name__ == "__main__":
    main()

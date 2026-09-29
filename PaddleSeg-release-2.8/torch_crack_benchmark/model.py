import importlib
import sys
from pathlib import Path

from torch_crack_benchmark.common import repo_root


def lmm_codes_root():
    return repo_root() / "Lightweight-Modular-Model-main" / "Codes"


def crackformer_root():
    return repo_root() / "_tmp" / "CrackFormer-II-main" / "CrackFormer-II-main" / "CrackFormer-II"


def resolve_model_spec(model_type):
    normalized = str(model_type).strip().lower()
    specs = {
        "crackformer_ii": {
            "name": "crackformer_ii",
            "package_root": crackformer_root(),
            "module_name": "nets.crackformerII",
            "factory_name": "crackformer",
            "kwargs": {},
        },
        "deepcrack": {
            "name": "deepcrack",
            "package_root": lmm_codes_root(),
            "module_name": "DeepCrack",
            "factory_name": "define_deepcrack",
            "kwargs": {
                "in_nc": 3,
                "num_classes": 1,
                "ngf": 64,
                "norm": "batch",
                "init_type": "xavier",
                "init_gain": 0.02,
            },
        },
        "efficientnet": {
            "name": "efficientnet",
            "package_root": lmm_codes_root(),
            "module_name": "Efficientnet",
            "factory_name": "EfficientNetSeg",
            "kwargs": {"efficientnet": ("symbol", "efficientnets", 0), "c1": 112, "c2": 320, "nclass": 1},
        },
        "mobilenetv3": {
            "name": "mobilenetv3",
            "package_root": lmm_codes_root(),
            "module_name": "Mobilenetv3",
            "factory_name": "MobileNetV3Seg",
            "kwargs": {"c1": 48, "c2": 576, "nclass": 1, "mode": "small"},
        },
        "shufflenetv2": {
            "name": "shufflenetv2",
            "package_root": lmm_codes_root(),
            "module_name": "Shufflenetv2",
            "factory_name": "ShuffleNetV2Seg",
            "kwargs": {"shufflenetv2": ("symbol", "shufflenets", 1), "c1": 232, "c2": 464, "nclass": 1},
        },
    }
    if normalized not in specs:
        raise ValueError(f"Unsupported benchmark model_type: {model_type}")
    return specs[normalized]


def _prepare_import_path(package_root):
    root_text = str(Path(package_root))
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def build_model(model_type):
    spec = resolve_model_spec(model_type)
    _prepare_import_path(spec["package_root"])
    module = importlib.import_module(spec["module_name"])
    factory = getattr(module, spec["factory_name"])
    kwargs = dict(spec["kwargs"])
    for key, value in list(kwargs.items()):
        if isinstance(value, tuple) and len(value) == 3 and value[0] == "symbol":
            kwargs[key] = getattr(module, value[1])[value[2]]
    return factory(**kwargs)


def extract_logits(model_output):
    if isinstance(model_output, (list, tuple)):
        return model_output[-1]
    return model_output

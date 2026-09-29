import sys
from pathlib import Path
import importlib


def codes_root():
    return Path(__file__).resolve().parents[1] / "Lightweight-Modular-Model-main" / "Codes"


def resolve_model_spec(model_type):
    normalized = str(model_type).strip().lower()
    specs = {
        "lmnet": {
            "module_name": "LM_Net",
            "factory_name": "LM_Net",
            "kwargs": {"img_ch": 3, "output_ch": 1},
        },
        "shufflenetv2": {
            "module_name": "Shufflenetv2",
            "factory_name": "ShuffleNetV2Seg",
            "kwargs": {"shufflenetv2": ("symbol", "shufflenets", 1), "c1": 232, "c2": 464, "nclass": 1},
        },
        "mobilenetv3": {
            "module_name": "Mobilenetv3",
            "factory_name": "MobileNetV3Seg",
            "kwargs": {"c1": 48, "c2": 576, "nclass": 1, "mode": "small"},
        },
    }
    if normalized not in specs:
        raise ValueError(f"Unsupported original-source model_type: {model_type}")
    return specs[normalized]


def build_model(model_type="lmnet"):
    code_dir = str(codes_root())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = resolve_model_spec(model_type)
    module = importlib.import_module(spec["module_name"])
    factory = getattr(module, spec["factory_name"])
    kwargs = dict(spec["kwargs"])
    for key, value in list(kwargs.items()):
        if isinstance(value, tuple) and len(value) == 3 and value[0] == "symbol":
            kwargs[key] = getattr(module, value[1])[value[2]]
    return factory(**kwargs)

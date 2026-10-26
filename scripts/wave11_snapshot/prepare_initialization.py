"""CPU-only, preregistered W11 compression and explicit paired initialization.

This script reads only the registered ImageNet checkpoint and local model source.
It never reads datasets, evaluates segmentation quality, or trains a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "PaddleSeg-2.10.0"
sys.path.insert(0, str(SOURCE))
import numpy as np
import paddle

from paddleseg.models import layers
from paddleseg.models.backbones.mix_transformer import Mlp
from paddleseg.models.backbones.mit_b0_s4ffn768 import (
    MixVisionTransformer_B0_S4FFN768, load_exact_state)
from paddleseg.models.segformer import MLP
from paddleseg.models.segformer_complementarity import (
    NarrowAlignedModule, SegFormerComplementarity)
from paddleseg.models.sfnet import AlignedModule

ORIGINAL_SHA256 = "211e15d5157b8d48b1931102c40094da0b578de1c153c5706360f7ad561a774e"
PROVENANCE = ROOT / "provenance"
WEIGHTS = ROOT / "weights"
INIT = WEIGHTS / "init"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def as_numpy(value):
    return np.ascontiguousarray(value.numpy() if hasattr(value, "numpy") else value)


def tensor_info(value):
    value = as_numpy(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "numel": int(value.size),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def state_info(state):
    return {key: tensor_info(state[key]) for key in sorted(state)}


def state_sha256(state):
    canonical = json.dumps(state_info(state), sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def arrays(state):
    return {key: as_numpy(value).copy() for key, value in state.items()}


def save_state(path, state):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    paddle.save({key: paddle.to_tensor(value) for key, value in state.items()}, str(path))
    actual = arrays(paddle.load(str(path)))
    assert_exact_state(state, actual, "saved checkpoint roundtrip")
    return {"path": str(path), "sha256": sha256_file(path),
            "tensor_sha256": state_sha256(state), "tensor_count": len(state)}


def assert_exact_state(expected, actual, context):
    if set(expected) != set(actual):
        raise AssertionError(f"{context}: key mismatch missing={set(expected)-set(actual)} "
                             f"extra={set(actual)-set(expected)}")
    for key in expected:
        a, b = as_numpy(expected[key]), as_numpy(actual[key])
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise AssertionError(f"{context}: tensor mismatch {key}")


def ffn_equivalence(original, derived, chosen):
    records = []
    with paddle.no_grad():
        for block in range(2):
            prefix = f"block4.{block}.mlp."
            source = {k[len(prefix):]: v for k, v in original.items()
                      if k.startswith(prefix)}
            trimmed = {k[len(prefix):]: v for k, v in derived.items()
                       if k.startswith(prefix)}
            # A full 1024-channel original FFN with discarded fc2 rows masked
            # to zero is exactly the original selected-hidden-channel function.
            masked = {k: v.copy() for k, v in source.items()}
            removed = np.setdiff1d(np.arange(1024), chosen[str(block)])
            masked["fc2.weight"][removed, :] = 0
            reference = Mlp(256, hidden_features=1024, out_features=256, drop=0.)
            compressed = Mlp(256, hidden_features=768, out_features=256, drop=0.)
            reference.set_state_dict({k: paddle.to_tensor(v) for k, v in masked.items()})
            compressed.set_state_dict({k: paddle.to_tensor(v) for k, v in trimmed.items()})
            reference.eval()
            compressed.eval()
            rng = np.random.default_rng(11000 + block)
            for batch, height, width in [(1, 13, 13), (2, 7, 9), (1, 1, 1)]:
                x = rng.normal(0., 0.2, (batch, height * width, 256)).astype("float32")
                x = paddle.to_tensor(x)
                expected = reference(x, height, width).numpy()
                actual = compressed(x, height, width).numpy()
                absolute = np.abs(expected - actual)
                relative = absolute / np.maximum(np.abs(expected), 1e-12)
                passed = bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-5))
                records.append({"block": block, "input_shape": list(x.shape),
                                "spatial_shape": [height, width], "device": "cpu",
                                "eval": True, "dropout": 0.,
                                "mask": "zero_discarded_fc2_rows_full_original_ffn",
                                "max_abs_error": float(absolute.max()),
                                "max_relative_error": float(relative.max()),
                                "atol": 1e-5, "rtol": 1e-5, "passed": passed})
                if not passed:
                    raise AssertionError(f"Masked FFN equivalence failed: {records[-1]}")
    return records


def prepare_compression():
    original_path = WEIGHTS / "mit_b0_imagenet.pdparams"
    original_sha = sha256_file(original_path)
    if original_sha != ORIGINAL_SHA256:
        raise ValueError(f"Original ImageNet SHA mismatch: {original_sha}")
    original = arrays(paddle.load(str(original_path)))
    derived = {key: value.copy() for key, value in original.items()}
    chosen = {}
    score_records = {}
    transforms = {}
    for block in range(2):
        prefix = f"block4.{block}.mlp."
        fc1 = original[prefix + "fc1.weight"].astype("float64")
        fc1_bias = original[prefix + "fc1.bias"].astype("float64")
        dw = original[prefix + "dwconv.dwconv.weight"].astype("float64")
        dw_bias = original[prefix + "dwconv.dwconv.bias"].astype("float64")
        fc2 = original[prefix + "fc2.weight"].astype("float64")
        if fc1.shape != (256, 1024) or dw.shape != (1024, 1, 3, 3) or fc2.shape != (1024, 256):
            raise ValueError("Unexpected Paddle [in,out] / DW source shapes")
        a = np.abs(fc1).mean(axis=0) + np.abs(fc1_bias)
        d = np.abs(dw).mean(axis=(1, 2, 3)) + np.abs(dw_bias)
        o = np.abs(fc2).mean(axis=1)
        score = (a / (a.mean() + 1e-12) + d / (d.mean() + 1e-12)
                 + o / (o.mean() + 1e-12)) / 3.
        order = np.lexsort((np.arange(1024, dtype=np.int64), -score))
        indices = np.sort(order[:768]).astype(np.int64)
        chosen[str(block)] = indices.tolist()
        rank = np.empty(1024, dtype=np.int64)
        rank[order] = np.arange(1, 1025)
        score_records[str(block)] = {
            "a": a.tolist(), "d": d.tolist(), "o": o.tolist(),
            "score": score.tolist(), "descending_score_rank": rank.tolist(),
            "calculation_dtype": "float64", "selection_count": 768,
            "tie_break": "original_channel_index_ascending",
            "stored_order": "original_channel_index_ascending"}
        csv_path = PROVENANCE / f"s4_block{block}_channel_scores.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["original_index", "a", "d", "o", "score", "rank", "selected"])
            selected = set(indices.tolist())
            for index in range(1024):
                writer.writerow([index, format(a[index], ".17g"), format(d[index], ".17g"),
                                 format(o[index], ".17g"), format(score[index], ".17g"),
                                 int(rank[index]), int(index in selected)])
        for suffix, axis in [("fc1.weight", 1), ("fc1.bias", 0),
                             ("dwconv.dwconv.weight", 0), ("dwconv.dwconv.bias", 0),
                             ("fc2.weight", 0)]:
            key = prefix + suffix
            derived[key] = np.take(original[key], indices, axis=axis)
            transforms[key] = {"operation": "take", "axis": axis,
                               "indices_block": block, "selected_count": 768}
    indices_path = WEIGHTS / "s4ffn768_indices.json"
    dump_json(indices_path, {"version": "W11-v1", "original_sha256": original_sha,
                             "blocks": chosen})
    scores_path = PROVENANCE / "s4ffn768_scores_float64.json"
    dump_json(scores_path, {"version": "W11-v1", "blocks": score_records})
    for key in original:
        if key not in transforms and not np.array_equal(original[key], derived[key]):
            raise AssertionError("Unmodified tensor changed: " + key)
    equivalence = ffn_equivalence(original, derived, chosen)
    full_source = save_state(WEIGHTS / "mit_b0_s4ffn768_imagenet_full_source.pdparams", derived)
    backbone = MixVisionTransformer_B0_S4FFN768(pretrained=None)
    expected = backbone.state_dict()
    excluded = sorted(set(derived) - set(expected))
    if excluded != ["head.bias", "head.weight"]:
        raise AssertionError("Unregistered original exclusion: " + repr(excluded))
    if set(expected) - set(derived):
        raise AssertionError("Source misses backbone keys")
    strict = {key: derived[key] for key in expected}
    for key, value in expected.items():
        if list(value.shape) != list(strict[key].shape):
            raise AssertionError("Compressed shape mismatch: " + key)
    strict_file = save_state(WEIGHTS / "mit_b0_s4ffn768_imagenet.pdparams", strict)
    load_exact_state(backbone, strict_file["path"])
    assert_exact_state(strict, backbone.state_dict(), "strict backbone constructor")
    ledger = {}
    for key in original:
        ledger[key] = {"original": tensor_info(original[key]),
                       "derived": tensor_info(derived[key]),
                       "transform": transforms.get(key, {"operation": "bitwise_copy"}),
                       "included_in_strict_backbone": key in strict}
    ledger_path = PROVENANCE / "compression_tensor_ledger.json"
    dump_json(ledger_path, ledger)
    audit = {"status": "passed", "device": paddle.get_device(),
             "original": {"path": str(original_path), "sha256": original_sha,
                          "tensor_count": len(original)},
             "derived_backbone": strict_file, "derived_full_source": full_source,
             "indices": {"path": str(indices_path), "sha256": sha256_file(indices_path)},
             "scores": {"path": str(scores_path), "sha256": sha256_file(scores_path)},
             "tensor_ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
             "changed_tensor_count": len(transforms),
             "unchanged_tensor_count": len(original) - len(transforms),
             "all_unchanged_tensors_bitwise_identical": True,
             "explicit_model_exclusions": {
                 key: {"reason": "ImageNet classification-only head absent in segmentation backbone",
                       "source_tensor": tensor_info(original[key]),
                       "preserved_in_full_source_checkpoint": True} for key in excluded},
             "masked_original_ffn_equivalence": equivalence}
    dump_json(PROVENANCE / "compression_audit.json", audit)
    return audit, strict


def component_seed(seed, name):
    material = f"W11-v1|initialization|{int(seed)}|{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little") % 2147483647


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def component_factories():
    nn = paddle.nn
    return {
        "linear_c1": lambda: MLP(32, 128),
        "linear_c2": lambda: MLP(64, 128),
        "linear_c3": lambda: MLP(160, 128),
        "linear_c4": lambda: MLP(256, 128),
        "linear_fuse": lambda: layers.ConvBNReLU(512, 128, 1, bias_attr=False),
        "balance_blocks.aligned": lambda: AlignedModule(128, 64),
        "linear_pred": lambda: nn.Conv2D(128, 2, 1, bias_attr=True),
        "m1_c1": lambda: layers.ConvBNReLU(32, 128, 1, bias_attr=False),
        "m1_c2": lambda: layers.ConvBNReLU(64, 32, 1, bias_attr=False),
        "m1_c3": lambda: layers.ConvBNReLU(160, 32, 1, bias_attr=False),
        "m1_c4": lambda: layers.ConvBNReLU(256, 32, 1, bias_attr=False),
        "m1_deep_fuse": lambda: layers.ConvBNReLU(96, 64, 1, bias_attr=False),
        "m1_aligned": lambda: NarrowAlignedModule(128, 64, 32),
        "m1_out_fuse": lambda: layers.ConvBNReLU(192, 128, 1, bias_attr=False),
        "detail_restore.detail": lambda: layers.ConvBNReLU(32, 16, 1, bias_attr=False),
        "detail_restore.semantic": lambda: layers.ConvBNReLU(128, 16, 1, bias_attr=False),
        "detail_restore.gate": lambda: nn.Conv2D(32, 1, 1, bias_attr=True),
        "detail_restore.depthwise": lambda: layers.ConvBNReLU(16, 16, 3, groups=16,
                                                             bias_attr=False),
        "detail_restore.output": lambda: layers.ConvBN(16, 128, 1, bias_attr=False),
    }


def build_components(seed, reverse=False):
    factories = component_factories()
    names = list(factories)
    if reverse:
        names.reverse()
    state, seeds = {}, {}
    for name in names:
        current_seed = component_seed(seed, name)
        seed_all(current_seed)
        component = factories[name]()
        state.update({name + "." + key: value for key, value in arrays(component.state_dict()).items()})
        seeds[name] = current_seed
    state["detail_restore.alpha"] = np.array([0.1], dtype="float32")
    seeds["detail_restore.alpha"] = {"initializer": "Constant", "value": 0.1}
    return state, seeds


def subset(state, name):
    if name == "backbone":
        predicate = lambda key: key.startswith("backbone.")
    elif name == "classifier":
        predicate = lambda key: key.startswith("linear_pred.")
    elif name == "old_head":
        predicate = lambda key: key.startswith(("linear_c", "linear_fuse.", "balance_blocks."))
    elif name == "m1_head":
        predicate = lambda key: key.startswith("m1_")
    elif name == "m2":
        predicate = lambda key: key.startswith("detail_restore.")
    elif name == "bn_running_state":
        predicate = lambda key: key.endswith(("._mean", "._variance"))
    else:
        raise ValueError(name)
    return {key: value for key, value in state.items() if predicate(key)}


def make_model(group, pretrained=None):
    return SegFormerComplementarity(
        num_classes=2, backbone=MixVisionTransformer_B0_S4FFN768(pretrained=None),
        embedding_dim=128, align_corners=False, pretrained=pretrained,
        use_m1=group in (1, 3), use_m2=group in (2, 3))


def prepare_models(compression, backbone_state):
    runs, shared_checks, construction_checks, seed_ledger = {}, [], [], {}
    init_tensor_dir = PROVENANCE / "initialization_tensors"
    for seed in (42, 43, 44):
        components, seeds = build_components(seed)
        reversed_components, _ = build_components(seed, reverse=True)
        assert_exact_state(components, reversed_components,
                           f"seed {seed} component construction order independence")
        construction_checks.append({"seed": seed, "forward_reverse_order_bitwise_equal": True,
                                    "tensor_count": len(components),
                                    "tensor_sha256": state_sha256(components)})
        seed_ledger[str(seed)] = seeds
        seed_states = {}
        for group in range(4):
            # Deliberately perturb constructor RNG differently in every cell;
            # every registered value is overwritten by the named components.
            seed_all(component_seed(seed, f"discarded_constructor_E{group}"))
            model = make_model(group)
            expected = model.state_dict()
            state = {}
            for key, value in expected.items():
                if key.startswith("backbone."):
                    item = backbone_state[key[len("backbone."):]]
                else:
                    if key not in components:
                        raise AssertionError("No explicitly initialized model key: " + key)
                    item = components[key]
                if list(value.shape) != list(item.shape):
                    raise AssertionError("Initialization shape mismatch: " + key)
                state[key] = item.copy()
            model.set_state_dict({key: paddle.to_tensor(value) for key, value in state.items()})
            assert_exact_state(state, model.state_dict(), "explicit full-state assignment")
            name = f"E{group}_s{seed}"
            record = save_state(INIT / f"{name}.pdparams", state)
            # Exercise exactly the strict model.pretrained path used by runtime.
            restored = make_model(group, pretrained=record["path"])
            assert_exact_state(state, restored.state_dict(), "complete model pretrained strict load")
            tensor_path = init_tensor_dir / f"{name}.json"
            dump_json(tensor_path, state_info(state))
            component_hashes = {}
            for component in ("backbone", "classifier", "old_head", "m1_head", "m2",
                              "bn_running_state"):
                values = subset(state, component)
                if values:
                    component_hashes[component] = {"tensor_sha256": state_sha256(values),
                                                   "tensor_count": len(values),
                                                   "numel": sum(int(v.size) for v in values.values())}
            bn = subset(state, "bn_running_state")
            for key, value in bn.items():
                expected_value = 0 if key.endswith("._mean") else 1
                if not np.all(value == expected_value):
                    raise AssertionError("Unexpected initial BN running state: " + key)
            if group in (2, 3) and not np.array_equal(state["detail_restore.alpha"],
                                                      np.array([0.1], dtype="float32")):
                raise AssertionError("M2 alpha is not exactly float32 .1")
            record.update({"seed": seed, "group": f"E{group}",
                           "tensor_manifest_path": str(tensor_path),
                           "tensor_manifest_sha256": sha256_file(tensor_path),
                           "component_sha256": component_hashes,
                           "registered_numel": int(sum(np.prod(v.shape) for v in model.parameters())),
                           "trainable_numel": int(sum(np.prod(v.shape) for v in model.parameters()
                                                       if not v.stop_gradient)),
                           "all_keys_explicitly_assigned": True,
                           "strict_pretrained_roundtrip_bitwise_equal": True,
                           "bn_running_state_initial_mean0_variance1": True})
            runs[name] = record
            seed_states[group] = state
            print(f"prepared {name}: {record['sha256']}", flush=True)
        for component, groups in [("backbone", [0, 1, 2, 3]),
                                  ("classifier", [0, 1, 2, 3]),
                                  ("old_head", [0, 2]), ("m1_head", [1, 3]),
                                  ("m2", [2, 3])]:
            reference = subset(seed_states[groups[0]], component)
            if not reference:
                raise AssertionError("Empty shared component " + component)
            for group in groups[1:]:
                assert_exact_state(reference, subset(seed_states[group], component),
                                   f"seed {seed} shared {component}")
            shared_checks.append({"seed": seed, "component": component,
                                  "groups": [f"E{group}" for group in groups],
                                  "tensor_count": len(reference),
                                  "tensor_sha256": state_sha256(reference),
                                  "bitwise_identical": True})
    manifest = {
        "version": "W11-v1", "status": "passed", "created_utc": utc_now(),
        "python": sys.executable, "paddle_version": paddle.__version__, "device": paddle.get_device(),
        "source_root": str(SOURCE), "script_sha256": sha256_file(__file__),
        "original_checkpoint_sha256": ORIGINAL_SHA256,
        "derived_backbone": compression["derived_backbone"],
        "indices": compression["indices"], "compression_audit_path": str(PROVENANCE / "compression_audit.json"),
        "runs": runs, "shared_checks": shared_checks,
        "named_component_seeds": seed_ledger,
        "component_seed_mapping": "uint32_little_endian(SHA256('W11-v1|initialization|SEED|COMPONENT')[:4]) % 2147483647",
        "component_initializers": "Paddle native constructors, each immediately preceded by its named seed; alpha constant .1",
        "constructor_rng_control": "full registered state explicitly overwritten; constructor order not used to share values",
        "component_construction_order_checks": construction_checks,
        "data_access": "none", "training": False, "gpu_execution": False,
        "runtime_requirements": {
            "model_type": "SegFormerComplementarity", "model_pretrained": "runs[NAME].path",
            "backbone_type": "MixVisionTransformer_B0_S4FFN768", "backbone_pretrained": None,
            "reset_python_numpy_paddle_data_seed_after_model_load_before_train": True,
            "training_data_rng_audit_owner": "w11_runtime"},
    }
    dump_json(PROVENANCE / "initialization_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress-only", action="store_true")
    args = parser.parse_args()
    for directory in (PROVENANCE, WEIGHTS, INIT):
        directory.mkdir(parents=True, exist_ok=True)
    paddle.set_device("cpu")
    if paddle.get_device() != "cpu":
        raise RuntimeError("Initialization must run on CPU")
    started = utc_now()
    compression, backbone_state = prepare_compression()
    manifest = None if args.compress_only else prepare_models(compression, backbone_state)
    result = {"status": "passed", "started_utc": started, "finished_utc": utc_now(),
              "device": paddle.get_device(), "compress_only": args.compress_only,
              "run_count": 0 if manifest is None else len(manifest["runs"]),
              "compression_audit": str(PROVENANCE / "compression_audit.json"),
              "initialization_manifest": None if manifest is None else str(PROVENANCE / "initialization_manifest.json")}
    dump_json(PROVENANCE / "initialization_result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        PROVENANCE.mkdir(parents=True, exist_ok=True)
        dump_json(PROVENANCE / "initialization_failure.json",
                  {"time_utc": utc_now(), "traceback": traceback.format_exc(),
                   "device": paddle.get_device()})
        raise

"""Generate new R0 controls with E0-matched heads and original ImageNet B0.

The three R0 runs are new controls. Historical trained A0 checkpoints are never
read, modified, or represented as matched initialization by this script.
"""
from pathlib import Path
import json
import sys
import traceback

import paddle

from prepare_initialization import (
    ROOT, ORIGINAL_SHA256, PROVENANCE, WEIGHTS, arrays, assert_exact_state,
    dump_json, save_state, sha256_file, state_info, state_sha256, utc_now)
from paddleseg.models.backbones.mix_transformer import MixVisionTransformer_B0
from paddleseg.models.segformer_complementarity import SegFormerComplementarity


def model(pretrained=None):
    return SegFormerComplementarity(
        num_classes=2, backbone=MixVisionTransformer_B0(pretrained=None),
        embedding_dim=128, align_corners=False, pretrained=pretrained,
        use_m1=False, use_m2=False)


def main():
    paddle.set_device("cpu")
    original_path = WEIGHTS / "mit_b0_imagenet.pdparams"
    if sha256_file(original_path) != ORIGINAL_SHA256:
        raise ValueError("Original ImageNet SHA mismatch")
    original = arrays(paddle.load(str(original_path)))
    source_manifest_path = PROVENANCE / "initialization_manifest.json"
    source_manifest_sha = sha256_file(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["status"] != "passed":
        raise ValueError("E0 initialization audit must have passed")
    records = {}
    for seed in (42, 43, 44):
        e0_path = WEIGHTS / "init" / f"E0_s{seed}.pdparams"
        e0_record = source_manifest["runs"][f"E0_s{seed}"]
        if sha256_file(e0_path) != e0_record["sha256"]:
            raise ValueError("E0 source checkpoint SHA mismatch")
        e0 = arrays(paddle.load(str(e0_path)))
        target_model = model()
        expected = target_model.state_dict()
        state = {}
        for key, value in expected.items():
            source = original[key[len("backbone."):]] if key.startswith("backbone.") else e0[key]
            if list(value.shape) != list(source.shape):
                raise ValueError("R0 state shape mismatch: " + key)
            state[key] = source.copy()
        if set(state) != set(e0):
            raise ValueError("R0 and E0 full model keys must match exactly")
        full_backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        excluded = sorted(set(original) - set(full_backbone))
        if excluded != ["head.bias", "head.weight"]:
            raise ValueError("Unexpected original source exclusion: " + repr(excluded))
        assert_exact_state({k: original[k] for k in full_backbone}, full_backbone,
                           "R0 backbone equals original ImageNet backbone")
        head = {k: v for k, v in state.items() if not k.startswith("backbone.")}
        e0_head = {k: v for k, v in e0.items() if not k.startswith("backbone.")}
        assert_exact_state(e0_head, head, "R0 head bitwise matches same-seed E0")
        target_model.set_state_dict({k: paddle.to_tensor(v) for k, v in state.items()})
        assert_exact_state(state, target_model.state_dict(), "R0 explicit full assignment")
        counts = {"registered_numel": sum(int(v.numel()) for v in target_model.parameters()),
                  "trainable_numel": sum(int(v.numel()) for v in target_model.parameters()
                                          if not v.stop_gradient)}
        if counts != {"registered_numel": 3470434, "trainable_numel": 3470178}:
            raise ValueError("Unexpected complete R0 parameter count: " + repr(counts))
        name = f"R0_s{seed}"
        record = save_state(WEIGHTS / "init" / (name + ".pdparams"), state)
        restored = model(pretrained=record["path"])
        assert_exact_state(state, restored.state_dict(), "R0 strict complete pretrained roundtrip")
        tensor_path = PROVENANCE / "matched_R0_tensors" / (name + ".json")
        dump_json(tensor_path, state_info(state))
        record.update(seed=seed, group="R0", paired_E0=f"E0_s{seed}",
                      E0_initialization_path=str(e0_path), E0_initialization_sha256=sha256_file(e0_path),
                      non_backbone_tensor_count=len(head),
                      non_backbone_tensor_sha256=state_sha256(head),
                      non_backbone_bitwise_matches_E0=True,
                      backbone_tensor_count=len(full_backbone),
                      backbone_tensor_sha256=state_sha256(full_backbone),
                      backbone_bitwise_matches_original_ImageNet=True,
                      strict_pretrained_roundtrip_bitwise_equal=True,
                      tensor_manifest_path=str(tensor_path),
                      tensor_manifest_sha256=sha256_file(tensor_path), **counts)
        records[name] = record
        print(f"prepared {name}: {record['sha256']}", flush=True)
    if sha256_file(source_manifest_path) != source_manifest_sha:
        raise ValueError("Original twelve-run initialization manifest changed")
    result = dict(status="passed", created_utc=utc_now(), device=paddle.get_device(),
                  python=sys.executable, paddle_version=paddle.__version__,
                  script_sha256=sha256_file(__file__),
                  original_ImageNet_path=str(original_path), original_ImageNet_sha256=ORIGINAL_SHA256,
                  original_classification_head_excluded=["head.bias", "head.weight"],
                  E0_initialization_manifest_path=str(source_manifest_path),
                  E0_initialization_manifest_sha256=source_manifest_sha,
                  original_twelve_run_manifest_unchanged=True,
                  runs=records, training=False, gpu_execution=False, data_access="none",
                  rationale="New R0 uses original uncompressed ImageNet B0 and exactly the same complete head as paired E0; historical A0 is only a reproduction reference",
                  runtime=dict(model_type="SegFormerComplementarity", use_m1=False, use_m2=False,
                               backbone_type="MixVisionTransformer_B0", backbone_pretrained=None,
                               model_pretrained="runs[NAME].path",
                               reset_python_numpy_paddle_seed_after_model_load_before_data=True))
    dump_json(PROVENANCE / "matched_R0_initialization.json", result)
    print(json.dumps({"status": "passed", "run_count": len(records),
                      "manifest": str(PROVENANCE / "matched_R0_initialization.json")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        dump_json(PROVENANCE / "matched_R0_initialization_failure.json",
                  {"time_utc": utc_now(), "traceback": traceback.format_exc()})
        raise

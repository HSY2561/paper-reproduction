import argparse
from copy import deepcopy
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import dump_yaml, load_yaml


def _num_classes_from_config(config):
    return config.get("train_dataset", {}).get("num_classes", 2)


def _apply_dataset_paths(config, dataset_root):
    dataset_root = Path(dataset_root).as_posix()
    train_dataset = deepcopy(config["train_dataset"])
    val_dataset = deepcopy(config["val_dataset"])

    train_dataset["dataset_root"] = dataset_root
    train_dataset["train_path"] = f"{dataset_root}/train.txt"
    val_dataset["dataset_root"] = dataset_root
    val_dataset["val_path"] = f"{dataset_root}/val.txt"

    config["train_dataset"] = train_dataset
    config["val_dataset"] = val_dataset
    return config


def _build_common_hrseg_template(baseline_config, final_config):
    template = {}
    template["batch_size"] = deepcopy(final_config.get("batch_size", baseline_config.get("batch_size")))
    template["iters"] = deepcopy(final_config.get("iters", baseline_config.get("iters")))
    template["optimizer"] = deepcopy(final_config.get("optimizer", baseline_config.get("optimizer")))
    template["loss"] = deepcopy(final_config.get("loss", baseline_config.get("loss")))
    template["lr_scheduler"] = deepcopy(final_config.get("lr_scheduler", baseline_config.get("lr_scheduler")))
    template["train_dataset"] = deepcopy(baseline_config.get("train_dataset"))
    template["val_dataset"] = deepcopy(baseline_config.get("val_dataset"))
    if "test_config" in baseline_config or "test_config" in final_config:
        template["test_config"] = deepcopy(final_config.get("test_config", baseline_config.get("test_config")))
    return template


def _num_loss_heads(loss_cfg):
    if not loss_cfg:
        return 1
    coef = loss_cfg.get("coef")
    if isinstance(coef, list) and coef:
        return len(coef)
    types = loss_cfg.get("types")
    if isinstance(types, list) and types:
        return len(types)
    return 1


def _build_uniform_loss(common_loss, num_heads):
    common_loss = deepcopy(common_loss or {})
    base_type = {"type": "OhemCrossEntropyLoss"}
    common_types = common_loss.get("types")
    if isinstance(common_types, list) and common_types:
        base_type = deepcopy(common_types[0])

    types = [deepcopy(base_type) for _ in range(num_heads)]
    coef = [1.0] + [0.5] * max(0, num_heads - 1)
    return {"types": types, "coef": coef}


def _apply_common_training_recipe(config, common_template, dataset_root, save_dir, force_uniform_loss=False):
    config = deepcopy(config)
    config["batch_size"] = deepcopy(common_template.get("batch_size"))
    config["iters"] = deepcopy(common_template.get("iters"))
    config["optimizer"] = deepcopy(common_template.get("optimizer"))
    config["lr_scheduler"] = deepcopy(common_template.get("lr_scheduler"))
    config["train_dataset"] = deepcopy(common_template.get("train_dataset"))
    config["val_dataset"] = deepcopy(common_template.get("val_dataset"))
    config = _apply_dataset_paths(config, dataset_root)
    config["save_dir"] = save_dir

    if force_uniform_loss:
        num_heads = _num_loss_heads(config.get("loss"))
        config["loss"] = _build_uniform_loss(common_template.get("loss"), num_heads)
    return config


def _build_single_base_variants(base, final_model, num_classes, baseline_type=None, final_type="HrSegNetB16A"):
    baseline_type = baseline_type or f"HrSegNetB{base}"
    aspp_rates = final_model.get("aspp_rates", [1, 6, 12, 18])
    aspp_out_channels = final_model.get("aspp_out_channels", 64)
    decoder_channels = final_model.get("decoder_channels", 64)
    low_level_channels = final_model.get("low_level_channels", 48)

    return {
        f"b{base}_baseline": {
            "type": baseline_type,
            "base": base,
            "num_classes": num_classes,
            "pretrained": None,
        },
        f"b{base}_aspp": {
            "type": "HrSegNetB16ASPP",
            "base": base,
            "num_classes": num_classes,
            "aspp_rates": aspp_rates,
            "aspp_out_channels": aspp_out_channels,
            "pretrained": None,
        },
        f"b{base}_decoder": {
            "type": "HrSegNetB16Decoder",
            "base": base,
            "num_classes": num_classes,
            "decoder_channels": decoder_channels,
            "low_level_channels": low_level_channels,
            "pretrained": None,
        },
        f"b{base}_final": {
            "type": final_type,
            "base": base,
            "num_classes": num_classes,
            "aspp_rates": aspp_rates,
            "aspp_out_channels": aspp_out_channels,
            "decoder_channels": decoder_channels,
            "low_level_channels": low_level_channels,
            "pretrained": None,
        },
    }


def _build_hrseg_model_variants(final_model, num_classes, b32_final_model=None, b64_final_model=None):
    variants = {}
    variants.update(_build_single_base_variants(16, final_model, num_classes))
    variants.update(
        _build_single_base_variants(
            32,
            b32_final_model or final_model,
            num_classes,
            baseline_type="HrSegNetB32",
            final_type="HrSegNetB16A",
        )
    )
    variants.update(
        _build_single_base_variants(
            64,
            b64_final_model or final_model,
            num_classes,
            baseline_type="HrSegNetB64",
            final_type="HrSegNetB16A",
        )
    )
    return variants


def _save_dir(dataset_name, experiment_name):
    return f"./output/offline_experiments/{dataset_name}/{experiment_name}"


def _write_ablation_configs(common_template, model_variants, dataset_root, output_root):
    generated_paths = []
    ablation_dir = Path(output_root) / "ablation"
    for experiment_name, model_config in model_variants.items():
        config = deepcopy(common_template)
        config = _apply_dataset_paths(config, dataset_root)
        config["model"] = deepcopy(model_config)
        config["save_dir"] = _save_dir("ablation", experiment_name)
        output_path = ablation_dir / f"{experiment_name}.yml"
        dump_yaml(config, output_path)
        generated_paths.append(output_path)
    return generated_paths


def _retarget_generic_config(config, dataset_root, save_dir):
    config = deepcopy(config)
    config = _apply_dataset_paths(config, dataset_root)
    config["save_dir"] = save_dir
    return config


def _write_crack500_configs(
    common_template,
    model_variants,
    unet_config,
    deeplab_config,
    pspnet_config,
    lmm_config,
    segformer_config,
    dataset_root,
    output_root,
):
    generated_paths = []
    crack500_dir = Path(output_root) / "crack500"
    mapping = {
        "hrsegnet_b16.yml": model_variants["b16_baseline"],
        "hrsegnet_b16_a.yml": model_variants["b16_final"],
        "hrsegnet_b32.yml": model_variants["b32_baseline"],
        "hrsegnet_b32_a.yml": model_variants["b32_final"],
        "hrsegnet_b64.yml": model_variants["b64_baseline"],
        "hrsegnet_b64_a.yml": model_variants["b64_final"],
    }

    for aspp_out_channels in (32, 64, 96, 128):
        model_config = deepcopy(model_variants["b16_final"])
        model_config["aspp_out_channels"] = aspp_out_channels
        mapping[f"hrsegnet_b16_a_aspp{aspp_out_channels}.yml"] = model_config

    for filename, model_config in mapping.items():
        config = deepcopy(common_template)
        config = _apply_dataset_paths(config, dataset_root)
        config["model"] = deepcopy(model_config)
        config["save_dir"] = _save_dir("crack500", Path(filename).stem)
        output_path = crack500_dir / filename
        dump_yaml(config, output_path)
        generated_paths.append(output_path)

    unet_result = _apply_common_training_recipe(
        unet_config,
        common_template,
        dataset_root,
        _save_dir("crack500", "unet"),
        force_uniform_loss=True,
    )
    dump_yaml(unet_result, crack500_dir / "unet.yml")
    generated_paths.append(crack500_dir / "unet.yml")

    deeplab_result = _apply_common_training_recipe(
        deeplab_config,
        common_template,
        dataset_root,
        _save_dir("crack500", "deeplabv3p"),
        force_uniform_loss=True,
    )
    dump_yaml(deeplab_result, crack500_dir / "deeplabv3p.yml")
    generated_paths.append(crack500_dir / "deeplabv3p.yml")

    pspnet_result = _apply_common_training_recipe(
        pspnet_config,
        common_template,
        dataset_root,
        _save_dir("crack500", "pspnet"),
        force_uniform_loss=True,
    )
    dump_yaml(pspnet_result, crack500_dir / "pspnet.yml")
    generated_paths.append(crack500_dir / "pspnet.yml")

    lmm_result = _apply_common_training_recipe(
        lmm_config,
        common_template,
        dataset_root,
        _save_dir("crack500", "lmnet"),
        force_uniform_loss=True,
    )
    dump_yaml(lmm_result, crack500_dir / "lmnet.yml")
    generated_paths.append(crack500_dir / "lmnet.yml")

    segformer_result = _apply_common_training_recipe(
        segformer_config,
        common_template,
        dataset_root,
        _save_dir("crack500", "segformer_b0"),
        force_uniform_loss=True,
    )
    dump_yaml(segformer_result, crack500_dir / "segformer_b0.yml")
    generated_paths.append(crack500_dir / "segformer_b0.yml")
    return generated_paths


def generate_all_configs(
    baseline_config_path,
    final_config_path,
    unet_config_path,
    deeplab_config_path,
    pspnet_config_path,
    lmm_config_path,
    segformer_config_path="configs/aa/segformer_b0_720.yml",
    b32_baseline_config_path="configs/aa/hrsegnetb32.yml",
    b32_final_config_path="configs/aa/hrsegnetb32_asppdelite.yml",
    b64_baseline_config_path="configs/aa/hrsegnetb64.yml",
    b64_final_config_path="configs/aa/hrsegnetb64_asppde.yml",
    custom_root="dataset/custom",
    crack500_root="dataset/Crack500",
    output_root="configs/offline_experiments",
):
    baseline_config = load_yaml(baseline_config_path)
    final_config = load_yaml(final_config_path)
    b32_baseline_config = load_yaml(b32_baseline_config_path)
    b32_final_config = load_yaml(b32_final_config_path)
    b64_baseline_config = load_yaml(b64_baseline_config_path)
    b64_final_config = load_yaml(b64_final_config_path)
    unet_config = load_yaml(unet_config_path)
    deeplab_config = load_yaml(deeplab_config_path)
    pspnet_config = load_yaml(pspnet_config_path)
    lmm_config = load_yaml(lmm_config_path)
    segformer_config = load_yaml(segformer_config_path)

    common_template = _build_common_hrseg_template(baseline_config, final_config)
    model_variants = _build_hrseg_model_variants(
        final_config.get("model", {}),
        _num_classes_from_config(baseline_config),
        b32_final_model=b32_final_config.get("model", {}),
        b64_final_model=b64_final_config.get("model", {}),
    )

    generated_paths = []
    generated_paths.extend(
        _write_ablation_configs(common_template, model_variants, custom_root, output_root)
    )
    generated_paths.extend(
        _write_crack500_configs(
            common_template,
            model_variants,
            unet_config,
            deeplab_config,
            pspnet_config,
            lmm_config,
            segformer_config,
            crack500_root,
            output_root,
        )
    )
    return generated_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline experiment configs.")
    parser.add_argument("--baseline_config", default="configs/aa/hrsegnetb16.yml", type=str)
    parser.add_argument("--final_config", default="configs/aa/hrsegnetb16_asppdelite.yml", type=str)
    parser.add_argument("--unet_config", default="configs/aa/unet.yml", type=str)
    parser.add_argument("--deeplab_config", default="configs/aa/deeplabv3p.yml", type=str)
    parser.add_argument("--pspnet_config", default="configs/aa/pspnet.yml", type=str)
    parser.add_argument("--lmm_config", default="configs/aa/lmnet.yml", type=str)
    parser.add_argument("--segformer_config", default="configs/aa/segformer_b0_720.yml", type=str)
    parser.add_argument("--b32_baseline_config", default="configs/aa/hrsegnetb32.yml", type=str)
    parser.add_argument("--b32_final_config", default="configs/aa/hrsegnetb32_asppdelite.yml", type=str)
    parser.add_argument("--b64_baseline_config", default="configs/aa/hrsegnetb64.yml", type=str)
    parser.add_argument("--b64_final_config", default="configs/aa/hrsegnetb64_asppde.yml", type=str)
    parser.add_argument("--custom_root", default="dataset/custom", type=str)
    parser.add_argument("--crack500_root", default="dataset/Crack500", type=str)
    parser.add_argument("--output_root", default="configs/offline_experiments", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    generated_paths = generate_all_configs(
        baseline_config_path=args.baseline_config,
        final_config_path=args.final_config,
        unet_config_path=args.unet_config,
        deeplab_config_path=args.deeplab_config,
        pspnet_config_path=args.pspnet_config,
        lmm_config_path=args.lmm_config,
        segformer_config_path=args.segformer_config,
        b32_baseline_config_path=args.b32_baseline_config,
        b32_final_config_path=args.b32_final_config,
        b64_baseline_config_path=args.b64_baseline_config,
        b64_final_config_path=args.b64_final_config,
        custom_root=args.custom_root,
        crack500_root=args.crack500_root,
        output_root=args.output_root,
    )
    print(f"Generated {len(generated_paths)} configs.")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()

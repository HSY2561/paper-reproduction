import tempfile
import unittest
from pathlib import Path

import yaml


class GenerateConfigsTests(unittest.TestCase):
    def test_generate_all_configs_keeps_ablation_training_strategy_consistent(self):
        from scripts.offline_experiments.generate_ablation_configs import generate_all_configs

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "configs" / "aa"
            source_dir.mkdir(parents=True)

            baseline_cfg = {
                "batch_size": 8,
                "iters": 18000,
                "train_dataset": {
                    "dataset_root": "data",
                    "train_path": "data/train.txt",
                    "num_classes": 2,
                    "type": "Dataset",
                    "mode": "train",
                    "transforms": [{"type": "Normalize"}],
                },
                "val_dataset": {
                    "dataset_root": "data",
                    "val_path": "data/val.txt",
                    "num_classes": 2,
                    "type": "Dataset",
                    "mode": "val",
                    "transforms": [{"type": "Normalize"}],
                },
                "model": {"type": "HrSegNetB16"},
                "optimizer": {"type": "SGD", "momentum": 0.9},
                "loss": {"types": [{"type": "OhemCrossEntropyLoss"}], "coef": [1]},
                "lr_scheduler": {"type": "PolynomialDecay", "learning_rate": 0.01},
            }
            final_cfg = {
                **baseline_cfg,
                "model": {
                    "type": "HrSegNet_Lite",
                    "base": 16,
                    "aspp_out_channels": 64,
                    "decoder_channels": 64,
                },
            }
            unet_cfg = {
                **baseline_cfg,
                "model": {"type": "UNet", "num_classes": 2},
            }
            deeplab_cfg = {
                **baseline_cfg,
                "model": {
                    "type": "DeepLabV3P",
                    "num_classes": 2,
                    "backbone": {"type": "ResNet18_vd"},
                },
            }
            pspnet_cfg = {
                **baseline_cfg,
                "model": {"type": "PSPNet", "num_classes": 2, "backbone": {"type": "ResNet18_vd"}},
            }
            lmm_cfg = {
                **baseline_cfg,
                "model": {"type": "LMMNet", "num_classes": 2},
            }
            segformer_cfg = {
                **baseline_cfg,
                "model": {
                    "type": "SegFormer",
                    "backbone": {"type": "MixVisionTransformer_B0"},
                    "embedding_dim": 256,
                    "num_classes": 2,
                },
            }

            for name, cfg in {
                "hrsegnetb16.yml": baseline_cfg,
                "hrsegnetb16_asppdelite.yml": final_cfg,
                "unet.yml": unet_cfg,
                "deeplabv3p.yml": deeplab_cfg,
                "pspnet.yml": pspnet_cfg,
                "lmnet.yml": lmm_cfg,
                "segformer_b0_720.yml": segformer_cfg,
                "hrsegnetb32.yml": {**baseline_cfg, "model": {"type": "HrSegNetB32"}},
                "hrsegnetb32_asppdelite.yml": {
                    **baseline_cfg,
                    "model": {
                        "type": "HrSegNet_Lite",
                        "base": 32,
                        "aspp_out_channels": 64,
                        "decoder_channels": 64,
                    },
                },
                "hrsegnetb64.yml": {**baseline_cfg, "model": {"type": "HrSegNetB64"}},
                "hrsegnetb64_asppde.yml": {
                    **baseline_cfg,
                    "model": {
                        "type": "HrSegNet_Lite",
                        "base": 64,
                        "aspp_out_channels": 64,
                        "decoder_channels": 64,
                    },
                },
            }.items():
                (source_dir / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            output_root = tmp_path / "configs" / "offline_experiments"
            generated = generate_all_configs(
                baseline_config_path=source_dir / "hrsegnetb16.yml",
                final_config_path=source_dir / "hrsegnetb16_asppdelite.yml",
                unet_config_path=source_dir / "unet.yml",
                deeplab_config_path=source_dir / "deeplabv3p.yml",
                pspnet_config_path=source_dir / "pspnet.yml",
                lmm_config_path=source_dir / "lmnet.yml",
                segformer_config_path=source_dir / "segformer_b0_720.yml",
                custom_root="dataset/custom",
                crack500_root="dataset/Crack500",
                output_root=output_root,
            )

            baseline = yaml.safe_load((output_root / "ablation" / "b16_baseline.yml").read_text(encoding="utf-8"))
            aspp = yaml.safe_load((output_root / "ablation" / "b16_aspp.yml").read_text(encoding="utf-8"))
            decoder = yaml.safe_load((output_root / "ablation" / "b16_decoder.yml").read_text(encoding="utf-8"))
            final = yaml.safe_load((output_root / "ablation" / "b16_final.yml").read_text(encoding="utf-8"))

            b32_baseline = yaml.safe_load((output_root / "ablation" / "b32_baseline.yml").read_text(encoding="utf-8"))
            b32_final = yaml.safe_load((output_root / "ablation" / "b32_final.yml").read_text(encoding="utf-8"))
            b64_baseline = yaml.safe_load((output_root / "ablation" / "b64_baseline.yml").read_text(encoding="utf-8"))
            b64_final = yaml.safe_load((output_root / "ablation" / "b64_final.yml").read_text(encoding="utf-8"))

            self.assertEqual(len(generated), 27)
            self.assertEqual(baseline["model"]["type"], "HrSegNetB16")
            self.assertEqual(aspp["model"]["type"], "HrSegNetB16ASPP")
            self.assertEqual(decoder["model"]["type"], "HrSegNetB16Decoder")
            self.assertEqual(final["model"]["type"], "HrSegNetB16A")
            self.assertEqual(b32_baseline["model"]["type"], "HrSegNetB32")
            self.assertEqual(b32_final["model"]["type"], "HrSegNetB16A")
            self.assertEqual(b32_final["model"]["base"], 32)
            self.assertEqual(b64_baseline["model"]["type"], "HrSegNetB64")
            self.assertEqual(b64_final["model"]["type"], "HrSegNetB16A")
            self.assertEqual(b64_final["model"]["base"], 64)
            self.assertEqual(baseline["batch_size"], aspp["batch_size"])
            self.assertEqual(baseline["iters"], decoder["iters"])
            self.assertEqual(baseline["optimizer"], final["optimizer"])
            self.assertEqual(baseline["lr_scheduler"], final["lr_scheduler"])
            self.assertEqual(final["train_dataset"]["dataset_root"], "dataset/custom")

            crack500_unet = yaml.safe_load(
                (output_root / "crack500" / "unet.yml").read_text(encoding="utf-8")
            )
            crack500_b32 = yaml.safe_load(
                (output_root / "crack500" / "hrsegnet_b32.yml").read_text(encoding="utf-8")
            )
            crack500_b32_a = yaml.safe_load(
                (output_root / "crack500" / "hrsegnet_b32_a.yml").read_text(encoding="utf-8")
            )
            crack500_b64 = yaml.safe_load(
                (output_root / "crack500" / "hrsegnet_b64.yml").read_text(encoding="utf-8")
            )
            crack500_b64_a = yaml.safe_load(
                (output_root / "crack500" / "hrsegnet_b64_a.yml").read_text(encoding="utf-8")
            )
            crack500_segformer = yaml.safe_load(
                (output_root / "crack500" / "segformer_b0.yml").read_text(encoding="utf-8")
            )
            self.assertEqual(crack500_unet["train_dataset"]["dataset_root"], "dataset/Crack500")
            self.assertEqual(crack500_b32["model"]["base"], 32)
            self.assertEqual(crack500_b32_a["model"]["base"], 32)
            self.assertEqual(crack500_b64["model"]["base"], 64)
            self.assertEqual(crack500_b64_a["model"]["base"], 64)
            self.assertEqual(crack500_segformer["model"]["type"], "SegFormer")


if __name__ == "__main__":
    unittest.main()

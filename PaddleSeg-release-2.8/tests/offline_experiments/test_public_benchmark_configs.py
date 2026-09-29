import unittest
from pathlib import Path


class PublicBenchmarkConfigTests(unittest.TestCase):
    def test_build_public_benchmark_specs_contains_expected_models(self):
        from scripts.offline_experiments.run_public_benchmarks import build_public_benchmark_specs

        specs = build_public_benchmark_specs()

        self.assertIn("crackseg9k", specs)
        self.assertIn("crack500_kaggle", specs)
        self.assertIn("segformer_b0", specs["crackseg9k"]["paddle"])
        self.assertIn("segformer_b0", specs["crack500_kaggle"]["paddle"])
        self.assertIn("pspnet", specs["crack500_kaggle"]["paddle"])
        self.assertIn("unet", specs["crack500_kaggle"]["paddle"])
        self.assertIn("deepcrack", specs["crack500_kaggle"]["torch"])
        self.assertIn("shufflenetv2", specs["crack500_kaggle"]["torch"])
        self.assertEqual(
            specs["crackseg9k"]["paddle"]["segformer_b0"]["dataset_root"],
            "dataset/Crackseg9k",
        )
        self.assertEqual(
            specs["crack500_kaggle"]["paddle"]["segformer_b0"]["dataset_root"],
            "dataset/Crack500_kaggle",
        )
        self.assertEqual(
            specs["crack500_kaggle"]["paddle"]["pspnet"]["output_dir"],
            "output/offline_experiments_torch/crack500_kaggle/pspnet",
        )
        self.assertEqual(
            specs["crack500_kaggle"]["paddle"]["unet"]["config_path"],
            "configs/offline_experiments/crack500_kaggle/unet.yml",
        )
        self.assertEqual(
            specs["crack500_kaggle"]["torch"]["shufflenetv2"]["output_dir"],
            "output/offline_experiments_torch/crack500_kaggle/shufflenetv2",
        )

    def test_build_segformer_config_rewrites_pretrained_path_to_relative(self):
        from scripts.offline_experiments.run_public_benchmarks import _build_paddle_config

        base_config = {
            "model": {
                "type": "SegFormer",
                "backbone": {
                    "type": "MixVisionTransformer_B0",
                    "pretrained": str(Path("D:/repo/pretrained/mix_vision_transformer_b0/model.pdparams")),
                },
            },
            "train_dataset": {"dataset_root": "data", "train_path": "data/train.txt"},
            "val_dataset": {"dataset_root": "data", "val_path": "data/val.txt"},
        }

        cfg = _build_paddle_config(
            base_config,
            "output/offline_experiments_torch/crackseg9k/segformer_b0",
            "dataset/Crackseg9k",
        )
        self.assertFalse(Path(cfg["model"]["backbone"]["pretrained"]).is_absolute())
        self.assertEqual(cfg["save_dir"], "./output/offline_experiments_torch/crackseg9k/segformer_b0")


if __name__ == "__main__":
    unittest.main()

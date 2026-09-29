import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class PredictCrack500KaggleModelsTests(unittest.TestCase):
    def test_discover_experiments_uses_metrics_metadata_when_runtime_config_missing(self):
        from scripts.offline_experiments.predict_crack500_kaggle_models import discover_experiments

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_root = root / "repo"
            result_root = repo_root / "output" / "offline_experiments_torch" / "crack500_kaggle"
            model_dir = result_root / "b32_final_r1-6-12-18_c64"
            config_path = repo_root / "output" / "offline_experiments" / "ablation" / "b32_final_r1-6-12-18_c64" / "runtime_config.yml"
            best_model = model_dir / "best_model" / "model.pdparams"

            best_model.parent.mkdir(parents=True, exist_ok=True)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("model:\n  type: HrSegNetB16A\n", encoding="utf-8")
            best_model.write_bytes(b"weights")
            (model_dir / "metrics.json").write_text(
                (
                    "{\n"
                    '  "experiment_name": "b32_final_r1-6-12-18_c64",\n'
                    '  "config_path": "output/offline_experiments/ablation/b32_final_r1-6-12-18_c64/runtime_config.yml",\n'
                    '  "best_model_path": "output/offline_experiments_torch/crack500_kaggle/b32_final_r1-6-12-18_c64/best_model/model.pdparams"\n'
                    "}\n"
                ),
                encoding="utf-8",
            )

            experiments = discover_experiments(result_root, repo_root=repo_root)

            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0].name, "b32_final_r1-6-12-18_c64")
            self.assertEqual(experiments[0].family, "paddle")
            self.assertEqual(experiments[0].config_path, config_path)
            self.assertEqual(experiments[0].model_path, best_model)

    def test_discover_experiments_detects_torch_checkpoints(self):
        from scripts.offline_experiments.predict_crack500_kaggle_models import discover_experiments

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_root = root / "repo"
            result_root = repo_root / "output" / "offline_experiments_torch" / "crack500_kaggle"
            model_dir = result_root / "deepcrack"
            runtime_config = model_dir / "runtime_config.yml"
            best_model = model_dir / "best_model.pt"

            model_dir.mkdir(parents=True, exist_ok=True)
            runtime_config.write_text("model_type: deepcrack\n", encoding="utf-8")
            best_model.write_bytes(b"weights")

            experiments = discover_experiments(result_root, repo_root=repo_root)

            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0].family, "torch")
            self.assertEqual(experiments[0].config_path, runtime_config)
            self.assertEqual(experiments[0].model_path, best_model)

    def test_save_binary_mask_writes_0_and_255_values(self):
        from scripts.offline_experiments.predict_crack500_kaggle_models import save_binary_mask

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "mask.png"
            mask = np.array([[False, True], [True, False]], dtype=bool)

            save_binary_mask(mask, output_path)

            arr = np.array(Image.open(output_path))
            self.assertEqual(set(arr.reshape(-1).tolist()), {0, 255})


if __name__ == "__main__":
    unittest.main()

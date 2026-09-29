import json
import tempfile
import unittest
from pathlib import Path


class LMMTorchSummaryTests(unittest.TestCase):
    def test_build_summary_row_writes_expected_fields(self):
        from experiments.lmm_torch.summarize import build_summary_row

        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir) / "output" / "offline_experiments_torch" / "crack500" / "lmnet"
            exp_dir.mkdir(parents=True)
            (exp_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "lmnet",
                        "dataset": "Crack500",
                        "config_path": "experiments/lmm_torch/configs/crack500.yml",
                        "best_model_path": "output/offline_experiments_torch/crack500/lmnet/best_model.pt",
                        "miou": 0.8123,
                        "metrics": {
                            "foreground_iou": 0.6345,
                            "foreground_precision": 0.801,
                            "foreground_recall": 0.744,
                            "foreground_f1": 0.7714,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (exp_dir / "model_stats.json").write_text(
                json.dumps({"params_m": 0.82}),
                encoding="utf-8",
            )
            (exp_dir / "fps.json").write_text(
                json.dumps({"fps": 4.3}),
                encoding="utf-8",
            )

            row = build_summary_row(exp_dir)
            self.assertEqual(row["Experiment"], "lmnet")
            self.assertEqual(row["Dataset"], "Crack500")
            self.assertAlmostEqual(row["Precision"], 0.801)
            self.assertAlmostEqual(row["Recall"], 0.744)
            self.assertAlmostEqual(row["F1"], 0.7714)
            self.assertAlmostEqual(row["IoU"], 0.6345)
            self.assertAlmostEqual(row["mIoU"], 0.8123)
            self.assertAlmostEqual(row["Params_M"], 0.82)
            self.assertAlmostEqual(row["FPS"], 4.3)
            self.assertTrue((exp_dir / "summary_row.json").exists())


if __name__ == "__main__":
    unittest.main()

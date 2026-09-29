import tempfile
import unittest
from pathlib import Path


class LMMOriginalSummaryTests(unittest.TestCase):
    def test_summary_row_exports_expected_fields(self):
        from lmm_original_crack500.common import dump_json
        from lmm_original_crack500.summarize import SUMMARY_COLUMNS, build_summary_row

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dump_json(
                {
                    "experiment_name": "lmnet_original",
                    "dataset": "Crack500",
                    "config_path": "configs/lmnet_crack500.yml",
                    "best_model_path": "output/best_model.pt",
                    "miou": 0.7,
                    "metrics": {
                        "foreground_precision": 0.8,
                        "foreground_recall": 0.6,
                        "foreground_f1": 0.6857,
                        "foreground_iou": 0.52,
                    },
                },
                root / "metrics.json",
            )
            dump_json({"params_m": 0.83}, root / "model_stats.json")
            dump_json({"fps": 12.5}, root / "fps.json")

            row = build_summary_row(root)

            self.assertEqual(list(row.keys()), SUMMARY_COLUMNS)
            self.assertEqual(row["Experiment"], "lmnet_original")
            self.assertEqual(row["Dataset"], "Crack500")
            self.assertEqual(row["Params_M"], 0.83)
            self.assertEqual(row["FPS"], 12.5)


if __name__ == "__main__":
    unittest.main()

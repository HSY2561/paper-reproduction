import tempfile
import unittest
from pathlib import Path

import pandas as pd


class SummarizeResultsTests(unittest.TestCase):
    def test_summarize_results_writes_csv_and_xlsx(self):
        from scripts.offline_experiments.summarize_results import summarize_result_roots

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exp_dir = tmp_path / "output" / "offline_experiments" / "ablation" / "b16_baseline"
            exp_dir.mkdir(parents=True)

            (exp_dir / "metrics.json").write_text(
                (
                    "{"
                    "\"experiment_name\": \"b16_baseline\", "
                    "\"dataset\": \"custom\", "
                    "\"metrics\": {"
                    "\"foreground_iou\": 0.81, "
                    "\"foreground_dice\": 0.895, "
                    "\"foreground_precision\": 0.91, "
                    "\"foreground_recall\": 0.88, "
                    "\"foreground_f1\": 0.8947"
                    "}, "
                    "\"best_model_path\": \"output/offline_experiments/ablation/b16_baseline/best_model/model.pdparams\""
                    "}"
                ),
                encoding="utf-8",
            )
            (exp_dir / "model_stats.json").write_text(
                "{\"params_m\": 1.23}",
                encoding="utf-8",
            )
            (exp_dir / "fps.json").write_text(
                "{\"fps\": 45.6}",
                encoding="utf-8",
            )

            tables_dir = tmp_path / "output" / "offline_experiments" / "tables"
            result = summarize_result_roots(
                result_roots=[tmp_path / "output" / "offline_experiments" / "ablation"],
                output_dir=tables_dir,
                table_prefix="ablation_results",
            )

            csv_path = tables_dir / "ablation_results.csv"
            xlsx_path = tables_dir / "ablation_results.xlsx"

            self.assertEqual(len(result), 1)
            self.assertTrue(csv_path.exists())
            self.assertTrue(xlsx_path.exists())

            frame = pd.read_csv(csv_path)
            self.assertEqual(frame.loc[0, "Experiment"], "b16_baseline")
            self.assertAlmostEqual(frame.loc[0, "F1"], 0.8947, places=4)


if __name__ == "__main__":
    unittest.main()

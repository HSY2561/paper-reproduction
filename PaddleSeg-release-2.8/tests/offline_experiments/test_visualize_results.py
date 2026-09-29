import json
import tempfile
import unittest
from pathlib import Path


class VisualizeResultsTests(unittest.TestCase):
    def test_visualization_script_creates_four_figures(self):
        from scripts.offline_experiments.visualize_results import generate_all_figures

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ablation = tmp_path / "ablation"
            crack500 = tmp_path / "crack500"
            for root, names in [
                (ablation, ["model_a", "model_b"]),
                (crack500, ["net_a", "net_b"]),
            ]:
                for idx, name in enumerate(names, start=1):
                    exp_dir = root / name
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    summary = {
                        "Experiment": name,
                        "Dataset": root.name,
                        "Config": f"{name}.yml",
                        "BestModelPath": f"{name}.pdparams",
                        "Precision": 0.7 + idx * 0.01,
                        "Recall": 0.6 + idx * 0.01,
                        "F1": 0.65 + idx * 0.01,
                        "IoU": 0.5 + idx * 0.01,
                        "mIoU": 0.75 + idx * 0.01,
                        "Params_M": 1.0 + idx,
                        "FPS": 20.0 + idx,
                    }
                    (exp_dir / "summary_row.json").write_text(json.dumps(summary), encoding="utf-8")

            output_dir = tmp_path / "figures"
            paths = generate_all_figures(ablation, crack500, output_dir)

            self.assertEqual(set(paths.keys()), {"ablation_metrics", "ablation_efficiency", "crack500_metrics", "crack500_efficiency"})
            for path in paths.values():
                self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()

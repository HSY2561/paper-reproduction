import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


class RunAblationTests(unittest.TestCase):
    def test_parse_aspp_rate_sets(self):
        from scripts.offline_experiments.run_ablation import parse_aspp_rate_sets

        parsed = parse_aspp_rate_sets(["1,6,12,18", "1,8,16,24"])
        self.assertEqual(parsed, [[1, 6, 12, 18], [1, 8, 16, 24]])

    def test_create_aspp_search_experiments(self):
        from scripts.offline_experiments.run_ablation import create_aspp_search_experiments

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "b16_final.yml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "type": "HrSegNetB16A",
                            "base": 16,
                            "num_classes": 2,
                            "aspp_rates": [1, 6, 12, 18],
                            "aspp_out_channels": 64,
                        },
                        "save_dir": "./output/offline_experiments/ablation/b16_final",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            experiments = create_aspp_search_experiments(
                base_name="b16_final",
                base_config_path=config_path,
                rate_sets=[[1, 6, 12, 18], [1, 8, 16, 24]],
                out_channels=[32, 64],
                generated_config_root=tmp_path / "generated",
            )

            self.assertEqual(len(experiments), 4)
            expected_name = "b16_final_r1-6-12-18_c32"
            self.assertIn(expected_name, experiments)
            generated_cfg = yaml.safe_load(Path(experiments[expected_name]).read_text(encoding="utf-8"))
            self.assertEqual(generated_cfg["model"]["aspp_rates"], [1, 6, 12, 18])
            self.assertEqual(generated_cfg["model"]["aspp_out_channels"], 32)

    def test_experiments_include_b32_b64_variants(self):
        from scripts.offline_experiments.run_ablation import EXPERIMENTS

        for name in [
            "b32_baseline",
            "b32_aspp",
            "b32_decoder",
            "b32_final",
            "b64_baseline",
            "b64_aspp",
            "b64_decoder",
            "b64_final",
        ]:
            self.assertIn(name, EXPERIMENTS)

    def test_parse_args_accepts_b32_aspp_search_base(self):
        from scripts.offline_experiments import run_ablation

        with mock.patch(
            "sys.argv",
            [
                "run_ablation.py",
                "--aspp_search_base",
                "b32_final",
            ],
        ):
            args = run_ablation.parse_args()

        self.assertEqual(args.aspp_search_base, "b32_final")


if __name__ == "__main__":
    unittest.main()

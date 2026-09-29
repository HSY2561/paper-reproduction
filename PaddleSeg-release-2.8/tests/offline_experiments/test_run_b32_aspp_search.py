import unittest


class RunB32AsppSearchTests(unittest.TestCase):
    def test_build_delegate_command_uses_b32_defaults(self):
        from scripts.offline_experiments.run_b32_aspp_search import build_delegate_command

        command = build_delegate_command(
            rates=["1,6,12,18", "1,8,16,24"],
            channels=[32, 64],
            base_variant="b32_final",
            device="gpu",
            summary_prefix="b32_aspp_search",
            extra_args=[],
        )

        self.assertEqual(command[1], "scripts/offline_experiments/run_ablation.py")
        self.assertIn("--aspp_search_base", command)
        self.assertIn("b32_final", command)
        self.assertIn("--aspp_rates", command)
        self.assertIn("1,6,12,18", command)
        self.assertIn("--aspp_out_channels", command)
        self.assertIn("32", command)
        self.assertIn("--aspp_summary_prefix", command)
        self.assertIn("b32_aspp_search", command)


if __name__ == "__main__":
    unittest.main()

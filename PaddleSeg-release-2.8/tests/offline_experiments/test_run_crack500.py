import unittest


class RunCrack500Tests(unittest.TestCase):
    def test_experiments_include_b32_b64_and_segformer_variants(self):
        from scripts.offline_experiments.run_crack500 import EXPERIMENTS

        for name in ["hrsegnet_b32", "hrsegnet_b32_a", "hrsegnet_b64", "hrsegnet_b64_a", "segformer_b0"]:
            self.assertIn(name, EXPERIMENTS)


if __name__ == "__main__":
    unittest.main()

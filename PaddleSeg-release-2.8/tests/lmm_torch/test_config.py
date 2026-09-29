import unittest


class LMMTorchConfigTests(unittest.TestCase):
    def test_default_crack500_config_matches_expected_protocol(self):
        from experiments.lmm_torch.config import default_config

        cfg = default_config()
        self.assertEqual(cfg["batch_size"], 1)
        self.assertEqual(cfg["accumulation_steps"], 8)
        self.assertEqual(cfg["iters"], 12000)
        self.assertEqual(cfg["train_crop_size"], 720)
        self.assertEqual(cfg["dataset_root"], "dataset/Crack500")
        self.assertEqual(
            cfg["output_dir"],
            "output/offline_experiments_torch/crack500/lmnet",
        )


if __name__ == "__main__":
    unittest.main()

import unittest


class LMMOriginalConfigTests(unittest.TestCase):
    def test_default_config_matches_hrsegnet_crack500_protocol(self):
        from lmm_original_crack500.config import default_config

        cfg = default_config()
        self.assertEqual(cfg["model_type"], "lmnet")
        self.assertEqual(cfg["dataset_root"], "dataset/Crack500")
        self.assertEqual(cfg["train_list"], "dataset/Crack500/train.txt")
        self.assertEqual(cfg["val_list"], "dataset/Crack500/val.txt")
        self.assertEqual(cfg["test_list"], "dataset/Crack500/test.txt")
        self.assertEqual(cfg["train_crop_size"], 720)
        self.assertEqual(cfg["iters"], 12000)
        self.assertEqual(cfg["batch_size"], 1)
        self.assertEqual(cfg["accumulation_steps"], 8)
        self.assertEqual(cfg["optimizer"]["type"], "SGD")
        self.assertEqual(cfg["lr_scheduler"]["type"], "PolynomialDecay")


if __name__ == "__main__":
    unittest.main()

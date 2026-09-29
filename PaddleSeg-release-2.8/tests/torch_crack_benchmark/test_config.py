import unittest


class TorchCrackBenchmarkConfigTests(unittest.TestCase):
    def test_load_config_keeps_relative_paths(self):
        from torch_crack_benchmark.config import load_config

        cfg = load_config(
            overrides={
                "dataset_name": "Crackseg9k",
                "dataset_root": "dataset/Crackseg9k",
                "train_list": "dataset/Crackseg9k/train.txt",
                "val_list": "dataset/Crackseg9k/val.txt",
                "test_list": "dataset/Crackseg9k/test.txt",
            }
        )

        self.assertEqual(cfg["dataset_root"], "dataset/Crackseg9k")
        self.assertEqual(cfg["train_list"], "dataset/Crackseg9k/train.txt")
        self.assertEqual(cfg["val_list"], "dataset/Crackseg9k/val.txt")
        self.assertEqual(cfg["test_list"], "dataset/Crackseg9k/test.txt")


if __name__ == "__main__":
    unittest.main()

import unittest


class TorchCrackBenchmarkModelTests(unittest.TestCase):
    def test_resolve_model_spec_supports_required_models(self):
        from torch_crack_benchmark.model import resolve_model_spec

        for model_name in ["crackformer_ii", "deepcrack", "efficientnet", "mobilenetv3", "shufflenetv2"]:
            spec = resolve_model_spec(model_name)
            self.assertEqual(spec["name"], model_name)
            self.assertIn("module_name", spec)
            self.assertIn("factory_name", spec)

        shufflenet = resolve_model_spec("shufflenetv2")
        self.assertEqual(shufflenet["module_name"], "Shufflenetv2")
        self.assertEqual(shufflenet["factory_name"], "ShuffleNetV2Seg")
        self.assertEqual(shufflenet["kwargs"]["c1"], 232)
        self.assertEqual(shufflenet["kwargs"]["c2"], 464)


if __name__ == "__main__":
    unittest.main()

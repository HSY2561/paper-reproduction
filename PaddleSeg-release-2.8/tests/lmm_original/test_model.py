import unittest


class LMMOriginalModelSpecTests(unittest.TestCase):
    def test_resolve_model_spec_supports_three_original_models(self):
        from lmm_original_crack500.model import resolve_model_spec

        lmnet = resolve_model_spec("lmnet")
        shufflenet = resolve_model_spec("shufflenetv2")
        mobilenet = resolve_model_spec("mobilenetv3")

        self.assertEqual(lmnet["module_name"], "LM_Net")
        self.assertEqual(lmnet["factory_name"], "LM_Net")

        self.assertEqual(shufflenet["module_name"], "Shufflenetv2")
        self.assertEqual(shufflenet["factory_name"], "ShuffleNetV2Seg")
        self.assertEqual(shufflenet["kwargs"]["nclass"], 1)

        self.assertEqual(mobilenet["module_name"], "Mobilenetv3")
        self.assertEqual(mobilenet["factory_name"], "MobileNetV3Seg")
        self.assertEqual(mobilenet["kwargs"]["mode"], "small")

    def test_resolve_model_spec_rejects_unknown_model(self):
        from lmm_original_crack500.model import resolve_model_spec

        with self.assertRaises(ValueError):
            resolve_model_spec("unknown_net")


if __name__ == "__main__":
    unittest.main()

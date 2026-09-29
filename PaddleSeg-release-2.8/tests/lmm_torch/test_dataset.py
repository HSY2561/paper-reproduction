import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class LMMTorchDatasetTests(unittest.TestCase):
    def test_parse_sample_line_and_resolve_paths(self):
        from experiments.lmm_torch.dataset import parse_sample_line, resolve_sample_paths

        line = "JPEGImages/train/a.jpg masks_bin/train/a_mask.png"
        image_rel, mask_rel = parse_sample_line(line)
        self.assertEqual(image_rel, "JPEGImages/train/a.jpg")
        self.assertEqual(mask_rel, "masks_bin/train/a_mask.png")

        image_path, mask_path = resolve_sample_paths("dataset/Crack500", line)
        self.assertEqual(image_path.as_posix(), "dataset/Crack500/JPEGImages/train/a.jpg")
        self.assertEqual(mask_path.as_posix(), "dataset/Crack500/masks_bin/train/a_mask.png")

    def test_load_sample_list_reads_non_empty_lines(self):
        from experiments.lmm_torch.dataset import load_sample_list

        with tempfile.TemporaryDirectory() as tmp_dir:
            list_path = Path(tmp_dir) / "train.txt"
            list_path.write_text(
                "\nJPEGImages/train/a.jpg masks_bin/train/a_mask.png\n\nJPEGImages/train/b.jpg masks_bin/train/b_mask.png\n",
                encoding="utf-8",
            )
            samples = load_sample_list(list_path)
            self.assertEqual(len(samples), 2)

    def test_validate_sample_shapes_detects_mismatch(self):
        from experiments.lmm_torch.dataset import validate_sample_shapes

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "img.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (64, 32)).save(image_path)
            Image.new("L", (64, 32)).save(mask_path)
            validate_sample_shapes(image_path, mask_path)

            Image.new("L", (32, 64)).save(mask_path)
            with self.assertRaises(ValueError):
                validate_sample_shapes(image_path, mask_path)

    def test_random_crop_supports_numpy_generator(self):
        from experiments.lmm_torch.dataset import _random_crop

        image = Image.new("RGB", (32, 32))
        mask = Image.new("L", (32, 32))
        cropped_image, cropped_mask = _random_crop(image, mask, (16, 16), np.random.default_rng(0))
        self.assertEqual(cropped_image.size, (16, 16))
        self.assertEqual(cropped_mask.size, (16, 16))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class PrepareDatasetsTests(unittest.TestCase):
    def test_prepare_crack500_lists_generates_relative_pairs(self):
        from scripts.offline_experiments.prepare_datasets import prepare_crack500_lists

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "Crack500"
            for split in ("train", "val", "test"):
                image_dir = root / split / "images"
                mask_dir = root / split / "masks"
                image_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)
                for index in range(2):
                    (image_dir / f"sample_{index}.jpg").write_bytes(b"img")
                    (mask_dir / f"sample_{index}.png").write_bytes(b"mask")

            summary = prepare_crack500_lists(root, normalize_masks=False)

            self.assertEqual(summary["train"], 2)
            self.assertEqual(summary["val"], 2)
            self.assertEqual(summary["test"], 2)

            train_lines = (root / "train.txt").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(
                train_lines[0],
                "train/images/sample_0.jpg train/masks/sample_0.png",
            )

    def test_prepare_crack500_lists_binarizes_masks_for_training(self):
        from scripts.offline_experiments.prepare_datasets import prepare_crack500_lists

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "Crack500"
            for split in ("train", "val", "test"):
                image_dir = root / split / "images"
                mask_dir = root / split / "masks"
                image_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)

                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                    image_dir / "sample_0.jpg"
                )
                mask = np.zeros((8, 8), dtype=np.uint8)
                mask[1:3, 1:3] = 3
                mask[3:5, 3:5] = 254
                mask[5:7, 5:7] = 255
                Image.fromarray(mask).save(mask_dir / "sample_0.jpg")

            prepare_crack500_lists(root)

            train_line = (root / "train.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(
                train_line,
                "train/images/sample_0.jpg train/masks_bin/sample_0.png",
            )
            bin_mask = np.array(Image.open(root / "train" / "masks_bin" / "sample_0.png"))
            self.assertTrue(set(np.unique(bin_mask).tolist()).issubset({0, 1}))

    def test_copy_custom_dataset_dry_run_does_not_modify_destination(self):
        from scripts.offline_experiments.prepare_datasets import copy_custom_dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            src_root = Path(tmp_dir) / "data"
            dst_root = Path(tmp_dir) / "dataset" / "custom"
            (src_root / "images" / "train").mkdir(parents=True)
            (src_root / "annotations" / "train").mkdir(parents=True)
            (src_root / "images" / "train" / "a.jpg").write_bytes(b"img")
            (src_root / "annotations" / "train" / "a.png").write_bytes(b"mask")
            (src_root / "train.txt").write_text(
                "images/train/a.jpg annotations/train/a.png\n",
                encoding="utf-8",
            )

            summary = copy_custom_dataset(src_root, dst_root, dry_run=True)

            self.assertGreaterEqual(summary["files_to_copy"], 1)
            self.assertFalse(dst_root.exists())

    def test_prepare_crackseg9k_lists_generates_root_level_splits(self):
        from scripts.offline_experiments.prepare_datasets import prepare_crackseg9k_lists

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "Crackseg9k"
            vol1_images = root / "Final-Dataset-Vol1" / "Images"
            vol2_images = root / "Final-Dataset-Vol2" / "Final-Dataset-Vol2" / "Images-2"
            mask_dir = root / "Final-Dataset-Vol1" / "Final_Masks" / "Masks"
            split_dir = root / "Final-Dataset-Vol1" / "Final_Masks"
            vol1_images.mkdir(parents=True)
            vol2_images.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            for name, image_dir in (
                ("sample_a.png", vol1_images),
                ("sample_b.png", vol1_images),
                ("sample_c.png", vol2_images),
                ("sample_d.png", vol2_images),
            ):
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / name)
                mask = np.zeros((8, 8), dtype=np.uint8)
                mask[2:6, 2:6] = 255
                Image.fromarray(mask).save(mask_dir / name)

            (split_dir / "train.txt").write_text(
                "sample_a.png\nsample_b.png\nsample_c.png\n",
                encoding="utf-8",
            )
            (split_dir / "test.txt").write_text("sample_d.png\n", encoding="utf-8")

            summary = prepare_crackseg9k_lists(root, val_ratio=1 / 3, seed=7)

            self.assertEqual(summary["train"], 2)
            self.assertEqual(summary["val"], 1)
            self.assertEqual(summary["test"], 1)
            self.assertEqual(summary["normalized_masks"], 4)

            train_lines = (root / "train.txt").read_text(encoding="utf-8").strip().splitlines()
            val_lines = (root / "val.txt").read_text(encoding="utf-8").strip().splitlines()
            test_lines = (root / "test.txt").read_text(encoding="utf-8").strip().splitlines()

            self.assertEqual(len(train_lines), 2)
            self.assertEqual(len(val_lines), 1)
            self.assertEqual(
                test_lines[0],
                "Final-Dataset-Vol2/Final-Dataset-Vol2/Images-2/sample_d.png masks_bin/sample_d.png",
            )
            for line in train_lines + val_lines + test_lines:
                self.assertIn("masks_bin/", line)

            bin_mask = np.array(Image.open(root / "masks_bin" / "sample_a.png"))
            self.assertTrue(set(np.unique(bin_mask).tolist()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()

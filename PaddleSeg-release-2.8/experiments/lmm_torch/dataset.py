from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


def parse_sample_line(line):
    parts = line.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid sample line: {line!r}")
    return parts[0], parts[1]


def resolve_sample_paths(dataset_root, line):
    dataset_root = Path(dataset_root)
    image_rel, mask_rel = parse_sample_line(line)
    return dataset_root / image_rel, dataset_root / mask_rel


def load_sample_list(list_path):
    list_path = Path(list_path)
    samples = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(line)
    return samples


def validate_sample_shapes(image_path, mask_path):
    with Image.open(image_path) as image_obj:
        image_size = image_obj.size
    with Image.open(mask_path) as mask_obj:
        mask_size = mask_obj.size
    if image_size != mask_size:
        raise ValueError(
            f"Image-mask size mismatch: image={image_path} {image_size}, mask={mask_path} {mask_size}"
        )


def _torch_modules():
    import torch
    from torch.utils.data import Dataset

    return torch, Dataset


def _round_to_step(value, step_size):
    if step_size <= 0:
        return value
    return round(value / step_size) * step_size


def _resize_by_scale(image, mask, scale):
    width, height = image.size
    target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = image.resize(target_size, resample=Image.Resampling.BILINEAR)
    mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)
    return image, mask


def _random_rescale(image, mask, cfg, rng):
    min_scale = cfg["min_scale_factor"]
    max_scale = cfg["max_scale_factor"]
    step = cfg["scale_step_size"]
    scale = rng.uniform(min_scale, max_scale)
    scale = _round_to_step(scale, step)
    scale = max(min_scale, min(max_scale, scale))
    return _resize_by_scale(image, mask, scale)


def _pad_if_needed(image, mask, target_size):
    target_w, target_h = target_size
    width, height = image.size
    pad_w = max(0, target_w - width)
    pad_h = max(0, target_h - height)
    if pad_w == 0 and pad_h == 0:
        return image, mask

    image_np = np.asarray(image)
    mask_np = np.asarray(mask)
    if image_np.ndim == 2:
        image_np = image_np[:, :, None]

    padded_image = np.zeros((height + pad_h, width + pad_w, image_np.shape[2]), dtype=image_np.dtype)
    padded_image[:height, :width] = image_np
    padded_mask = np.zeros((height + pad_h, width + pad_w), dtype=mask_np.dtype)
    padded_mask[:height, :width] = mask_np
    return Image.fromarray(padded_image), Image.fromarray(padded_mask)


def _random_crop(image, mask, target_size, rng):
    image, mask = _pad_if_needed(image, mask, target_size)
    target_w, target_h = target_size
    width, height = image.size
    if width == target_w and height == target_h:
        return image, mask
    left = int(rng.integers(0, width - target_w + 1))
    top = int(rng.integers(0, height - target_h + 1))
    box = (left, top, left + target_w, top + target_h)
    return image.crop(box), mask.crop(box)


def _random_horizontal_flip(image, mask, rng):
    if rng.random() < 0.5:
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image, mask


def _random_distort(image, cfg, rng):
    brightness = cfg.get("brightness_range", 0.5)
    contrast = cfg.get("contrast_range", 0.5)
    saturation = cfg.get("saturation_range", 0.5)

    image = ImageEnhance.Brightness(image).enhance(1.0 + rng.uniform(-brightness, brightness))
    image = ImageEnhance.Contrast(image).enhance(1.0 + rng.uniform(-contrast, contrast))
    image = ImageEnhance.Color(image).enhance(1.0 + rng.uniform(-saturation, saturation))
    return image


class Crack500Dataset:
    def __new__(cls, *args, **kwargs):
        _, dataset_base = _torch_modules()

        class _Crack500Dataset(dataset_base):
            def __init__(self, cfg, list_path=None, mode="train"):
                self.cfg = cfg
                self.dataset_root = Path(cfg["dataset_root"])
                self.mode = mode
                self.list_path = Path(list_path or cfg[f"{mode}_list"])
                self.samples = load_sample_list(self.list_path)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                torch, _ = _torch_modules()
                rng = np.random.default_rng()
                line = self.samples[index]
                image_path, mask_path = resolve_sample_paths(self.dataset_root, line)
                validate_sample_shapes(image_path, mask_path)

                with Image.open(image_path) as image_obj:
                    image = image_obj.convert("RGB")
                with Image.open(mask_path) as mask_obj:
                    mask = mask_obj.convert("L")

                if self.mode == "train":
                    image, mask = _random_rescale(image, mask, self.cfg, rng)
                    image, mask = _random_crop(
                        image,
                        mask,
                        (self.cfg["train_crop_size"], self.cfg["train_crop_size"]),
                        rng,
                    )
                    image, mask = _random_horizontal_flip(image, mask, rng)
                    image = _random_distort(image, self.cfg, rng)

                image_np = np.asarray(image).astype("float32") / 255.0
                mean = np.array(self.cfg["normalize"]["mean"], dtype="float32").reshape(1, 1, 3)
                std = np.array(self.cfg["normalize"]["std"], dtype="float32").reshape(1, 1, 3)
                image_np = (image_np - mean) / std
                image_np = np.transpose(image_np, (2, 0, 1))
                mask_np = np.asarray(mask).astype("int64")
                mask_np = (mask_np > 0).astype("int64")

                return {
                    "image": torch.from_numpy(image_np),
                    "mask": torch.from_numpy(mask_np),
                }

        return _Crack500Dataset(*args, **kwargs)

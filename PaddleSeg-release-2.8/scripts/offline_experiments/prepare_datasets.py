import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import ensure_dir


def _iter_files(root):
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.is_file())


def copy_custom_dataset(src_root, dst_root, dry_run=False):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_root.exists():
        raise FileNotFoundError(f"Custom dataset source does not exist: {src_root}")

    files = _iter_files(src_root)
    summary = {
        "source_root": str(src_root),
        "destination_root": str(dst_root),
        "files_to_copy": len(files),
    }

    if dry_run:
        return summary

    for src_path in files:
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        ensure_dir(dst_path.parent)
        shutil.copy2(src_path, dst_path)
    return summary


def _match_image_and_mask_paths(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_map = {_normalized_pair_key(path.stem): path for path in image_dir.iterdir() if path.is_file()}
    mask_map = {_normalized_pair_key(path.stem): path for path in mask_dir.iterdir() if path.is_file()}

    missing_masks = sorted(set(image_map) - set(mask_map))
    extra_masks = sorted(set(mask_map) - set(image_map))
    if missing_masks or extra_masks:
        raise ValueError(
            f"Image-mask mismatch under {image_dir.parent}: "
            f"missing_masks={missing_masks[:5]}, extra_masks={extra_masks[:5]}"
        )

    return [(image_map[key], mask_map[key]) for key in sorted(image_map)]


def _normalized_pair_key(stem):
    if stem.endswith("_mask"):
        return stem[: -len("_mask")]
    return stem


def _detect_crack500_layout(crack500_root):
    crack500_root = Path(crack500_root)
    split_layout = all(
        (crack500_root / split / "images").exists() and (crack500_root / split / "masks").exists()
        for split in ("train", "val", "test")
    )
    raw_layout = all(
        (crack500_root / "JPEGImages" / split).exists()
        and (crack500_root / "Annotations" / split).exists()
        for split in ("train", "val", "test")
    )
    if split_layout:
        return "split"
    if raw_layout:
        return "raw"
    raise FileNotFoundError(
        f"Unsupported Crack500 layout under {crack500_root}. "
        "Expected either split/{images,masks} or {JPEGImages,Annotations}/split."
    )


def _get_split_dirs(crack500_root, split, layout, normalized_mask_dir_name):
    crack500_root = Path(crack500_root)
    if layout == "split":
        image_dir = crack500_root / split / "images"
        mask_dir = crack500_root / split / "masks"
        normalized_mask_dir = crack500_root / split / normalized_mask_dir_name
    else:
        image_dir = crack500_root / "JPEGImages" / split
        mask_dir = crack500_root / "Annotations" / split
        normalized_mask_dir = crack500_root / normalized_mask_dir_name / split
    return image_dir, mask_dir, normalized_mask_dir


def _binarize_mask(mask_path, output_path, threshold=127, target_size=None):
    # Crack500 masks are often JPG and may contain non-class values after compression.
    # Convert to strict 0/1 PNG labels for stable segmentation training.
    mask = Image.open(mask_path).convert("L")
    if target_size is not None and mask.size != tuple(target_size):
        expected_size = tuple(target_size)
        if mask.size == (expected_size[1], expected_size[0]):
            # Common raw-data issue: width/height swapped for a few masks.
            mask = mask.transpose(Image.Transpose.TRANSPOSE)
        else:
            # Fallback to nearest-neighbor resize to keep label IDs intact.
            mask = mask.resize(expected_size, resample=Image.Resampling.NEAREST)
    mask_array = np.array(mask, dtype=np.uint8)
    binary = (mask_array > threshold).astype(np.uint8)
    # Write via a temp file and replace atomically so dataloader never sees a half-written PNG.
    output_path = Path(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    Image.fromarray(binary, mode="L").save(temp_path, format="PNG")
    os.replace(temp_path, output_path)


def prepare_crack500_lists(
    crack500_root,
    dry_run=False,
    normalize_masks=True,
    normalize_threshold=127,
    normalized_mask_dir_name="masks_bin",
):
    crack500_root = Path(crack500_root)
    if not crack500_root.exists():
        raise FileNotFoundError(f"Crack500 root does not exist: {crack500_root}")

    layout = _detect_crack500_layout(crack500_root)
    summary = {"layout": layout, "size_fixes": 0}
    for split in ("train", "val", "test"):
        image_dir, mask_dir, normalized_mask_dir = _get_split_dirs(
            crack500_root, split, layout, normalized_mask_dir_name
        )
        pairs = _match_image_and_mask_paths(image_dir, mask_dir)
        summary[split] = len(pairs)
        if dry_run:
            continue

        lines = []
        for image_path, mask_path in pairs:
            image_rel = image_path.relative_to(crack500_root).as_posix()
            if normalize_masks:
                with Image.open(image_path) as image_obj:
                    image_size = image_obj.size
                with Image.open(mask_path) as mask_obj:
                    raw_mask_size = mask_obj.size
                ensure_dir(normalized_mask_dir)
                normalized_mask_path = normalized_mask_dir / f"{mask_path.stem}.png"
                _binarize_mask(
                    mask_path,
                    normalized_mask_path,
                    threshold=normalize_threshold,
                    target_size=image_size,
                )
                if raw_mask_size != image_size:
                    summary["size_fixes"] += 1
                mask_rel = normalized_mask_path.relative_to(crack500_root).as_posix()
            else:
                mask_rel = mask_path.relative_to(crack500_root).as_posix()
            lines.append(f"{image_rel} {mask_rel}")
        (crack500_root / f"{split}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return summary


def _find_crackseg9k_image_dirs(crackseg9k_root):
    crackseg9k_root = Path(crackseg9k_root)
    image_dirs = [
        crackseg9k_root / "Final-Dataset-Vol1" / "Images",
        crackseg9k_root / "Final-Dataset-Vol2" / "Final-Dataset-Vol2" / "Images-2",
    ]
    missing = [path for path in image_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Crackseg9k image directories missing: {missing}")
    return image_dirs


def _find_crackseg9k_mask_dir(crackseg9k_root):
    mask_dir = Path(crackseg9k_root) / "Final-Dataset-Vol1" / "Final_Masks" / "Masks"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Crackseg9k mask directory does not exist: {mask_dir}")
    return mask_dir


def _read_name_list(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_filename_map(directories):
    mapping = {}
    for directory in directories:
        for path in sorted(Path(directory).iterdir()):
            if path.is_file():
                mapping[path.name] = path
    return mapping


def _write_split_file(dataset_root, split_name, pairs):
    lines = [
        f"{image_path.relative_to(dataset_root).as_posix()} {mask_path.relative_to(dataset_root).as_posix()}"
        for image_path, mask_path in pairs
    ]
    (Path(dataset_root) / f"{split_name}.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prepare_crackseg9k_lists(
    crackseg9k_root,
    dry_run=False,
    val_ratio=0.1,
    seed=42,
    normalize_masks=True,
    normalize_threshold=0,
    normalized_mask_dir_name="masks_bin",
):
    crackseg9k_root = Path(crackseg9k_root)
    if not crackseg9k_root.exists():
        raise FileNotFoundError(f"Crackseg9k root does not exist: {crackseg9k_root}")

    image_dirs = _find_crackseg9k_image_dirs(crackseg9k_root)
    mask_dir = _find_crackseg9k_mask_dir(crackseg9k_root)
    split_dir = crackseg9k_root / "Final-Dataset-Vol1" / "Final_Masks"
    train_names = _read_name_list(split_dir / "train.txt")
    test_names = _read_name_list(split_dir / "test.txt")
    if set(train_names) & set(test_names):
        raise ValueError("Crackseg9k train/test split files overlap.")

    image_map = _build_filename_map(image_dirs)
    mask_map = _build_filename_map([mask_dir])
    all_split_names = train_names + test_names
    missing_images = sorted(name for name in all_split_names if name not in image_map)
    missing_masks = sorted(name for name in all_split_names if name not in mask_map)
    if missing_images or missing_masks:
        raise ValueError(
            f"Crackseg9k split mismatch: missing_images={missing_images[:5]}, "
            f"missing_masks={missing_masks[:5]}"
        )

    shuffled_train_names = list(train_names)
    random.Random(seed).shuffle(shuffled_train_names)
    val_count = max(1, int(round(len(shuffled_train_names) * val_ratio)))
    if len(shuffled_train_names) - val_count < 1:
        raise ValueError("Crackseg9k val_ratio leaves no training samples.")
    val_names = sorted(shuffled_train_names[:val_count])
    final_train_names = sorted(shuffled_train_names[val_count:])
    final_test_names = sorted(test_names)

    normalized_mask_dir = crackseg9k_root / normalized_mask_dir_name
    summary = {
        "train": len(final_train_names),
        "val": len(val_names),
        "test": len(final_test_names),
        "source_train": len(train_names),
        "source_test": len(test_names),
        "normalized_masks": len(set(all_split_names)) if normalize_masks else 0,
        "mask_dir": str(mask_dir),
    }
    if dry_run:
        return summary

    mask_pairs = {}
    for name in sorted(set(all_split_names)):
        image_path = image_map[name]
        mask_path = mask_map[name]
        with Image.open(image_path) as image_obj:
            image_size = image_obj.size
        if normalize_masks:
            ensure_dir(normalized_mask_dir)
            normalized_mask_path = normalized_mask_dir / f"{Path(name).stem}.png"
            _binarize_mask(
                mask_path,
                normalized_mask_path,
                threshold=normalize_threshold,
                target_size=image_size,
            )
            target_mask_path = normalized_mask_path
        else:
            target_mask_path = mask_path
        mask_pairs[name] = (image_path, target_mask_path)

    _write_split_file(crackseg9k_root, "train", [mask_pairs[name] for name in final_train_names])
    _write_split_file(crackseg9k_root, "val", [mask_pairs[name] for name in val_names])
    _write_split_file(crackseg9k_root, "test", [mask_pairs[name] for name in final_test_names])
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare offline experiment datasets.")
    parser.add_argument("--custom_src", default="data", type=str)
    parser.add_argument("--custom_dst", default="dataset/custom", type=str)
    parser.add_argument("--crack500_root", default="dataset/Crack500", type=str)
    parser.add_argument("--crackseg9k_root", default="dataset/Crackseg9k", type=str)
    parser.add_argument("--prepare_custom", action="store_true")
    parser.add_argument("--prepare_crack500", action="store_true")
    parser.add_argument("--prepare_crackseg9k", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--normalize_crack500_masks", action="store_true", default=True)
    parser.add_argument("--no_normalize_crack500_masks", action="store_true")
    parser.add_argument("--normalize_crackseg9k_masks", action="store_true", default=True)
    parser.add_argument("--no_normalize_crackseg9k_masks", action="store_true")
    parser.add_argument("--normalize_threshold", default=127, type=int)
    parser.add_argument("--normalized_mask_dir_name", default="masks_bin", type=str)
    parser.add_argument("--crackseg9k_val_ratio", default=0.1, type=float)
    parser.add_argument("--crackseg9k_seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    no_explicit_target = not any((args.prepare_custom, args.prepare_crack500, args.prepare_crackseg9k))
    prepare_custom = args.prepare_custom or no_explicit_target
    prepare_crack500 = args.prepare_crack500 or no_explicit_target
    prepare_crackseg9k = args.prepare_crackseg9k or no_explicit_target

    if prepare_custom:
        summary = copy_custom_dataset(args.custom_src, args.custom_dst, dry_run=args.dry_run)
        print(f"Custom dataset summary: {summary}")
    if prepare_crack500:
        normalize_masks = args.normalize_crack500_masks and not args.no_normalize_crack500_masks
        summary = prepare_crack500_lists(
            args.crack500_root,
            dry_run=args.dry_run,
            normalize_masks=normalize_masks,
            normalize_threshold=args.normalize_threshold,
            normalized_mask_dir_name=args.normalized_mask_dir_name,
        )
        print(f"Crack500 summary: {summary}")
    if prepare_crackseg9k:
        normalize_masks = args.normalize_crackseg9k_masks and not args.no_normalize_crackseg9k_masks
        summary = prepare_crackseg9k_lists(
            args.crackseg9k_root,
            dry_run=args.dry_run,
            val_ratio=args.crackseg9k_val_ratio,
            seed=args.crackseg9k_seed,
            normalize_masks=normalize_masks,
            normalize_threshold=0,
            normalized_mask_dir_name=args.normalized_mask_dir_name,
        )
        print(f"Crackseg9k summary: {summary}")


if __name__ == "__main__":
    main()

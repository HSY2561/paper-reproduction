"""Read-only dataset checks; writes audit artifacts only beside this script.

This checks train/val/test files, not model predictions or test performance.
Public reference: https://arxiv.org/html/1901.06340 (section IV-B1).
"""
import datetime as dt
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import zipfile
import zlib

import numpy as np
from PIL import Image


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
ACTIVE = Path(r"C:\Users\user\algolab_data")
ORIGINAL = PROJECT / "data/Crack500_original/pavement crack datasets/CRACK500"
EXPECTED = {"train": 1896, "val": 348, "test": 1124}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def entries(path):
    return [tuple(line.replace("\\", "/").split()) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def review_shared_names():
    """Filename collisions are a flag, not proof that two source photos are identical."""
    names = {split: {Path(pair[0]).name for pair in entries(ACTIVE / f"{split}.txt")}
             for split in EXPECTED}
    output = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        rows = []
        for name in sorted(names[first] & names[second]):
            with Image.open(ACTIVE / f"{first}crop" / name) as image:
                a = np.asarray(image.convert("RGB"))
            with Image.open(ACTIVE / f"{second}crop" / name) as image:
                b = np.asarray(image.convert("RGB"))
            same_shape = a.shape == b.shape
            rows.append(dict(filename=name, same_shape=same_shape,
                             pixels_identical=bool(np.array_equal(a, b)),
                             mean_absolute_rgb_difference=float(np.abs(a.astype(np.int16) - b).mean())
                             if same_shape else None))
        output[f"{first}_{second}"] = dict(shared_filenames=len(rows),
            pixel_identical_pairs=sum(r["pixels_identical"] for r in rows), pairs=rows)
    return output


def main():
    issues, records, splits = [], {}, {}
    image_hashes, parents = {}, {}
    for split, expected in EXPECTED.items():
        pairs = entries(ACTIVE / f"{split}.txt")
        original_pairs = entries(ORIGINAL / f"{split}.txt")
        info = dict(expected_pairs=expected, listed_pairs=len(pairs),
                    unique_pairs=len(set(pairs)),
                    list_matches_original_order=pairs == original_pairs,
                    list_matches_original_set=set(pairs) == set(original_pairs),
                    image_files=len(list((ACTIVE / f"{split}crop").glob("*.jpg"))),
                    mask_files=len(list((ACTIVE / f"{split}crop").glob("*.png"))),
                    images_identical_to_original=0,
                    masks_byte_identical_to_original=0,
                    masks_foreground_identical_to_original=0,
                    source_files_matching_crop_zip_crc=0,
                    mirror_files_matching_active=0, decoded_pairs=0)
        if not (len(pairs) == len(set(pairs)) == expected
                == info["image_files"] == info["mask_files"]
                and info["list_matches_original_set"]):
            issues.append(dict(split=split, issue="count_or_original_list_mismatch"))
        listed = {p for pair in pairs for p in pair}
        actual = {p.relative_to(ACTIVE).as_posix() for p in
                  (ACTIVE / f"{split}crop").iterdir() if p.is_file()}
        if listed != actual:
            issues.append(dict(split=split, issue="list_vs_folder_mismatch",
                               missing=sorted(listed - actual), extra=sorted(actual - listed)))
        sizes, mask_values, original_values = Counter(), set(), set()
        image_hashes[split], parents[split] = set(), set()
        with zipfile.ZipFile(ORIGINAL / f"{split}crop.zip") as archive:
            zip_files = {Path(i.filename).name: i for i in archive.infolist()
                         if not i.is_dir() and Path(i.filename).suffix.lower() in (".jpg", ".png")}
            info["crop_zip_files"] = len(zip_files)
            if set(zip_files) != {Path(p).name for p in listed}:
                issues.append(dict(split=split, issue="zip_filename_set_mismatch"))
            for pair in pairs:
                if len(pair) != 2:
                    issues.append(dict(split=split, issue="malformed_pair", pair=pair))
                    continue
                image_rel, mask_rel = pair
                image_path, mask_path = ACTIVE / image_rel, ACTIVE / mask_rel
                if image_path.stem != mask_path.stem:
                    issues.append(dict(pair=pair, issue="image_mask_name_mismatch"))
                try:
                    active_bytes = [image_path.read_bytes(), mask_path.read_bytes()]
                    original_bytes = [(ORIGINAL / p).read_bytes() for p in pair]
                    current_sha = [digest(b) for b in active_bytes]
                    original_sha = [digest(b) for b in original_bytes]
                    for rel, data, original_data, sha, original_hash in zip(
                            pair, active_bytes, original_bytes, current_sha, original_sha):
                        records[rel] = dict(sha256=sha, original_sha256=original_hash)
                        entry = zip_files[Path(rel).name]
                        if (zlib.crc32(original_data) == entry.CRC
                                and len(original_data) == entry.file_size):
                            info["source_files_matching_crop_zip_crc"] += 1
                        else:
                            issues.append(dict(path=rel, issue="original_vs_crop_zip_crc_mismatch"))
                    with Image.open(io.BytesIO(active_bytes[0])) as image:
                        image.load()
                        image_size = image.size
                    with Image.open(io.BytesIO(active_bytes[1])) as label:
                        label.load()
                        label_size = label.size
                        label_arr = np.asarray(label)
                    with Image.open(io.BytesIO(original_bytes[1])) as label:
                        original_arr = np.asarray(label)
                    if image_size != label_size or label_arr.ndim != 2:
                        issues.append(dict(path=mask_rel, issue="image_mask_shape_mismatch"))
                    sizes[f"{image_size[0]}x{image_size[1]}"] += 1
                    mask_values.update(int(v) for v in np.unique(label_arr))
                    original_values.update(int(v) for v in np.unique(original_arr))
                    info["decoded_pairs"] += 1
                    same_image = current_sha[0] == original_sha[0]
                    info["images_identical_to_original"] += int(same_image)
                    info["masks_byte_identical_to_original"] += int(current_sha[1] == original_sha[1])
                    same_foreground = np.array_equal(label_arr > 0, original_arr > 0)
                    info["masks_foreground_identical_to_original"] += int(same_foreground)
                    if not same_image or not same_foreground:
                        issues.append(dict(path=image_rel, issue="active_vs_original_content_mismatch"))
                    for rel, suffix, sha in zip(pair, ("img", "lab"), current_sha):
                        mirror = ACTIVE / f"{split}_{suffix}" / Path(rel).name
                        if mirror.exists() and digest(mirror.read_bytes()) == sha:
                            info["mirror_files_matching_active"] += 1
                        else:
                            issues.append(dict(path=str(mirror), issue="mirror_content_mismatch"))
                    image_hashes[split].add(current_sha[0])
                    parents[split].add(image_path.stem.rsplit("_", 2)[0])
                except Exception as error:
                    issues.append(dict(pair=pair, issue="file_read_or_decode_error", error=str(error)))
        info.update(image_sizes=dict(sizes), active_mask_values=sorted(mask_values),
                    original_mask_values=sorted(original_values),
                    unique_image_hashes=len(image_hashes[split]),
                    unique_filename_parent_ids=len(parents[split]))
        if not mask_values.issubset({0, 1}):
            issues.append(dict(split=split, issue="active_labels_not_0_1", values=sorted(mask_values)))
        splits[split] = info
        print(json.dumps(dict(split=split, checked=info["decoded_pairs"],
                              issues_so_far=len(issues))), flush=True)
    cross_split = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap_hashes = sorted(image_hashes[first] & image_hashes[second])
        overlap_parents = sorted(parents[first] & parents[second])
        cross_split[f"{first}_{second}"] = dict(
            identical_image_sha256_count=len(overlap_hashes),
            filename_parent_overlap_count=len(overlap_parents),
            filename_parent_overlap=overlap_parents)
    raw_archives = {}
    for name in ("testdata.zip", "valdata.zip"):
        with zipfile.ZipFile(ORIGINAL / name) as archive:
            raw_archives[name] = dict(Counter(Path(i.filename).suffix.lower()
                                     for i in archive.infolist() if not i.is_dir()))
    report = dict(
        checked_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        active_dataset_root=str(ACTIVE), local_original_package_root=str(ORIGINAL),
        official_repository="https://github.com/fyangneil/pavement-crack-detection",
        primary_paper="https://arxiv.org/html/1901.06340", splits=splits,
        cross_split=cross_split, shared_filename_review=review_shared_names(),
        local_raw_archive_file_extensions=raw_archives,
        file_integrity_issues=issues, integrity_issue_count=len(issues),
        total_pairs=sum(s["listed_pairs"] for s in splits.values()),
        test_access="File integrity audit only; no model inference, metrics or selection.",
        scope="Counts compared to primary paper; bytes/foreground compared to locally retained public package and its crop ZIP CRC metadata. No fresh remote archive checksum comparison.")
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "file_sha256.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(total_pairs=report["total_pairs"],
                          integrity_issue_count=len(issues), cross_split=cross_split,
                          output=str(OUT / "audit.json")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

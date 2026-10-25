"""Build the pinned, minimal PaddleSeg runtime from an offline upstream ZIP.

No model, dataset, or Python dependency is downloaded or executed here.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.10.0"
ARCHIVE_SHA256 = "8ad8541e3ebb7fe9fc670f4026972be77385e73764f66e94390297e794c268d6"
ARCHIVE_URL = "https://github.com/PaddlePaddle/PaddleSeg/archive/refs/tags/v2.10.0.zip"
PREFIX = "PaddleSeg-2.10.0/"
OVERLAY_SHA256 = "ad292bca4d0171028360ec00bbd1f7b87214c03a494ed2d13a02871e9c57d362"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
TOP_FILES = {"LICENSE", "requirements.txt", "tools/train.py", "tools/val.py"}
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(info):
    """Return a safe relative path; reject platform-specific escapes."""
    name = info.filename
    original = info.orig_filename  # ZipInfo may normalize backslashes or truncate at NUL.
    if not name.startswith(PREFIX) or "\\" in original or "\x00" in original:
        raise ValueError(f"Unexpected archive member: {name!r}")
    relative = name[len(PREFIX):].rstrip("/")
    if not relative:
        return None
    parts = relative.split("/")
    if any(part in ("", ".", "..") or ":" in part
           or part.endswith((" ", "."))
           or part.split(".")[0].upper() in RESERVED for part in parts):
        raise ValueError(f"Unsafe archive member: {name!r}")
    if PurePosixPath(relative).is_absolute():
        raise ValueError(f"Absolute archive member: {name!r}")
    return PurePosixPath(relative)


@contextmanager
def staging_directory(parent):
    # On Windows, tempfile's owner-only ACL can exclude a sandbox token.
    # A fresh ordinary child directory inherits the caller's usable ACL.
    temporary = parent / (".paddleseg-bootstrap-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        yield temporary
    except BaseException:
        print(f"Incomplete staging directory retained for inspection: {temporary}", file=sys.stderr)
        raise
    else:
        temporary.rmdir()  # Only the now-empty directory; never recursively delete user files.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path,
                        help="Previously downloaded official v2.10.0 ZIP; SHA must match.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runtime" / "PaddleSeg-2.10.0")
    args = parser.parse_args()
    if sha256(args.archive) != ARCHIVE_SHA256:
        parser.error("Archive SHA256 mismatch. Nothing was extracted.")
    overlay = ROOT / "overlays" / "paddleseg" / "models" / "segformer.py"
    if sha256(overlay) != OVERLAY_SHA256:
        parser.error("The committed SegFormer overlay does not match its pinned SHA256.")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("Output already exists; choose a fresh directory. Nothing is overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with staging_directory(output.parent) as temporary:
        staged = temporary / "PaddleSeg-2.10.0"
        staged.mkdir()
        extracted = []
        total = 0
        seen = set()
        with zipfile.ZipFile(args.archive) as archive:
            for info in archive.infolist():
                relative = safe_member(info)
                if relative is None or info.is_dir():
                    continue
                name = relative.as_posix()
                if not (name.startswith("paddleseg/") or name in TOP_FILES):
                    continue
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError(f"Non-regular selected archive member: {name!r}")
                if name.casefold() in seen:
                    raise ValueError(f"Duplicate archive destination: {name}")
                seen.add(name.casefold())
                total += info.file_size
                if info.file_size > 32 * 1024 * 1024 or total > 256 * 1024 * 1024:
                    raise ValueError("Selected archive content exceeds the extraction limit.")
                destination = staged.joinpath(*relative.parts)
                if not destination.resolve().is_relative_to(staged.resolve()):
                    raise ValueError("Archive path escapes the staging directory.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
                extracted.append(name)
        required = TOP_FILES | {"paddleseg/__init__.py", "paddleseg/models/segformer.py"}
        if not required.issubset(extracted):
            raise ValueError("The archive does not contain the required upstream files.")
        if sha256(staged / "LICENSE") != LICENSE_SHA256:
            raise ValueError("Unexpected upstream license checksum.")
        shutil.copyfile(overlay, staged / "paddleseg" / "models" / "segformer.py")
        provenance = {
            "upstream": "PaddlePaddle/PaddleSeg", "tag": "v" + VERSION,
            "archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
            "license_sha256": LICENSE_SHA256,
            "overlay": "paddleseg/models/segformer.py", "overlay_sha256": OVERLAY_SHA256,
            "extracted_file_count": len(extracted),
            "scope": "paddleseg package, official train/val entry points, requirements, LICENSE",
            "notice": "See third_party/NOTICE.md, including the upstream NVlabs/SegFormer notice.",
        }
        (staged / "BOOTSTRAP_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    print(json.dumps({"runtime": str(output), "files": len(extracted),
                      "archive_sha256": ARCHIVE_SHA256, "overlay_sha256": OVERLAY_SHA256}, indent=2))


if __name__ == "__main__":
    main()

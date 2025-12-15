#!/usr/bin/env python3
"""
Local validation script for DATA/batch_* samples.
Checks each sample directory has the expected number of files (default: 6)
and that each image file can be opened and verified.

Usage (from repo root):
    python src/tools/local_validate_data.py --root DATA --expected 6

If Pillow is not installed, install once:
    pip install pillow
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from PIL import Image
except ImportError:  # Pillow missing locally
    Image = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:  # tqdm missing locally
    tqdm = None  # type: ignore


def find_sample_dirs(root: Path) -> Iterable[Path]:
    for batch_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("batch_")):
        for sample_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
            yield sample_dir


def is_image_valid(path: Path) -> bool:
    if Image is None:
        return True  # Skip image validation if Pillow isn't available
    try:
        with Image.open(path) as img:
            img.verify()  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def validate_sample(sample_dir: Path, expected: int) -> Tuple[bool, List[Path], List[Path]]:
    files = sorted(p for p in sample_dir.iterdir() if p.is_file())
    missing_or_extra = []
    bad_images: List[Path] = []

    if len(files) != expected:
        missing_or_extra.append(sample_dir)

    for f in files:
        if not is_image_valid(f):
            bad_images.append(f)

    ok = len(files) == expected and not bad_images
    return ok, missing_or_extra, bad_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local DATA batches")
    parser.add_argument("--root", default="DATA", help="Root directory containing batch_* folders")
    parser.add_argument("--expected", type=int, default=6, help="Expected number of files per sample")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker threads")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 1

    total = 0
    ok_samples = 0
    missing_or_extra: List[Path] = []
    bad_images: List[Path] = []

    sample_dirs = list(find_sample_dirs(root))
    total = len(sample_dirs)

    if total == 0:
        print(f"No samples found under {root}")
        return 1

    iterator = None
    pbar = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(validate_sample, sd, args.expected) for sd in sample_dirs]
        iterator = concurrent.futures.as_completed(futures)
        if tqdm:
            pbar = tqdm(total=total, desc="Samples", unit="sample")

        for fut in iterator:
            ok, missing_list, bad_list = fut.result()
            if ok:
                ok_samples += 1
            else:
                missing_or_extra.extend(missing_list)
                bad_images.extend(bad_list)
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    print("========== Validation Summary ==========")
    print(f"Root: {root}")
    print(f"Total sample folders: {total}")
    print(f"OK samples: {ok_samples}")
    print(f"Missing/extra files: {len(missing_or_extra)}")
    print(f"Bad images: {len(bad_images)}")

    if missing_or_extra:
        print("\nSamples with missing/extra files (first 50):")
        for p in missing_or_extra[:50]:
            print(f"  {p}")
        if len(missing_or_extra) > 50:
            print(f"  ...and {len(missing_or_extra) - 50} more")

    if bad_images:
        print("\nImages that failed to open/verify (first 50):")
        for p in bad_images[:50]:
            print(f"  {p}")
        if len(bad_images) > 50:
            print(f"  ...and {len(bad_images) - 50} more")

    if Image is None:
        print("\nNote: Pillow not installed; image validity checks were skipped.")
    if tqdm is None:
        print("Note: tqdm not installed; progress bar was not shown.")

    return 0 if ok_samples == total and not bad_images else 2


if __name__ == "__main__":
    raise SystemExit(main())

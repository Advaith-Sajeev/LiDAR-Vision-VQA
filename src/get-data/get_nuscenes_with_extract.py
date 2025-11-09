#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import tarfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

# -------- Settings (defaults can be overridden via CLI) --------
BUCKET = "motional-nuscenes"
REGION = "ap-northeast-1"
PREFIX = "public/v1.0/"  # nuScenes objects live under this prefix

TRAIN_FILES = ["v1.0-trainval_meta.tgz"] + [f"v1.0-trainval{i:02d}_blobs.tgz" for i in range(1, 11)]
TEST_FILES  = ["v1.0-test_meta.tgz", "v1.0-test_blobs.tgz"]


def make_s3_client(use_https_only: bool):
    """
    Create an unsigned S3 client in the correct region.
    - Path-style addressing helps on some HPC networks.
    - use_https_only=True switches to s3.<region>.amazonaws.com (still S3 API).
    """
    cfg = Config(
        signature_version=UNSIGNED,
        region_name=REGION,
        s3={"addressing_style": "path"},
        retries={"max_attempts": 10, "mode": "standard"},
    )
    endpoint = f"https://s3.{REGION}.amazonaws.com" if use_https_only else None
    return boto3.client("s3", config=cfg, endpoint_url=endpoint)


def head_size(s3, key: str) -> int | None:
    try:
        resp = s3.head_object(Bucket=BUCKET, Key=key)
        return int(resp["ContentLength"])
    except ClientError:
        return None


def download_one(s3, key: str, dst: Path, global_bar: tqdm | None = None, total_bytes: int | None = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    remote_size = total_bytes if total_bytes is not None else head_size(s3, key)

    # Skip if fully downloaded
    if dst.exists() and remote_size is not None and dst.stat().st_size == remote_size:
        tqdm.write(f"✓ Skipping (already complete): {dst.name}")
        if global_bar is not None:
            global_bar.update(remote_size)  # count it in total
        return

    # Per-file progress bar
    with tqdm(
        total=remote_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dst.name,
        leave=True,
    ) as pbar:

        def _cb(bytes_amount):
            pbar.update(bytes_amount)
            if global_bar is not None:
                global_bar.update(bytes_amount)

        s3.download_file(Bucket=BUCKET, Key=key, Filename=str(dst), Callback=_cb)


def extract_safe(tar_path: Path, dest_dir: Path):
    """
    Extract tar.gz safely with a progress bar and path traversal guard.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    def is_within_directory(directory: Path, target: Path) -> bool:
        try:
            directory = directory.resolve()
            target = target.resolve()
        except Exception:
            return False
        return str(target).startswith(str(directory))

    with tarfile.open(tar_path, mode="r:gz") as tf:
        members = tf.getmembers()
        # Use only files' sizes for progress total
        total_bytes = sum(m.size for m in members if m.isfile())
        with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=f"Extract {tar_path.name}", leave=True) as pbar:
            for m in members:
                target_path = dest_dir / m.name
                # Guard against tarbomb/path traversal
                if not is_within_directory(dest_dir, target_path.parent):
                    raise RuntimeError(f"Unsafe path in tar: {m.name}")
                # Extract member
                tf.extract(m, path=dest_dir)
                if m.isfile():
                    pbar.update(m.size)


def main():
    ap = argparse.ArgumentParser(description="Download nuScenes blobs with progress and extract to Dataset/{train,test}.")
    ap.add_argument("--download-dir", default="Tarballs", help="Where to store .tgz files (default: Tarballs)")
    ap.add_argument("--dataset-dir", default="Dataset", help="Where to extract (default: Dataset)")
    ap.add_argument("--only", choices=["train", "test", "both"], default="both", help="What to fetch/extract (default: both)")
    ap.add_argument("--https-only", action="store_true",
                    help="Use S3 HTTPS endpoint explicitly (can help on restricted HPC networks).")
    ap.add_argument("--keep-tars", action="store_true", help="Keep .tgz files after extraction.")
    args = ap.parse_args()

    download_dir = Path(args.download_dir)
    dataset_dir = Path(args.dataset_dir)
    train_tar_dir = download_dir / "train"
    test_tar_dir = download_dir / "test"
    train_out = dataset_dir / "train"
    test_out = dataset_dir / "test"

    s3 = make_s3_client(use_https_only=args.https_only)

    # Build key->destination map
    keys = []
    if args.only in ("train", "both"):
        keys += [(PREFIX + f, train_tar_dir / f) for f in TRAIN_FILES]
    if args.only in ("test", "both"):
        keys += [(PREFIX + f, test_tar_dir / f) for f in TEST_FILES]

    # Compute overall total for nice global progress (ignore HEAD failures)
    sizes = []
    for key, _dst in keys:
        sizes.append(head_size(s3, key))
    total_sum = sum(sz for sz in sizes if isinstance(sz, int))
    size_map = {k: sz for (k, _), sz in zip(keys, sizes)}

    # Download with overall progress bar
    with tqdm(total=(total_sum if total_sum > 0 else None),
              unit="B", unit_scale=True, unit_divisor=1024,
              desc="TOTAL DOWNLOAD", leave=True) as gbar:
        for key, dst in keys:
            download_one(s3, key, dst, global_bar=gbar, total_bytes=size_map.get(key))

    # Extract to Dataset/train and Dataset/test
    if args.only in ("train", "both"):
        for tgz in sorted(train_tar_dir.glob("*.tgz")):
            extract_safe(tgz, train_out)
    if args.only in ("test", "both"):
        for tgz in sorted(test_tar_dir.glob("*.tgz")):
            extract_safe(tgz, test_out)

    # Optionally delete tarballs
    if not args.keep_tars:
        for tgz in download_dir.rglob("*.tgz"):
            try:
                tgz.unlink()
            except Exception as e:
                tqdm.write(f"Warning: could not delete {tgz}: {e}")

    print("\n✅ Done.")
    print(f"Extracted to: {train_out if args.only!='test' else '(train skipped)'} and {test_out if args.only!='train' else '(test skipped)'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

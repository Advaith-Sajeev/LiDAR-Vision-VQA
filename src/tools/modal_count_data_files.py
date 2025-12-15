#!/usr/bin/env python3
"""
Modal helper to count files inside the attached DATA volume quickly.

Run:
    modal run src/tools/modal_count_data_files.py

Optional args (local entrypoint):
    modal run src/tools/modal_count_data_files.py --root /data/DATA
"""

import os
import time
from typing import Optional

import modal

app = modal.App("lidar-vision-count-files")
volume = modal.Volume.from_name("lidar-llm", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tqdm")
)


def _count_files(root: str) -> tuple[int, float]:
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        tqdm = None  # Fallback to no progress if not installed locally

    start = time.time()
    total = 0
    stack = [root]
    pbar = None
    if tqdm:
        pbar = tqdm(total=0, unit="file", desc=f"Counting {root}", mininterval=1.0, dynamic_ncols=True)
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                pending = 0
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += 1
                        pending += 1
                        if pbar and pending >= 500:
                            pbar.update(pending)
                            pending = 0
                if pbar and pending:
                    pbar.update(pending)
        except FileNotFoundError:
            continue
    if pbar:
        pbar.close()
    elapsed = time.time() - start
    return total, elapsed


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=2.0,
    memory=2048,
    timeout=900,
)
def count_files(root: Optional[str] = None):
    target = root or "/data/DATA"
    total, elapsed = _count_files(target)
    print(f"Total files under {target}: {total}")
    print(f"Elapsed: {elapsed:.2f}s")
    return {"root": target, "files": total, "seconds": round(elapsed, 2)}


@app.local_entrypoint()
def main(root: str = "/data/DATA"):
    # If the root contains batch_* dirs, iterate each batch separately and aggregate.
    batches = []
    if os.path.basename(root.rstrip("/")) == "DATA":
        try:
            for entry in os.scandir(root):
                if entry.is_dir() and entry.name.startswith("batch_"):
                    batches.append(entry.path)
        except FileNotFoundError:
            pass

    summary = {}
    total_files = 0
    total_seconds = 0.0

    if batches:
        print(f"Found {len(batches)} batches under {root}. Counting batch by batch...")
        for batch_path in tqdm(sorted(batches), desc="Batches", unit="batch"):
            result = count_files.remote(batch_path)
            summary[batch_path] = result
            total_files += result.get("files", 0)
            total_seconds += result.get("seconds", 0.0)
        print("\nPer-batch counts:")
        for b, res in summary.items():
            print(f"{b}: {res['files']} files (took {res['seconds']}s)")
        print("\nAggregate:")
        print(f"Total files: {total_files}")
        print(f"Total time (sum of batch times): {total_seconds:.2f}s")
        return {"batches": summary, "total_files": total_files, "total_seconds": round(total_seconds, 2)}
    else:
        result = count_files.remote(root)
        print(result)
        return result


if __name__ == "__main__":
    print("Run with: modal run src/tools/modal_count_data_files.py [--root /data/DATA]")

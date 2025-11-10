"""Data utility functions"""

import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_json_any(path: str) -> Iterable[Dict]:
    """Load JSON from file or JSONL format."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for r in json.load(f):
                yield r
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def collect_feature_tokens(feature_dirs: List[str]) -> Dict[str, str]:
    """
    Collect mapping of sample tokens to feature file paths.
    
    Args:
        feature_dirs: List of directories containing .npy feature files
        
    Returns:
        Dictionary mapping sample_token to feature file path
    """
    import os
    
    def is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0
    
    token2path = {}
    for root in feature_dirs:
        r = Path(root)
        if not r.is_dir():
            if is_main_process():
                print(f"[warn] feature root missing: {root}")
            continue
        # Use recursive glob to find .npy files in subdirectories (e.g., train/, val/)
        for npy in r.glob("**/*.npy"):
            token2path.setdefault(npy.stem, str(npy))
    return token2path

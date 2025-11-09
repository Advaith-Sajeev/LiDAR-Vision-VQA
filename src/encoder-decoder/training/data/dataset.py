"""Dataset class for mixed nuScenes data"""

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional

from .utils import load_json_any, collect_feature_tokens


class MixedNuDataset(Dataset):
    """
    Dataset for nuScenes with BEV features and QA pairs.
    
    Returns items with:
      - token: str (nuScenes sample_token)
      - bev:   Tensor [C,H,W]   (loaded from <feature>.npy)
      - question / answer strings
    """
    
    def __init__(
        self,
        json_paths: List[str],
        feature_dirs: List[str],
        target_field: str = "answer_lidar",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.target_field = target_field
        self.token2path = collect_feature_tokens(feature_dirs)
        
        from ..utils.distributed import is_main_process
        
        if is_main_process():
            print("[features] scanning roots...")
            print(f"[features] unique tokens indexed: {len(self.token2path)}")

        rows = []
        total = 0
        no_feature = 0
        no_qa = 0
        rng = random.Random(seed)
        
        for jp in json_paths:
            jp_name = Path(jp).stem  # Extract filename for source tracking
            for r in load_json_any(jp):
                total += 1
                tok = r.get("sample_token")
                if not tok or tok not in self.token2path:
                    no_feature += 1
                    continue
                    
                ans = (r.get(self.target_field) or "").strip()
                if not ans:
                    ans = (r.get("answer") or "").strip()
                    if not ans:
                        no_qa += 1
                        continue
                    r[self.target_field] = ans
                    
                q = (r.get("question") or "").strip()
                r["question"] = q
                r["dataset_source"] = jp_name  # Add source tracking
                rows.append(r)

        if max_samples is not None and len(rows) > max_samples:
            rng.shuffle(rows)
            rows = rows[:max_samples]

        self.rows = rows
        
        if is_main_process():
            print(f"[dataset] total={total}  kept={len(self.rows)}  no_feature/qa={no_feature}/{no_qa}")
            
        if not self.rows:
            raise RuntimeError("No usable rows; check feature dirs and jsons.")

    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx):
        r = self.rows[idx]
        tok = r["sample_token"]
        bev = np.load(self.token2path[tok])  # [C,H,W]
        return {
            "token": tok,
            "bev": torch.from_numpy(bev).float(),
            "question": r.get("question", ""),
            "answer": r.get(self.target_field, "")
        }

"""Dataset class for mixed nuScenes data"""

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional

from .utils import load_json_any, collect_feature_tokens


# Import debug logger
try:
    from ..utils import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False


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
        if DEBUG_AVAILABLE:
            debug.info("dataset", "Initializing MixedNuDataset")
            debug.debug("dataset", f"JSON paths: {json_paths}")
            debug.debug("dataset", f"Feature dirs: {feature_dirs}")
            debug.debug("dataset", f"Target field: {target_field}")
            debug.debug("dataset", f"Max samples: {max_samples}")
        
        self.target_field = target_field
        
        if DEBUG_AVAILABLE:
            debug.data_flow("dataset", "feature_indexing", "Scanning feature directories")
        
        self.token2path = collect_feature_tokens(feature_dirs)
        
        from ..utils.distributed import is_main_process
        
        if is_main_process():
            print("[features] scanning roots...")
            print(f"[features] unique tokens indexed: {len(self.token2path)}")
            if DEBUG_AVAILABLE:
                debug.info("dataset", f"Indexed {len(self.token2path)} BEV feature files")

        rows = []
        total = 0
        no_feature = 0
        no_qa = 0
        filtered_grounding = 0  # Track filtered grounding samples
        rng = random.Random(seed)
        
        if DEBUG_AVAILABLE:
            debug.data_flow("dataset", "json_loading", f"Loading from {len(json_paths)} JSON files")
        
        for jp in json_paths:
            jp_name = Path(jp).stem  # Extract filename for source tracking
            if DEBUG_AVAILABLE:
                debug.debug("dataset", f"Loading: {jp_name}")
            
            # Check if this is nuGrounding dataset
            is_grounding = "grounding" in jp_name.lower()
            
            for r in load_json_any(jp):
                total += 1
                tok = r.get("sample_token")
                if not tok or tok not in self.token2path:
                    no_feature += 1
                    continue
                
                # Filter nuGrounding: only keep det_area template_type
                # This prevents data leakage from det_object which contains coordinates in questions
                if is_grounding:
                    template_type = r.get("template_type", "")
                    if template_type != "det_area":
                        filtered_grounding += 1
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
            if DEBUG_AVAILABLE:
                debug.debug("dataset", f"Sampling {max_samples} from {len(rows)} rows")
            rng.shuffle(rows)
            rows = rows[:max_samples]

        self.rows = rows
        
        if is_main_process():
            print(f"[dataset] total={total}  kept={len(self.rows)}  no_feature/qa={no_feature}/{no_qa}")
            if filtered_grounding > 0:
                print(f"[dataset] filtered {filtered_grounding} nuGrounding samples (kept only det_area, removed det_object to prevent data leakage)")
            if DEBUG_AVAILABLE:
                debug.info("dataset", f"Dataset ready: {len(self.rows)} samples")
                debug.debug("dataset", f"Dropped: no_feature={no_feature}, no_qa={no_qa}, filtered_grounding={filtered_grounding}")
            
        if not self.rows:
            raise RuntimeError("No usable rows; check feature dirs and jsons.")

    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx):
        if DEBUG_AVAILABLE and debug.get_debug_level() >= 3:  # TRACE level
            debug.trace("dataset", f"Loading sample {idx}")
        
        r = self.rows[idx]
        tok = r["sample_token"]
        
        if DEBUG_AVAILABLE and debug.get_debug_level() >= 3:
            debug.trace("dataset", f"Sample token: {tok}")
        
        bev = np.load(self.token2path[tok])  # [C,H,W]
        
        if DEBUG_AVAILABLE and debug.get_debug_level() >= 3:
            debug.shape("dataset", f"bev_{idx}", bev)
        
        return {
            "token": tok,
            "bev": torch.from_numpy(bev).float(),
            "question": r.get("question", ""),
            "answer": r.get(self.target_field, "")
        }

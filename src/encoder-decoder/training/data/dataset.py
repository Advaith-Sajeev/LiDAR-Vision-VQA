"""Dataset class for mixed nuScenes data"""

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence

from .utils import load_json_any, collect_feature_tokens


# Import debug logger
try:
    from ..utils import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False


# Default camera view order for nuScenes
DEFAULT_VIEW_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


class MixedNuDataset(Dataset):
    """
    Dataset for nuScenes with BEV features, camera images, and QA pairs.
    
    Returns items with:
      - token: str (nuScenes sample_token)
      - bev:   Tensor [C,H,W]   (loaded from <feature>.npy)
      - images: List[Optional[Tensor]] (6 camera views, each [1,3,1024,1024] or None)
      - question / answer strings
    """
    
    def __init__(
        self,
        json_paths: List[str],
        feature_dirs: List[str],
        target_field: str = "answer_lidar",
        max_samples: Optional[int] = None,
        seed: int = 42,
        nusc=None,  # NuScenes object for image loading
        load_images: bool = False,  # Whether to load camera images
        view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
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
                
                # Keep both det_area and det_object for training
                # det_object samples are needed for bbox evaluation metrics
                # No filtering needed - both types help the model learn
                    
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
        self.nusc = nusc
        self.load_images = load_images
        self.view_order = view_order
        
        # Import image loading helper if needed
        if self.load_images:
            from deepencoder import load_and_preprocess_image, resolve_cam_image_paths
            self._load_and_preprocess_image = load_and_preprocess_image
            self._resolve_cam_image_paths = resolve_cam_image_paths
        
        if is_main_process():
            print(f"[dataset] total={total}  kept={len(self.rows)}  no_feature/qa={no_feature}/{no_qa}")
            if self.load_images:
                print(f"[dataset] Image loading enabled (workers will load {len(self.view_order)} views per sample)")
            if DEBUG_AVAILABLE:
                debug.info("dataset", f"Dataset ready: {len(self.rows)} samples")
                debug.debug("dataset", f"Dropped: no_feature={no_feature}, no_qa={no_qa}")
            
        if not self.rows:
            raise RuntimeError("No usable rows; check feature dirs and jsons.")

    def __len__(self):
        return len(self.rows)
    
    def _load_camera_images(self, sample_token: str) -> List[Optional[torch.Tensor]]:
        """Load and preprocess camera images for a sample (runs in DataLoader workers)."""
        if self.nusc is None:
            return [None] * len(self.view_order)
        
        # Resolve image paths
        image_paths = self._resolve_cam_image_paths(self.nusc, sample_token, self.view_order)
        
        # Load and preprocess each image
        images = []
        for path in image_paths:
            img_tensor = self._load_and_preprocess_image(path)
            images.append(img_tensor)
        
        return images
        
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
        
        result = {
            "token": tok,
            "bev": torch.from_numpy(bev).float(),
            "question": r.get("question", ""),
            "answer": r.get(self.target_field, "")
        }
        
        # Load camera images if enabled (runs in worker process)
        if self.load_images:
            result["images"] = self._load_camera_images(tok)
        
        return result

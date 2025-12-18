"""Dataset class for mixed nuScenes data"""

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence

from .utils import (
    load_json_any, 
    validate_json_schema as _validate_json_schema,
    validate_image_paths as _validate_image_paths,
)


# Import debug logger
try:
    from ..utils import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False


# Import camera view config from centralized config
try:
    from configs.constants import DEFAULT_VIEW_ORDER
except ImportError:
    # Fallback if configs not in path
    DEFAULT_VIEW_ORDER = (
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
        "CAM_BACK_LEFT",
    )


class VisionNuDataset(Dataset):
    """
    Dataset for nuScenes with camera images and QA pairs.
    
    Returns items with:
      - token: str (nuScenes sample_token)
      - images: List[Optional[Tensor]] (6 camera views, each [1,3,1024,1024] or None)
      - question / answer strings
    """
    
    def __init__(
        self,
        json_paths: List[str],
        target_field: str = "answer",
        max_samples: Optional[int] = None,
        seed: int = 42,
        nusc=None,  # NuScenes object for image loading
        load_images: bool = False,  # Whether to load camera images
        view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
        # Validation settings (can be passed from config)
        validate_json_schema: bool = True,
        validate_image_paths: bool = True,
    ):
        if DEBUG_AVAILABLE:
            debug.info("dataset", "Initializing MixedNuDataset")
            debug.debug("dataset", f"JSON paths: {json_paths}")
            debug.debug("dataset", f"Target field: {target_field}")
            debug.debug("dataset", f"Max samples: {max_samples}")
        
        self.target_field = target_field
        
        from ..utils.distributed import is_main_process
        
        # =====================================================================
        # Phase 2: JSON Schema Validation
        # =====================================================================
        if validate_json_schema:
            if DEBUG_AVAILABLE:
                debug.data_flow("dataset", "json_validation", "Validating JSON schema")
            json_result = _validate_json_schema(
                json_paths,
                required_fields=("sample_token", "question"),
                answer_fields=("answer",),
                token2path=None,  
            )
            if json_result['issues']:
                if is_main_process():
                    for issue in json_result['issues'][:5]:
                        print(f"[JSON validation] WARNING: {issue}")

        rows = []
        total = 0
        no_qa = 0
        rng = random.Random(seed)
        
        if DEBUG_AVAILABLE:
            debug.data_flow("dataset", "json_loading", f"Loading from {len(json_paths)} JSON files")
        
        for jp in json_paths:
            jp_name = Path(jp).stem  # Extract filename for source tracking
            if DEBUG_AVAILABLE:
                debug.debug("dataset", f"Loading: {jp_name}")
            
            for r in load_json_any(jp):
                total += 1
                tok = r.get("sample_token")
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

        # Apply max_samples limit if specified
        total_before_limit = len(rows)
        if max_samples is not None and len(rows) > max_samples:
            if DEBUG_AVAILABLE:
                debug.debug("dataset", f"Sampling {max_samples} from {len(rows)} rows")
            rng.shuffle(rows)
            rows = rows[:max_samples]
            if is_main_process():
                print(f"[dataset] ⚠️  max_samples={max_samples} applied: using {len(rows)} of {total_before_limit} available samples")

        self.rows = rows
        self.nusc = nusc
        self.load_images = load_images
        self.view_order = view_order
        
        # Import image loading helper if needed
        if self.load_images:
            from deepencoder import load_and_preprocess_image, resolve_cam_image_paths
            self._load_and_preprocess_image = load_and_preprocess_image
            self._resolve_cam_image_paths = resolve_cam_image_paths
        
        # =====================================================================
        # Phase 5: Camera Image Path Validation (when loading images)
        # =====================================================================
        if validate_image_paths and self.load_images and nusc is not None and self.rows:
            if DEBUG_AVAILABLE:
                debug.data_flow("dataset", "image_path_validation", "Validating camera image paths")
            sample_tokens = [r["sample_token"] for r in self.rows]
            image_validation = _validate_image_paths(
                nusc=nusc,
                sample_tokens=sample_tokens,
                view_order=self.view_order,
                 num_workers=0, # Use main process for simplicity or add argument if needed
                # max_samples=None checks ALL samples (entire dataset)
            )
            if image_validation.get('tokens_with_missing', 0) > 0:
                if is_main_process():
                    print(f"[Image validation] ⚠️  {image_validation['tokens_with_missing']} samples have missing camera views")
        
        if is_main_process():
            print(f"[dataset] total={total}  kept={len(self.rows)}  no_qa={no_qa}")
            if self.load_images:
                print(f"[dataset] Image loading enabled (workers will load {len(self.view_order)} views per sample)")
            if DEBUG_AVAILABLE:
                debug.info("dataset", f"Dataset ready: {len(self.rows)} samples")
                debug.debug("dataset", f"Dropped: no_qa={no_qa}")
            
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
        
        result = {
            "token": tok,
            "question": r.get("question", ""),
            "answer": r.get(self.target_field, "")
        }
        
        # Load camera images if enabled (runs in worker process)
        if self.load_images:
            result["images"] = self._load_camera_images(tok)
        
        return result

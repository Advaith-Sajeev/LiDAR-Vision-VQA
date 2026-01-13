"""Main trainer class"""

import sys
import math
import random
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict

from ..data import VisionNuDataset, make_collate, SingleProcessDetSampler
from ..utils import (
    init_dist_if_needed,
    is_main_process,
    Tee,
    set_seed,
    count_trainable_params,
    save_state,
    try_load_state,
    plot_loss_curve,
    plot_all_metrics,
    debug,
    set_debug_mode,
    set_log_file,
    DEBUG_INFO,
    DEBUG_DEBUG,
    DEBUG_TRACE,
)
from ..utils.sequence_builder import build_training_sequence, ModalityPosition
from .model_setup import setup_models, setup_optimizer_and_scheduler
from .validation import run_validation, run_inference_sampling
from deepencoder.deepencoder_infer import multiview_tokens_from_sample_token, batch_multiview_tokens_from_sample_tokens
from configs.constants import (
    DEFAULT_VIEW_ORDER,
    TOKENS_PER_VIEW,
    PROJECTOR_DIM,
    NUM_VIEWS,
    validate_config,
)


class Trainer:
    """Main training orchestrator"""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        # Initialize debug mode from config
        debug_mode = config.get("debug_mode", False)
        debug_level = config.get("debug_level", DEBUG_DEBUG)
        if debug_mode:
            set_debug_mode(True, debug_level)
            debug.info("trainer", f"Debug mode enabled (level={debug_level})")
        
        debug.section("trainer", "TRAINER INITIALIZATION", DEBUG_INFO)
        
        self.config = config
        
        # Validate configuration for conflicts and inefficiencies
        validate_config(config, is_main=is_main_process())
        
        self.rank, self.local_rank, self.world_size = init_dist_if_needed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        debug.info("trainer", f"Device: {self.device.type}")
        debug.info("trainer", f"Distributed: world_size={self.world_size} rank={self.rank}")
        
        if self.device.type == "cuda" and self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            debug.debug("trainer", f"Set CUDA device to local_rank={self.local_rank}")
        
        # Enable cuDNN benchmark mode for faster convolutions
        # This is safe because our cuDNN inputs have fixed sizes:
        #   - Camera images: resized to fixed 384x384 in preprocessing
        # Variable-length text sequences don't use cuDNN (they use matmuls)
        if config.get("cudnn_benchmark", True) and self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            debug.info("trainer", "cuDNN benchmark mode enabled (fixed input sizes verified)")
        
        # Logging
        self.tee = None
        self.out_dir = Path(config["out_dir"])
        if is_main_process():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.tee = Tee(self.out_dir / "train.log")
            sys.stdout = self.tee
            sys.stderr = self.tee  # Also capture stderr (warnings, etc.)
            
            # Debug logger outputs to terminal (captured by Tee)
            # No separate debug.log file needed
            debug.info("system", f"Logging to: {self.out_dir / 'train.log'}")
        
        self._config_dumped = False
        
        mixed_prec = config.get('mixed_precision', 'fp16' if config.get('fp16', False) else 'no')
        print(f"[device] {self.device.type}  mixed_precision={mixed_prec}  GPUs={self.world_size}")
        
        # Set seed
        debug.debug("trainer", f"Setting random seed: {config['seed']}")
        set_seed(config["seed"])
        
        # Initialize models
        debug.info("trainer", "Initializing models...")
        self._setup_models()
        
        # CRITICAL FIX: Cast all trainable parameters to float32 for GradScaler compatibility
        # Mixed precision (fp16) requires master weights/trainable params to be fp32
        # This prevents "ValueError: Attempting to unscale FP16 gradients"
        if is_main_process():
            print("[dtype] Casting trainable parameters to float32 for mixed precision stability...")
        
        def cast_trainable_to_fp32(model_or_iterator, name):
            count = 0
            # Handle both nn.Module and iterator/list of parameters
            if isinstance(model_or_iterator, torch.nn.Module):
                iterator = model_or_iterator.parameters()
            else:
                iterator = model_or_iterator
                
            for p in iterator:
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.to(dtype=torch.float32)
                    count += 1
            if count > 0 and is_main_process():
                print(f"[dtype] Cast {count} trainable parameters in {name} to float32")

        cast_trainable_to_fp32(self.base, "LLM (Base) - LoRA adapters")
        # NOTE: VisionAdapter stays in FP16 - it doesn't use LoRA and receives FP16 inputs from DeepEncoder
        # Only cast CLIP LoRA adapters and Projector (which outputs to FP16 VisionAdapter input)
        
        if hasattr(self.runtime, "clip_vit"):
            # Only cast LoRA parameters in CLIP (not the entire model)
            lora_count = 0
            for n, p in self.runtime.clip_vit.named_parameters():
                if p.requires_grad and "lora_" in n and p.dtype != torch.float32:
                    p.data = p.data.to(dtype=torch.float32)
                    lora_count += 1
            if lora_count > 0 and is_main_process():
                print(f"[dtype] Cast {lora_count} LoRA parameters in DeepEncoder CLIP to float32")
        
        # Initialize datasets
        debug.info("trainer", "Initializing datasets...")
        self._setup_datasets()
        
        # Verify dtype consistency across all models
        if is_main_process():
            print("\n[dtype] Model dtype verification:")
            # Check if using QLoRA/LoRA
            use_qlora = (config.get("tuning_mode", "qlora") == "qlora")
            if use_qlora:
                print(f"[dtype]   LLM: 4-bit NF4 (compute_dtype={config.get('qlora_compute_dtype', 'bfloat16')})")
            else:
                print(f"[dtype]   LLM: {next(self.base.parameters()).dtype}")
            print(f"[dtype]   Vision Adapter: {next(self.vision_adapter.parameters()).dtype}")
            print(f"[dtype]   DeepEncoder CLIP: {next(self.runtime.clip_vit.parameters()).dtype}")
            print(f"[dtype]   DeepEncoder Projector: {next(self.runtime.projector.parameters()).dtype}")
            print(f"[dtype]   DeepEncoder SAM: {next(self.runtime.sam.parameters()).dtype}")
        
        # Print parameter counts
        t_base, a_base, _ = count_trainable_params(self.base)
        t_adapter, a_adapter, _ = count_trainable_params(self.vision_adapter)
        
        # Check if runtime has parameters() method 
        if hasattr(self.runtime, "parameters"):
           # Create a dummy container to reuse the counting utility
            class RuntimeContainer(nn.Module):
                def __init__(self, params):
                    super().__init__()
                    self.params = nn.ParameterList(list(params))
            
            t_runtime, a_runtime, _ = count_trainable_params(RuntimeContainer(self.runtime.parameters()))
        else:
            # Fallback (should not happen with our fix)
            t_runtime, a_runtime = 0, 0
            
        print(f"[param] trainable={t_base + t_adapter + t_runtime:,}")
        
        debug.param_count("trainer", "base_model", self.base)
        debug.param_count("trainer", "vision_adapter", self.vision_adapter)
        
        # Setup DDP
        debug.info("trainer", "Setting up DDP...")
        self._setup_ddp()
        
        # Setup optimizer
        debug.info("trainer", "Setting up optimizer and scheduler...")
        self._setup_optimizer()
        
        # Training state
        self.start_epoch = 1
        self.global_step = 0
        self.start_step_in_epoch = 0  # For mid-epoch resume
        self.epoch_losses = []
        self.val_losses = []
        self.val_epochs = []
        self.best_val_loss = float("inf")
        self.best_step = None
        
        # Step-level loss tracking for detailed plots
        self.step_losses = []  # List of loss values at each step
        self.step_loss_steps = []  # Corresponding global step numbers
        
        # Cache special token embeddings to avoid tokenizing them every step
        self._special_token_cache = {}
        
        # Metric tracking for live plotting (three dashboards)
        self.caption_metrics_history = {
            "bleu4": [],
            "cider": [],
            "spice": [],
            "bertscore_f1": []
        }

        self.metrics_epochs = []
        
        # Handle mixed precision config (supports legacy fp16 and new mixed_precision)
        # NOTE: Must initialize scaler BEFORE _try_resume() so it can restore scaler state
        mixed_prec = config.get('mixed_precision', 'fp16' if config.get('fp16', False) else 'no')
        self.use_amp = mixed_prec in ['fp16', 'bf16'] and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if mixed_prec == 'fp16' else torch.bfloat16 if mixed_prec == 'bf16' else torch.float32
        debug.info("trainer", f"Mixed precision (AMP): {self.use_amp}, dtype: {self.amp_dtype}")
        
        # GradScaler for mixed precision training (handles loss scaling to prevent underflow)
        # Note: GradScaler is only needed for fp16; bf16 typically doesn't need scaling
        # but we enable it anyway for consistent handling
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and mixed_prec == 'fp16')
        if self.use_amp and mixed_prec == 'fp16':
            debug.info("trainer", "GradScaler enabled for fp16 mixed precision")
        
        # Resume if configured (must be after scaler initialization to restore scaler state)
        if config["resume"]:
            debug.info("trainer", "Attempting to resume from checkpoint...")
            self._try_resume()
        
        if is_main_process():
            print(f"[train] epochs={config['epochs']} steps/epoch={self.steps_per_epoch} total_steps={self.total_steps}")
            debug.info("trainer", f"Training plan: epochs={config['epochs']}, steps_per_epoch={self.steps_per_epoch}, total_steps={self.total_steps}")
        
        debug.section("trainer", "INITIALIZATION COMPLETE", DEBUG_INFO)
    
    def _setup_models(self):
        """Initialize all models"""
        (
            self.tok,
            self.base,
            self.vision_adapter,
            self.runtime,
            self.nusc,
            self.d_model,
        ) = setup_models(self.config, self.device, is_main_process())
        
        # Logging & Verification
        if is_main_process():
            self._log_model_stats("LLM (Base)", self.base)
            self._log_model_stats("DeepEncoder CLIP", self.runtime.clip_vit)
            self._log_model_stats("Vision Adapter", self.vision_adapter)
            
            self._verify_initialization("LLM", self.base)
            self._verify_initialization("DeepEncoder CLIP", self.runtime.clip_vit)
            self._verify_initialization("Vision Adapter", self.vision_adapter)

    def _log_model_stats(self, name: str, model: nn.Module):
        """Log detailed parameter statistics (Frozen vs Trainable/Random)"""
        print(f"\n[param_stats] ========= {name} Statistics =========")
        
        total_storage = 0
        total_logical = 0
        trainable_params = 0
        frozen_params = 0
        
        # Categorize by module/type
        categories = {
            "Pretrained/Frozen": 0,
            "Adapter/LoRA (Trainable)": 0,
            "Projector/Head (Trainable)": 0,
            "Other (Trainable)": 0
        }
        
        for n, p in model.named_parameters():
            storage_count = p.numel()
            
            # Check for 4-bit quantization (packed 2 params per byte)
            if p.dtype == torch.uint8:
                logical_count = storage_count * 2
            else:
                logical_count = storage_count
                
            total_storage += storage_count
            total_logical += logical_count
            
            if p.requires_grad:
                trainable_params += logical_count
                # Heuristic categorization
                if "lora_" in n or "adapter" in n:
                    categories["Adapter/LoRA (Trainable)"] += logical_count
                elif "projector" in n or "head" in n or "embed" in n:
                    categories["Projector/Head (Trainable)"] += logical_count
                else:
                    categories["Other (Trainable)"] += logical_count
            else:
                frozen_params += logical_count
                categories["Pretrained/Frozen"] += logical_count
        
        print(f"[param_stats] Total Params (Logical): {total_logical:,}")
        if total_logical != total_storage:
             print(f"[param_stats] Total Storage (Packed): {total_storage:,} (Quantized)")
             
        if total_logical > 0:
            print(f"[param_stats] Frozen (Pretrained): {frozen_params:,} ({frozen_params/total_logical:.1%})")
            print(f"[param_stats] Trainable (New/Fine-tuned): {trainable_params:,} ({trainable_params/total_logical:.1%})")
        
        for cat, count in categories.items():
            if count > 0:
                print(f"[param_stats]   - {cat}: {count:,}")

    def _verify_initialization(self, name: str, model: nn.Module):
        """Verify weights are valid (no NaN/Inf) and key layers are initialized"""
        print(f"[param_stats] Verifying {name} initialization...")
        has_issue = False
        
        for n, p in model.named_parameters():
            # Check for NaNs
            if torch.isnan(p).any():
                print(f"[param_stats] ❌ NaN found in {n}")
                has_issue = True
            if torch.isinf(p).any():
                print(f"[param_stats] ❌ Inf found in {n}")
                has_issue = True
                
            # Check for Zero Init (only warnings)
            # lora_B is typically zero-initialized, so we skip it
            if p.requires_grad and "lora_B" not in n:
                if (p == 0).all():
                    # Only warn if it's not a bias (biases can be zero)
                    if "bias" not in n:
                        print(f"[param_stats] ⚠️  Warning: {n} is all zeros (unexpected for weight)")
                        
        if not has_issue:
            print(f"[param_stats] ✓ Weights verified: No NaNs/Infs found.")
    
    def _setup_datasets(self):
        """Initialize datasets and dataloaders"""
        # Check if we should load images in DataLoader workers
        load_images = True
        
        # Master validation toggle - if True, skip ALL validation checks
        skip_all = self.config.get("skip_all_validation", False)
        if skip_all and is_main_process():
            print("[dataset] ⚠️  skip_all_validation=True - ALL data validation disabled!")
        
        # Full dataset with comprehensive validation
        ds_full = VisionNuDataset(
            json_paths=self.config["jsons"],
            target_field=self.config["target_field"],
            max_samples=self.config["max_samples"],
            nusc=self.nusc,
            load_images=True,
            view_order=DEFAULT_VIEW_ORDER,
            validate_json_schema=not skip_all,
            validate_image_paths=not skip_all,
        )

        
        # Train/val split
        val_size = max(1, int(len(ds_full) * self.config["val_split"]))
        train_size = len(ds_full) - val_size
        
        set_seed(self.config["seed"])
        ds_train, ds_val = torch.utils.data.random_split(
            ds_full, [train_size, val_size], generator=torch.Generator().manual_seed(self.config["seed"])
        )
        
        # Store reference to full dataset for token2path access
        # self.ds_full = ds_full
        
        if is_main_process():
            print(f"[dataset] train={train_size}  val={val_size}")
        
        
        # Samplers
        sampler_train = (
            DistributedSampler(ds_train, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            if self.world_size > 1
            else SingleProcessDetSampler(ds_train, seed=self.config["seed"], shuffle=True)
        )
        
        # DataLoaders - num_workers > 0 enables parallel data loading
        # This improves GPU utilization by overlapping data loading with training
        num_workers = self.config.get("num_workers", 0)
        
        if is_main_process():
            print(f"[dataloader] num_workers={num_workers}")
        
        # persistent_workers keeps workers alive between epochs (avoids respawn overhead)
        # prefetch_factor controls how many batches each worker pre-loads
        use_persistent = num_workers > 0
        prefetch = self.config.get("prefetch_factor", 2) if num_workers > 0 else None
        
        if is_main_process() and use_persistent:
            print(f"[dataloader] persistent_workers=True, prefetch_factor={prefetch}")
        
        # Create collate function with image loading flag
        collate_fn = make_collate(
            self.tok, 
            self.config["max_ans_toks"], 
            self.config.get("system_prompt", ""),
            load_images=load_images
        )
        
        self.dl_train = DataLoader(
            ds_train,
            batch_size=self.config["batch_size"],
            shuffle=False,
            sampler=sampler_train,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=use_persistent,
            prefetch_factor=prefetch,
            collate_fn=collate_fn,
        )
        
        self.dl_val = DataLoader(
            ds_val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=use_persistent,
            prefetch_factor=prefetch,
            collate_fn=collate_fn,
        )
        
        self.ds_val = ds_val
        self.sampler_train = sampler_train
        self.train_size = train_size
        self.load_images = load_images  # Store for use in training loop
    
    def _setup_ddp(self):
        """Setup distributed data parallel"""
        if self.world_size > 1:
            # find_unused_parameters=False improves performance when all params are always used
            # Set to True only if you see "unused parameter" errors during training
            # Removed LiDAR VAT DDP wrapping
            self.base = nn.parallel.DistributedDataParallel(
                self.base, device_ids=[self.local_rank], find_unused_parameters=False
            )
            self.vision_adapter = nn.parallel.DistributedDataParallel(
                self.vision_adapter, device_ids=[self.local_rank], find_unused_parameters=False
            )
            self.runtime.projector = nn.parallel.DistributedDataParallel(
                self.runtime.projector, device_ids=[self.local_rank], find_unused_parameters=False
            )
            self.runtime.clip_vit = nn.parallel.DistributedDataParallel(
                self.runtime.clip_vit, device_ids=[self.local_rank], find_unused_parameters=False
            )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optim, self.sched, self.sched_meta = setup_optimizer_and_scheduler(
            self.base,
            self.vision_adapter,
            self.runtime,
            self.config,
            self.train_size,
            self.world_size,
        )
        
        self.total_steps = self.sched_meta["total_steps"]
        effective_batch_size = self.config["batch_size"] * max(1, self.world_size) * self.config["grad_accum"]
        self.steps_per_epoch = max(1, math.ceil(self.train_size / effective_batch_size))
    
    def _validate_lora_config(self, adapter_path: Path, adapter_type: str = "LLM") -> bool:
        """
        Validate that the saved LoRA adapter config matches the current config.
        
        Args:
            adapter_path: Path to the saved adapter directory
            adapter_type: "LLM" or "CLIP" to determine which config keys to use
            
        Returns:
            True if configs match, False otherwise (with warnings printed)
        """
        import json
        
        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            if is_main_process():
                print(f"[resume] WARNING: No adapter_config.json found in {adapter_path}, skipping validation")
            return True  # Allow loading for backward compatibility with old checkpoints
        
        try:
            with open(config_path, "r") as f:
                saved_config = json.load(f)
        except Exception as e:
            if is_main_process():
                print(f"[resume] WARNING: Failed to read adapter_config.json: {e}")
            return True  # Allow loading on read failure
        
        # Get current config values based on adapter type
        if adapter_type == "CLIP":
            current_r = self.config.get("clip_lora_r", 8)
            current_alpha = self.config.get("clip_lora_alpha", 16)
            current_target_modules = self.config.get("clip_lora_target_modules")
        else:  # LLM
            current_r = self.config.get("llm_lora_r", 8)
            current_alpha = self.config.get("llm_lora_alpha", 16)
            current_target_modules = self.config.get("llm_lora_targets")
        
        # Get saved config values
        saved_r = saved_config.get("r")
        saved_alpha = saved_config.get("lora_alpha")
        saved_target_modules = saved_config.get("target_modules")
        
        mismatches = []
        
        # Check rank
        if saved_r is not None and saved_r != current_r:
            mismatches.append(f"lora_r: saved={saved_r}, current={current_r}")
        
        # Check alpha
        if saved_alpha is not None and saved_alpha != current_alpha:
            mismatches.append(f"lora_alpha: saved={saved_alpha}, current={current_alpha}")
        
        # Check target modules (convert to sets for comparison)
        if saved_target_modules is not None and current_target_modules is not None:
            saved_set = set(saved_target_modules) if isinstance(saved_target_modules, list) else {saved_target_modules}
            current_set = set(current_target_modules) if isinstance(current_target_modules, list) else {current_target_modules}
            if saved_set != current_set:
                mismatches.append(f"target_modules: saved={sorted(saved_set)}, current={sorted(current_set)}")
        
        if mismatches:
            if is_main_process():
                print(f"[resume] ERROR: {adapter_type} LoRA config mismatch detected!")
                for m in mismatches:
                    print(f"[resume]   - {m}")
                print(f"[resume] This may cause corrupted weights or runtime errors.")
                print(f"[resume] Either use the same LoRA config or start training from scratch.")
            return False
        
        return True

    def _try_resume(self):
        """Try to resume from checkpoint"""
        prev_state, tag = try_load_state(self.out_dir)
        if prev_state is not None:
            if is_main_process():
                print(f"[resume] loading from {tag}")
            
            # Load vision components
            # Note: LLM LoRA is loaded separately below (lines 529-567) from qwen2_lora_adapter_latest/
            va_path = self.out_dir / f"vision_adapter_{tag}.pt"
            proj_path = self.out_dir / f"projector_{tag}.pt"
            
            if va_path.exists():
                vision_adapter_model = (
                    self.vision_adapter.module
                    if isinstance(self.vision_adapter, nn.parallel.DistributedDataParallel)
                    else self.vision_adapter
                )
                vision_adapter_model.load_state_dict(torch.load(va_path, map_location=self.device))
            
            if proj_path.exists():
                proj_model = (
                    self.runtime.projector.module
                    if isinstance(self.runtime.projector, nn.parallel.DistributedDataParallel)
                    else self.runtime.projector
                )
                proj_model.load_state_dict(torch.load(proj_path, map_location=self.device))
                
                # Load CLIP LoRA adapter
                clip_lora_path = self.out_dir / f"clip_lora_adapter_{tag}"
                if clip_lora_path.exists():
                    if is_main_process():
                        print(f"[resume] loading CLIP LoRA adapter from {clip_lora_path}")
                    
                    # Validate LoRA config before loading
                    if not self._validate_lora_config(clip_lora_path, adapter_type="CLIP"):
                        raise RuntimeError(
                            f"CLIP LoRA config mismatch. Cannot resume with different LoRA configuration. "
                            f"Please use the same lora_r, lora_alpha, and clip_lora_target_modules as the checkpoint, "
                            f"or start training from scratch with resume=False."
                        )
                    
                    clip_vit_unwrapped = (
                        self.runtime.clip_vit.module
                        if isinstance(self.runtime.clip_vit, nn.parallel.DistributedDataParallel)
                        else self.runtime.clip_vit
                    )
                    # Load adapter weights using PEFT's set_peft_model_state_dict
                    adapter_weights_path = clip_lora_path / "adapter_model.safetensors"
                    if adapter_weights_path.exists():
                        from safetensors.torch import load_file
                        adapter_state = load_file(str(adapter_weights_path))
                    else:
                        adapter_weights_path = clip_lora_path / "adapter_model.bin"
                        if adapter_weights_path.exists():
                            adapter_state = torch.load(adapter_weights_path, map_location=self.device)
                        else:
                            # This is a critical error - adapter_config.json exists but weights are missing
                            # This indicates a corrupted or incomplete checkpoint
                            raise FileNotFoundError(
                                f"CLIP LoRA adapter config exists at {clip_lora_path}/adapter_config.json "
                                f"but no adapter weights found (checked adapter_model.safetensors and adapter_model.bin). "
                                f"This indicates a corrupted checkpoint. Either restore the weights or start fresh with resume=False."
                            )
                    
                    # adapter_state is guaranteed to be set if we reach here (otherwise exception raised)
                    from peft import set_peft_model_state_dict
                    set_peft_model_state_dict(clip_vit_unwrapped, adapter_state)
                    if is_main_process():
                        print(f"[resume] CLIP LoRA adapter loaded successfully via set_peft_model_state_dict()")
                
                # Load SAM compression head (net_2 and net_3 - the trainable DeepEncoder/VARY layers)
                sam_compression_head_path = self.out_dir / f"sam_compression_head_{tag}.pt"
                if sam_compression_head_path.exists():
                    if is_main_process():
                        print(f"[resume] loading SAM compression head from {sam_compression_head_path}")
                    sam_model = (
                        self.runtime.sam.module
                        if isinstance(self.runtime.sam, nn.parallel.DistributedDataParallel)
                        else self.runtime.sam
                    )
                    sam_compression_head_state = torch.load(sam_compression_head_path, map_location=self.device)
                    # Load only the compression head parameters (net_2, net_3)
                    current_state = sam_model.state_dict()
                    for name, param in sam_compression_head_state.items():
                        if name in current_state:
                            current_state[name] = param
                    sam_model.load_state_dict(current_state)
                    if is_main_process():
                        print(f"[resume] SAM compression head loaded successfully ({len(sam_compression_head_state)} parameters)")
                else:
                    if is_main_process():
                        print(f"[resume] WARNING: No SAM compression head found at {sam_compression_head_path}")
            
            # Load LLM LoRA adapter
            lora_path = self.out_dir / f"qwen2_lora_adapter_{tag}"
            if lora_path.exists():
                if is_main_process():
                    print(f"[resume] loading LLM LoRA adapter from {lora_path}")
                
                # Validate LoRA config before loading
                if not self._validate_lora_config(lora_path, adapter_type="LLM"):
                    raise RuntimeError(
                        f"LLM LoRA config mismatch. Cannot resume with different LoRA configuration. "
                        f"Please use the same lora_r, lora_alpha, and lora_target_modules as the checkpoint, "
                        f"or start training from scratch with resume=False."
                    )
                
                base_model = (
                    self.base.module
                    if isinstance(self.base, nn.parallel.DistributedDataParallel)
                    else self.base
                )
                # Load adapter weights using PEFT's set_peft_model_state_dict
                adapter_weights_path = lora_path / "adapter_model.safetensors"
                if adapter_weights_path.exists():
                    from safetensors.torch import load_file
                    adapter_state = load_file(str(adapter_weights_path))
                else:
                    adapter_weights_path = lora_path / "adapter_model.bin"
                    if adapter_weights_path.exists():
                        adapter_state = torch.load(adapter_weights_path, map_location=self.device)
                    else:
                        # This is a critical error - adapter_config.json exists but weights are missing
                        # This indicates a corrupted or incomplete checkpoint
                        raise FileNotFoundError(
                            f"LLM LoRA adapter config exists at {lora_path}/adapter_config.json "
                            f"but no adapter weights found (checked adapter_model.safetensors and adapter_model.bin). "
                            f"This indicates a corrupted checkpoint. Either restore the weights or start fresh with resume=False."
                        )
                
                # adapter_state is guaranteed to be set if we reach here (otherwise exception raised)
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(base_model, adapter_state)
                if is_main_process():
                    print(f"[resume] LLM LoRA adapter loaded successfully via set_peft_model_state_dict()")
            
            # Load optimizer/scheduler
            self.optim.load_state_dict(prev_state["optimizer"])
            self.sched.load_state_dict(prev_state["scheduler"])
            
            # Migrate optimizer state tensors to the correct device
            # Checkpoint is loaded with map_location="cpu", but models are on GPU
            # This prevents device mismatch warnings and CPU-GPU transfer slowdowns
            for state in self.optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            if is_main_process():
                print(f"[resume] Optimizer state migrated to {self.device}")
            
            # Validate mixed_precision mode consistency and restore GradScaler state
            current_mixed_prec = self.config.get('mixed_precision', 'fp16' if self.config.get('fp16', False) else 'no')
            saved_mixed_prec = prev_state.get("mixed_precision")
            
            if saved_mixed_prec is not None and saved_mixed_prec != current_mixed_prec:
                # Mixed precision mode changed between checkpoint and current run
                if is_main_process():
                    print(f"[resume] WARNING: mixed_precision mode changed from '{saved_mixed_prec}' to '{current_mixed_prec}'")
                    print(f"[resume] GradScaler state will NOT be restored due to mode change")
                # Don't restore scaler state - it's incompatible
            elif prev_state.get("scaler") is not None and hasattr(self, 'scaler') and current_mixed_prec == 'fp16':
                # Only restore scaler state if:
                # 1. Scaler state exists in checkpoint
                # 2. Current mode is fp16 (where scaler is actually enabled)
                # 3. Mixed precision mode is consistent (or old checkpoint without saved mode)
                self.scaler.load_state_dict(prev_state["scaler"])
                if is_main_process():
                    print(f"[resume] GradScaler state restored for fp16 mode")
            elif current_mixed_prec == 'bf16' and is_main_process():
                # bf16 mode doesn't use GradScaler, so no restoration needed
                print(f"[resume] bf16 mode: GradScaler not used, skipping scaler state restoration")
            
            # Restore training state (supports both epoch-level and step-level resume)
            saved_epoch = prev_state["epoch"]
            self.global_step = prev_state["global_step"]
            self.epoch_losses = prev_state.get("epoch_losses", [])
            self.val_losses = prev_state.get("val_losses", [])
            self.val_epochs = prev_state.get("val_epochs", [])
            self.best_val_loss = prev_state.get("best_loss", float("inf"))
            self.best_step = prev_state.get("best_step", None)
            
            # Check if this is a mid-epoch checkpoint (step-based save)
            step_in_epoch = prev_state.get("step_in_epoch", None)
            if step_in_epoch is not None and step_in_epoch > 0:
                # Mid-epoch resume: continue from this epoch at the saved step
                self.start_epoch = saved_epoch
                self.start_step_in_epoch = step_in_epoch
                if is_main_process():
                    print(f"[resume] Mid-epoch resume: epoch {saved_epoch}, step {step_in_epoch}/{self.steps_per_epoch}")
            else:
                # End-of-epoch checkpoint: resume from next epoch
                self.start_epoch = saved_epoch + 1
                self.start_step_in_epoch = 0
            
            # Restore metrics history for live plotting
            if prev_state.get("caption_metrics_history") is not None:
                self.caption_metrics_history = prev_state["caption_metrics_history"]
            if prev_state.get("metrics_epochs") is not None:
                self.metrics_epochs = prev_state["metrics_epochs"]
            
            # Restore step-level loss history
            if prev_state.get("step_losses") is not None:
                self.step_losses = prev_state["step_losses"]
            if prev_state.get("step_loss_steps") is not None:
                self.step_loss_steps = prev_state["step_loss_steps"]
            
            # Restore RNG states for reproducibility
            random.setstate(prev_state["rng"]["py_random"])
            np.random.set_state(prev_state["rng"]["np_random"])
            torch.set_rng_state(prev_state["rng"]["torch"])
            if prev_state["rng"]["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(prev_state["rng"]["torch_cuda"])
            
            if is_main_process():
                if self.start_step_in_epoch > 0:
                    print(f"[resume] Resuming epoch {self.start_epoch} from step {self.start_step_in_epoch}, global_step {self.global_step}")
                else:
                    print(f"[resume] Completed epoch {saved_epoch}, resuming from epoch {self.start_epoch}, global_step {self.global_step}")
        else:
            if is_main_process():
                print(f"[resume] no checkpoint found in {self.out_dir}, starting fresh")
    
    def _set_epoch(self, epoch: int):
        """Set epoch for samplers"""
        if isinstance(self.sampler_train, (SingleProcessDetSampler, DistributedSampler)):
            self.sampler_train.set_epoch(epoch)
    
    def train(self):
        """Main training loop"""
        # Log component toggles
        if is_main_process():
            # Log image loading mode
            if self.load_images:
                print(f"[vision_pipeline] Using DataLoader workers for image loading (fast path)")
            else:
                print(f"[vision_pipeline] Loading images in training loop (fallback path)")
        
        # Set models to train mode
        self.base.train()
        # Removed LiDAR VAT train mode
        if True:
            self.vision_adapter.train()
            # Use runtime.train() to properly set CLIP/Projector to train mode
            # while keeping SAM frozen in eval mode
            self.runtime.train()
        
        # Check if training is already complete
        if self.start_epoch > self.config["epochs"]:
            if is_main_process():
                print(f"[train] Training already complete (epoch {self.start_epoch - 1}/{self.config['epochs']})")
                print(f"[train] To continue training, increase 'epochs' in config")
            return
        
        if self.global_step >= self.total_steps:
            if is_main_process():
                print(f"[train] Training already complete (step {self.global_step}/{self.total_steps})")
            return
        
        pbar = tqdm(
            total=self.total_steps,
            initial=self.global_step,
            disable=not is_main_process(),
            dynamic_ncols=True,
        )
        
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            self._set_epoch(epoch)
            
            epoch_loss_sum = 0.0
            epoch_count = 0
            step_in_epoch = 0  # Track steps within this epoch for mid-epoch checkpointing
            
            # Determine if we need to skip batches for mid-epoch resume
            skip_batches = 0
            if epoch == self.start_epoch and self.start_step_in_epoch > 0:
                # Calculate how many batches to skip based on the step we're resuming from
                skip_batches = self.start_step_in_epoch * self.config["grad_accum"]
                if is_main_process():
                    print(f"[resume] Skipping first {skip_batches} batches to reach step {self.start_step_in_epoch}")
            
            for it, batch in enumerate(self.dl_train, start=1):
                # Skip batches for mid-epoch resume
                if it <= skip_batches:
                    continue
                
                loss = self._train_step(batch)
                epoch_loss_sum += loss
                epoch_count += 1
                
                if it % self.config["grad_accum"] == 0:
                    self._optimizer_step()
                    self.global_step += 1
                    step_in_epoch += 1
                    pbar.update(1)
                    pbar.set_postfix(loss=loss, lr=f"{self.sched.get_last_lr()[0]:.2e}")
                    
                    # Track step-level loss for detailed plotting
                    if is_main_process():
                        self.step_losses.append(loss)
                        self.step_loss_steps.append(self.global_step)
                    
                    # Step-based checkpoint saving and plotting
                    save_every = self.config.get("save_every_steps", 0)
                    if save_every > 0 and self.global_step % save_every == 0 and is_main_process():
                        self._save_step_checkpoint(epoch, step_in_epoch)
                        # Plot step-level training curve
                        self._plot_step_loss_curve()
                    
                    if self.global_step >= self.total_steps:
                        break
            
            # Flush any remaining accumulated gradients at end of epoch
            # This handles cases where len(dataloader) % grad_accum != 0
            remaining_grads = epoch_count % self.config["grad_accum"]
            if remaining_grads > 0 and self.global_step < self.total_steps:
                self._optimizer_step()
                self.global_step += 1
                pbar.update(1)
            
            # Epoch complete
            avg_epoch_loss = epoch_loss_sum / max(1, epoch_count)
            self.epoch_losses.append(avg_epoch_loss)
            
            if is_main_process():
                print(f"\n[epoch {epoch}] train_loss={avg_epoch_loss:.4f}")
            
            # Validation
            if epoch % self.config["validate_every"] == 0:
                self._run_validation(epoch)
            
            # Save checkpoint at end of each epoch
            if is_main_process():
                self._save_checkpoint(epoch)
            
            # Run inference sampling
            # Run on: 1) Every N epochs (where epoch % N == 0), OR 2) Final epoch
            # If inference_every <= 0, inference sampling is disabled
            inference_every = self.config.get("inference_sampling_every", 3)
            is_final_epoch = (epoch == self.config["epochs"])
            
            # Handle disabled inference (inference_every <= 0)
            if inference_every > 0:
                should_run_inference = (epoch % inference_every == 0) or is_final_epoch
            else:
                should_run_inference = False  # Disabled
            
            if is_main_process() and should_run_inference:
                if is_final_epoch and (epoch % inference_every != 0):
                    print(f"\n[inference_sampling] Running at FINAL epoch {epoch}")
                else:
                    print(f"\n[inference_sampling] Running at epoch {epoch}")
                metrics = run_inference_sampling(
                    self.base,
                    self.vision_adapter,
                    self.runtime,
                    self.nusc,
                    self.tok,
                    self.config,
                    self.out_dir,
                    epoch,
                    self.device,
                    self.best_step,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    val_dataset=self.ds_val,
                )
                
                # Store metrics for live plotting
                if metrics:
                    self.metrics_epochs.append(epoch)
                    
                    # Store caption metrics if available
                    if "caption_dashboard" in metrics:
                        cap_dash = metrics["caption_dashboard"]
                        self.caption_metrics_history["bleu4"].append(cap_dash.get("bleu4", 0.0))
                        self.caption_metrics_history["cider"].append(cap_dash.get("cider", 0.0))
                        self.caption_metrics_history["spice"].append(cap_dash.get("spice", 0.0))
                        self.caption_metrics_history["bertscore_f1"].append(cap_dash.get("bertscore_f1", 0.0))
                        print(f"[trainer] Stored caption metrics for epoch {epoch}")
                    
                    # Generate live plots
                    try:
                        plot_all_metrics(
                            self.caption_metrics_history,
                            self.metrics_epochs,
                            self.out_dir
                        )
                        print(f"[trainer] Updated metric plots at epoch {epoch}")
                    except Exception as plot_error:
                        print(f"[trainer] Warning: Failed to generate plots: {plot_error}")

            
            # Plot
            if is_main_process():
                plot_loss_curve(self.epoch_losses, self.val_losses, self.val_epochs, self.out_dir)
            
            if self.global_step >= self.total_steps:
                break
        
        pbar.close()
        
        if is_main_process():
            print(f"[done] training complete. Best val loss: {self.best_val_loss:.4f} at step {self.best_step}")
            if isinstance(sys.stdout, Tee):
                orig = sys.stdout.stdout
                tee = sys.stdout
                sys.stdout = orig
                sys.stderr = orig  # Restore stderr as well
                tee.close()
    
    def _train_step(self, batch):
        """Single training step"""
        debug.start_timer("trainer", "train_step")
        debug.trace("trainer", "=" * 60)
        debug.trace("trainer", "TRAINING STEP START")
        debug.trace("trainer", "=" * 60)
        
        # Use non_blocking=True for async CPU→GPU transfers (overlaps with other work)
        p_ids = batch["prompt_ids"].to(self.device, non_blocking=True)
        a_ids = batch["answer_ids"].to(self.device, non_blocking=True)
        sample_tokens = batch["sample_tokens"]
        
        debug.shape("trainer", "prompt_ids", p_ids)
        debug.shape("trainer", "answer_ids", a_ids)
        debug.debug("trainer", f"Batch size: {p_ids.shape[0]}")
        debug.debug("trainer", f"Sample tokens: {sample_tokens[:3] if len(sample_tokens) > 3 else sample_tokens}...")
        
        # Check training toggles
        vision_kv = None  # Initialize to avoid unbound error if encoding fails
        if self.config["use_vision"]:
            debug.start_timer("trainer", "vision_processing")
            debug.data_flow("trainer", "vision_start", f"Processing {len(sample_tokens)} samples")
            
            # Check if images are pre-loaded by DataLoader workers
            if self.load_images and "images" in batch:
                # Fast path: images already loaded by workers, just encode on GPU
                batch_images = batch["images"]  # List[List[Optional[Tensor]]]
                debug.debug("trainer", f"Using pre-loaded images from DataLoader workers")
                
                try:
                    # Encode pre-loaded images (no I/O, just GPU work)
                    # encode_preloaded_views_batch returns tensors already on GPU
                    batch_view_tokens = self.runtime.encode_preloaded_views_batch(
                        batch_images, view_order=DEFAULT_VIEW_ORDER
                    )
                    
                    # Tensors are already on GPU from encode_preloaded_views_batch
                    # Use batched VisionAdapter forward pass
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        vision_kv = self.vision_adapter.forward_batch(batch_view_tokens)
                    # vision_kv is List[List[Tensor]], each tensor is [HW, d_model]
                    debug.debug("trainer", f"Vision batch complete: {len(vision_kv)} samples, {len(vision_kv[0])} views")
                    debug.data_flow("trainer", "vision_complete", f"samples={len(vision_kv)}, views={len(vision_kv[0])}")
                    
                except Exception as e:
                    debug.warn("trainer", f"Pre-loaded image encoding failed: {e}")
                    if is_main_process():
                        print(f"[warn] Pre-loaded image encoding failed: {e}")
                    # vision_kv remains None, will be skipped
            else:
                # Fallback: load images in training loop (slower, original path)
                debug.debug("trainer", "Loading images in training loop (fallback path)")
                try:
                    batch_view_tokens = batch_multiview_tokens_from_sample_tokens(
                        sample_tokens, self.nusc, runtime=self.runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                    )
                    
                    # Move all tensors to device first
                    batch_view_tokens_device = [
                        [t.to(self.device) for t in view_tokens]
                        for view_tokens in batch_view_tokens
                    ]
                    
                    # Use batched VisionAdapter forward pass (single operation for all B samples)
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        vision_kv = self.vision_adapter.forward_batch(batch_view_tokens_device)  # List[List[Tensor]]
                    # vision_kv is List[List[Tensor]], each tensor is [HW, d_model]
                    debug.debug("trainer", f"Vision batch complete: {len(vision_kv)} samples, {len(vision_kv[0])} views")
                    debug.data_flow("trainer", "vision_complete", f"samples={len(vision_kv)}, views={len(vision_kv[0])}")
                    
                except Exception as e:
                    # Fallback to sequential processing if batched fails
                    debug.warn("trainer", f"Batched vision encoding failed: {e}, falling back to sequential")
                    if is_main_process():
                        print(f"[warn] Batched vision encoding failed, using sequential fallback...")
                    
                    vision_kvs = []
                    for idx, tok_str in enumerate(sample_tokens):
                        mv = multiview_tokens_from_sample_token(
                            tok_str, self.nusc, runtime=self.runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                        )
                        
                        if not mv.get("tokens") or len(mv["tokens"]) != NUM_VIEWS:
                            dummy_shape = (TOKENS_PER_VIEW, PROJECTOR_DIM)  # [256, 2048] per view
                            mv["tokens"] = [torch.zeros(dummy_shape, device=self.device, dtype=self.amp_dtype) for _ in range(NUM_VIEWS)]
                        
                        vt = [t.to(self.device) for t in mv["tokens"]]
                        
                        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                            kv = self.vision_adapter(vt)
                            kv = kv.unsqueeze(0)
                        vision_kvs.append(kv)
                    
                    vision_kv = torch.cat(vision_kvs, dim=0)
            
            debug.tensor_stats("trainer", "vision_kv", vision_kv)
            debug.end_timer("trainer", "vision_processing")
        else:
            debug.debug("trainer", "Vision processing skipped (disabled or not configured)")
        
        # Forward pass
        debug.start_timer("trainer", "forward_pass")
        debug.data_flow("trainer", "forward_start", "Building embeddings")
        
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            E = self.base.get_input_embeddings()
            debug.debug("trainer", f"Embedding layer: {E}")
            
            # Cached embedding function for special tokens (avoids tokenizing every step)
            def get_cached_emb(txt: str) -> torch.Tensor:
                if txt not in self._special_token_cache:
                    ids = self.tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                    self._special_token_cache[txt] = E(ids)  # [1, 1, d_model]
                return self._special_token_cache[txt]
            
            # Process LiDAR (skipped/removed)
            prefix_lidar = None

            
            # Process vision (if enabled and available)
            # Flatten batch of lists -> list of batched tensors
            batched_view_tokens = None
            if vision_kv is not None and self.config["use_vision"]:
                debug.start_timer("trainer", "vision_collation")
                # vision_kv is List[List[Tensor]] -> [Batch][View]
                # We need List[Tensor] -> [View][Batch, Seq, Dim]
                try:
                    num_views = len(vision_kv[0])
                    batched_view_tokens = []
                    for v_idx in range(num_views):
                        # Stack view v for all samples
                        # each t is [256, d_model]
                        # stacked is [B, 256, d_model]
                        v_tokens = torch.stack([sample[v_idx] for sample in vision_kv], dim=0)
                        batched_view_tokens.append(v_tokens)
                    debug.debug("trainer", f"Collated {num_views} views")
                except Exception as e:
                    debug.error("trainer", f"Vision collation failed: {e}")
                    batched_view_tokens = None
                debug.end_timer("trainer", "vision_collation")
            else:
                debug.debug("trainer", "Vision processing skipped (vision_kv not available or disabled)")

            
            # Get text embeddings
            tok_emb = E(p_ids)
            debug.shape("trainer", "text_embeddings", tok_emb)
            
            # Get answer embeddings
            ans_emb = E(a_ids)
            debug.shape("trainer", "answer_embeddings", ans_emb)
            
            # Build input sequence with explicit position markers
            # Order is guaranteed: vision → lidar → text → answer
            # See sequence_builder.py for position definitions
            debug.data_flow("trainer", "embedding_assembly", "Building input sequence with explicit positions")
            
            B = p_ids.size(0)
            inp, seq_info = build_training_sequence(
                E=E,
                device=self.device,
                dtype=self.amp_dtype,
                batch_size=B,
                tok_emb=tok_emb,
                ans_emb=ans_emb,
                view_tokens_list=batched_view_tokens,
                get_special_token_emb=get_cached_emb,
            )
            
            debug.debug("trainer", f"Input sequence order: {' → '.join(seq_info['order'])}")
            debug.shape("trainer", "full_input_with_answer", inp)
            
            total_len = inp.size(1)
            debug.debug("trainer", f"Final sequence: batch_size={B}, total_length={total_len}")
            
            # Create labels (only answer tokens are supervised)
            # Use explicit position info to find answer location
            if ModalityPosition.ANSWER_TOKENS in seq_info['positions']:
                ans_start, ans_end = seq_info['positions'][ModalityPosition.ANSWER_TOKENS]
                labels = torch.full((B, total_len), -100, dtype=torch.long, device=self.device)
                labels[:, ans_start:ans_end] = a_ids
                debug.debug("trainer", f"Supervised tokens: positions {ans_start}:{ans_end} ({ans_end - ans_start} tokens)")
            else:
                raise RuntimeError("Answer tokens not found in sequence - this should never happen")
            
            debug.shape("trainer", "labels", labels)
            
            # Create attention mask
            attn = torch.ones((B, total_len), dtype=torch.long, device=self.device)
            debug.shape("trainer", "attention_mask", attn)
            
            # Forward through LLM
            debug.data_flow("trainer", "llm_forward", f"Input: {tuple(inp.shape)}")
            debug.memory_usage("trainer", "before_llm")
            
            out = self.base(inputs_embeds=inp, attention_mask=attn, labels=labels)
            loss = out.loss / self.config["grad_accum"]
            
            debug.debug("trainer", f"Loss: {loss.item() * self.config['grad_accum']:.6f}")
            debug.memory_usage("trainer", "after_llm")
            debug.end_timer("trainer", "forward_pass")
        
        # Backward pass with GradScaler for mixed precision
        debug.start_timer("trainer", "backward_pass")
        debug.data_flow("trainer", "backward", "Computing gradients")
        
        # Use scaler for fp16 (scales loss to prevent gradient underflow)
        # For bf16, scaler is disabled but we use consistent code path
        self.scaler.scale(loss).backward()
        
        debug.end_timer("trainer", "backward_pass")
        
        debug.end_timer("trainer", "train_step")
        debug.trace("trainer", "TRAINING STEP END")
        debug.trace("trainer", "=" * 60)
        
        return loss.item() * self.config["grad_accum"]
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping and GradScaler"""
        debug.start_timer("trainer", "optimizer_step")
        debug.trace("trainer", "Clipping gradients and updating weights")
        
        # Unscale gradients before clipping (required for proper grad norm computation)
        self.scaler.unscale_(self.optim)
        
        # --- Value Verification: Vision Adapter Gradients ---
        if self.global_step % 10 == 0:
            total_norm = 0.0
            for p in self.vision_adapter.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            if is_main_process():
                print(f"[grads] Step {self.global_step}: Vision Adapter Grad Norm = {total_norm:.4f} " + 
                      ("✅ (Updating)" if total_norm > 0 else "⚠️ (Zero/No Update)"))
        # ----------------------------------------------------
        
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.base.parameters() if p.requires_grad], self.config["clip_norm"]
        )
        if self.config["use_vision"]:
            torch.nn.utils.clip_grad_norm_(self.vision_adapter.parameters(), self.config["clip_norm"])
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.runtime.projector.parameters() if p.requires_grad], self.config["clip_norm"]
            )
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.runtime.clip_vit.parameters() if p.requires_grad], self.config["clip_norm"]
            )
            # SAM compression head gradients are clipped via runtime.sam trainable parameters
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.runtime.sam.parameters() if p.requires_grad], self.config["clip_norm"]
            )
        
        # Step optimizer with scaler (handles inf/nan gradients gracefully)
        self.scaler.step(self.optim)
        self.scaler.update()
        
        current_lr = self.sched.get_last_lr()[0]
        self.sched.step()
        debug.trace("trainer", f"Learning rate: {current_lr:.2e}")
        
        self.optim.zero_grad(set_to_none=True)
        debug.end_timer("trainer", "optimizer_step")
    
    def _run_validation(self, epoch: int):
        """Run validation"""
        if is_main_process():
            print(f"[validation] epoch {epoch}...")
        
        val_loss = run_validation(
            self.dl_val,
            self.device,
            self.tok,
            self.base,
            self.vision_adapter if self.config["use_vision"] else None,
            self.runtime if self.config["use_vision"] else None,
            self.nusc if self.config["use_vision"] else None,
            self.config,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
        )
        
        if is_main_process():
            print(f"[validation] epoch={epoch} val_loss={val_loss:.4f}")
            self.val_losses.append(val_loss)
            self.val_epochs.append(epoch)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_step = self.global_step
                print(f"[best-val] new best: {self.best_val_loss:.4f} at step {self.best_step}")
                self._save_best_checkpoint()
    
    def _save_checkpoint(self, epoch: int):
        """Save epoch checkpoint (end of epoch)"""
        self._ensure_config_dumped()
        save_state(
            self.out_dir,
            "latest",
            step=self.global_step,
            epoch=epoch,
            global_step=self.global_step,
            epoch_losses=self.epoch_losses,
            best_loss=self.best_val_loss,
            best_step=self.best_step,
            optim=self.optim,
            sched=self.sched,
            scaler=self.scaler,  # Save GradScaler state for mixed precision
            base=self.base,
            clip_vit=self.runtime.clip_vit if self.config["use_vision"] else None,
            vision_adapter=self.vision_adapter if self.config["use_vision"] else None,
            projector=self.runtime.projector if self.config["use_vision"] else None,
            sam=self.runtime.sam if self.config["use_vision"] else None,  # SAM compression head
            sched_meta=self.sched_meta,
            config=self.config,
            val_losses=self.val_losses,
            val_epochs=self.val_epochs,
            # Metrics history for live plotting (restore on resume)
            caption_metrics_history=self.caption_metrics_history,
            metrics_epochs=self.metrics_epochs,
            # Step-level loss tracking (for detailed plots)
            step_losses=self.step_losses,
            step_loss_steps=self.step_loss_steps,
            # End-of-epoch checkpoint (not mid-epoch)
            step_in_epoch=None,
        )
    
    def _save_step_checkpoint(self, epoch: int, step_in_epoch: int):
        """
        Save mid-epoch checkpoint every N steps for resumability.
        
        This allows resuming training from the last saved step if training
        stops unexpectedly (e.g., preemption, crash, timeout).
        """
        print(f"[checkpoint] Saving step checkpoint at step {self.global_step} (epoch {epoch}, step {step_in_epoch})")
        
        # Save with step-specific tag
        self._ensure_config_dumped()
        save_state(
            self.out_dir,
            "latest",  # We still use "latest" tag but include step_in_epoch for mid-epoch detection
            step=self.global_step,
            epoch=epoch,
            global_step=self.global_step,
            epoch_losses=self.epoch_losses,
            best_loss=self.best_val_loss,
            best_step=self.best_step,
            optim=self.optim,
            sched=self.sched,
            scaler=self.scaler,
            base=self.base,
            clip_vit=self.runtime.clip_vit if self.config["use_vision"] else None,
            vision_adapter=self.vision_adapter if self.config["use_vision"] else None,
            projector=self.runtime.projector if self.config["use_vision"] else None,
            sam=self.runtime.sam if self.config["use_vision"] else None,
            sched_meta=self.sched_meta,
            config=self.config,
            val_losses=self.val_losses,
            val_epochs=self.val_epochs,
            caption_metrics_history=self.caption_metrics_history,
            metrics_epochs=self.metrics_epochs,
            step_in_epoch=step_in_epoch,  # Key field for mid-epoch resume detection
            step_losses=self.step_losses,  # Step-level loss history
            step_loss_steps=self.step_loss_steps,  # Steps where loss was recorded
        )
    
    def _plot_step_loss_curve(self):
        """
        Plot step-level training loss curve.
        
        This provides a detailed view of training progress at each step,
        complementing the epoch-level plots.
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for server environments
        import matplotlib.pyplot as plt
        
        if len(self.step_losses) < 2:
            return  # Need at least 2 points to plot
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Full training loss curve (all steps)
            ax1 = axes[0]
            ax1.plot(self.step_loss_steps, self.step_losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw Loss')
            
            # Add smoothed line (moving average)
            window_size = min(50, len(self.step_losses) // 4) if len(self.step_losses) > 10 else 1
            if window_size > 1:
                smoothed = []
                for i in range(len(self.step_losses)):
                    start = max(0, i - window_size + 1)
                    smoothed.append(sum(self.step_losses[start:i+1]) / (i - start + 1))
                ax1.plot(self.step_loss_steps, smoothed, 'b-', linewidth=1.5, label=f'Smoothed (window={window_size})')
            
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Training Loss (Step {self.global_step})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Recent loss curve (last 1000 steps or all if less)
            ax2 = axes[1]
            recent_n = min(1000, len(self.step_losses))
            recent_steps = self.step_loss_steps[-recent_n:]
            recent_losses = self.step_losses[-recent_n:]
            
            ax2.plot(recent_steps, recent_losses, 'g-', alpha=0.5, linewidth=0.8, label='Raw Loss')
            
            # Smoothed for recent
            window_size_recent = min(20, recent_n // 4) if recent_n > 10 else 1
            if window_size_recent > 1:
                smoothed_recent = []
                for i in range(len(recent_losses)):
                    start = max(0, i - window_size_recent + 1)
                    smoothed_recent.append(sum(recent_losses[start:i+1]) / (i - start + 1))
                ax2.plot(recent_steps, smoothed_recent, 'g-', linewidth=2, label=f'Smoothed (window={window_size_recent})')
            
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.set_title(f'Recent Training Loss (Last {recent_n} steps)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.out_dir / "step_loss_curve.png", dpi=150)
            plt.close(fig)
            
            print(f"[plot] Saved step-level loss curve to {self.out_dir / 'step_loss_curve.png'}")
            
        except Exception as e:
            print(f"[plot] Warning: Failed to generate step loss plot: {e}")
    
    def _save_best_checkpoint(self):
        """Save best model checkpoint"""
        self._ensure_config_dumped()
        def unwrap(model):
            return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        
        
        # save_embedding_layers=True: We resize embeddings for special tokens, so explicitly save them
        unwrap(self.base).save_pretrained(self.out_dir / "qwen2_lora_adapter_best", save_embedding_layers=True)
        
        if self.config["use_vision"]:
            torch.save(unwrap(self.vision_adapter).state_dict(), self.out_dir / "vision_adapter_best.pt")
            torch.save(unwrap(self.runtime.projector).state_dict(), self.out_dir / "projector_best.pt")
            unwrap(self.runtime.clip_vit).save_pretrained(self.out_dir / "clip_lora_adapter_best")
            
            # Save SAM compression head (net_2 and net_3 - the trainable DeepEncoder/VARY layers)
            sam_model = unwrap(self.runtime.sam)
            sam_compression_head_state = {
                name: param.clone() for name, param in sam_model.named_parameters()
                if name.startswith("net_2") or name.startswith("net_3")
            }
            if sam_compression_head_state:
                torch.save(sam_compression_head_state, self.out_dir / "sam_compression_head_best.pt")
            
            print(f"[best-val] saved all vision components (including SAM compression head)")

    def _ensure_config_dumped(self):
        """Persist resolved training config once per run."""
        if self._config_dumped:
            return
        if not is_main_process():
            return
        config_path = self.out_dir / "config.json"
        if config_path.exists():
            self._config_dumped = True
            return
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            print(f"[config] Saved training config to {config_path}")
            self._config_dumped = True
        except Exception as exc:
            print(f"[config] Warning: failed to write {config_path}: {exc}")

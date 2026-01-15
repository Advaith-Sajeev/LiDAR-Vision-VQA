#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-Vision-LLM Training Script :: src/encoder-decoder/train.py 

Entry point for trainig 

"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add the 'src' directory to the Python path BEFORE any local imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from typing import Dict, List, Optional
from training.core import Trainer

# Import from modal-trainer to reuse logic if possible, or reimplement
# We will reimplement the config structure here to be standalone/local-optimized

def get_training_config() -> Dict:
    """
    Get comprehensive training configuration for LOCAL training (V100 16GB Optimized).
    
    This config replicates the 'modal_config.py' structure but with settings 
    tuned for the local environment and specific hardware constraints.
    """
    
    config = {
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                    DEBUG LOGGING TOGGLE                        ║
        # ╚════════════════════════════════════════════════════════════════╝
        "debug_mode": False,      # ← SET TO True TO ENABLE DEBUG LOGGING
        "debug_level": 0,         # ← 0=DISABLED, 1=INFO, 2=DEBUG, 3=TRACE
        "debug_modules": [],      # ← [] = all

        # ──────────────────────────────────────────────────────────────────
        # JSON/JSONL file paths
        # User path: Dataset_subset/external/vision_finetuning_dataset.json
        "caption_json": "/home/j_bindu/fyp-26-grp-38/Dataset_subset/external/vision_finetuning_dataset.json",
        # ──────────────────────────────────────────────────────────────────

        # ──────────────────────────────────────────────────────────────────
        # Output directory for checkpoints, logs, and plots
        "out_dir": "./checkpoints",
        # ──────────────────────────────────────────────────────────────────
        
        # Maximum number of samples to use (None = use all data)
        # Set to 5000 for the current sanity test
        "max_samples": 5000,
        
        
        # ==================== Validation Configuration ====================
        # Phase 1: Run with skip_all_validation=False to validate data
        "skip_all_validation": False,
        
        # Percentage of data to use for validation
        "val_split": 0.1,
        
        # Run validation every N epochs
        "validate_every": 1,

        
        # ==================== Vision Toggle ====================
        "use_vision": True,
        
        
        # ==================== Training Configuration ====================
        "epochs": 20,
        
        # Batch size per GPU
        # With vision tokens + 0.5B LLM on 16GB V100, keep batch small
        "batch_size": 1,
        
        # Gradient accumulation to match larger effective batches
        # 4 accumulation steps * 1 batch size = 4 effective batch size
        "grad_accum": 4,
        
        "num_workers": 12, # HPC has 28 cores; 12 uses more headroom per GPU

        "prefetch_factor": 4,  # Higher prefetch to reduce GPU starvation on HPC
        
        "seed": 42,
        
        # Mixed precision: fp16 for V100 (native support, half memory)
        "mixed_precision": "fp16",
        
        "gradient_checkpointing": True,
        
        "resume": False,  # Disabled for debugging
        
        # Resume strategy: "latest" or "best"
        "resume_from_best": False,  # If True, resume from 'best' checkpoint instead of 'latest'
        
        "save_every_steps": 1000,
        
        "keep_last_n": 3,
        
        "plot_every": 1,
        
        
        # ==================== Inference Configuration ====================
        "inference_sampling_every": 5,
        "inference_samples_n": 40,  # Disabled for benchmarking 
        "inference_caption_json": None,
        
        "inference_max_tokens": 256,
        "inference_temperature": 0.0,
        "inference_do_sample": False,
        "inference_num_beams": 1,
        "inference_batch_size": 1, # Match training batch size
        
        # ==================== Evaluation Metrics ====================
        "eval_caption_bleu4": True,
        "eval_caption_cider": True,
        "eval_caption_spice": False,
        "eval_caption_bertscore": False,
        

        # ==================== Model Configuration ====================
        # User Request: Keep 0.5B model
        "model_id": "Qwen/Qwen2.5-0.5B",
        
        "target_field": "answer",
        "max_ans_toks": 256,
        "system_prompt": "You are an autonomous driving assistant. Analyze the camera images to answer questions about the driving environment, traffic rules, and scene details accurately.",
        

        # ==================== QLoRA / LoRA Configuration ====================
        # User Request: Disable QLoRA, Use Standard LoRA
        "use_qlora": False,
        "tuning_mode": "lora", # "qlora", "lora", or "full"
        
        # QLoRA settings (Ignored when use_qlora=False but good to keep structure)
        "qlora_quant_type": "nf4",
        "qlora_double_quant": True,
        "qlora_compute_dtype": "float32", 
        
        # LoRA Config (All linear layers enabled)
        "llm_lora_r": 4,
        "llm_lora_alpha": 8,
        "llm_lora_dropout": 0.3,
        "llm_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        
        # ==================== CLIP LoRA Configuration ====================
        # CLIP remains frozen for the CLIP-only encoder path
        "clip_lora_enabled": False,
        "clip_lora_r": 2,
        "clip_lora_alpha": 4,
        "clip_lora_dropout": 0.3,
        "clip_lora_target_modules": None, # Auto-detect all compatible layers (unused when disabled)
        
        
        # ==================== Optimization Configuration ====================
        "lr_lora": 3e-4,
        "lr_vision_vat": 5e-4,
        "lr_vision": 5e-4,
        "weight_decay": 0.2,
        "warmup_steps": 3000, # ~5-6% of total steps for 20 epochs with eff. batch 4
        "clip_norm": 1.0,
        
        # ==================== Hardware Optimization ====================
        # Flash Attention not supported on target; leave disabled
        "use_flash_attn": False,
        
        # Torch compile (Disable for debugging/local iteration)
        "use_torch_compile": False,
        
        
        # ==================== nuScenes / DeepEncoder Configuration ====================
        # User path: /home/j_bindu/fyp-26-grp-38/Dataset_subset
        "nu_dataroot": "/home/j_bindu/fyp-26-grp-38/Dataset_subset",
        "nu_version": "v1.0-trainval",
        
        # SAM Checkpoint (Local path or None to download)
        # SAM is not used in the CLIP-only encoder path
        "sam_ckpt": None,
        "auto_download_sam": False,
        
        # Dtype for DeepEncoder: fp16 for V100 (native support)
        "deep_dtype": "float16",
        
        # OpenCLIP: Use openai to save memory if needed, or laion based on preference
        "openclip_pretrained": "openai",
        
        # Timeout (irrelevant locally but keeps schema consistent)
        "timeout": 86400,
    }
    
    # Auto-configure jsons list
    config["jsons"] = [config["caption_json"]]
    
    return config


def setup_output_directory(config: Dict) -> str:
    """
    Setup output directory logic (Interactive for Local).
    
    Matches the 'Smart Resume' logic:
    - If checkpoints exist, prompt to resume.
    - If fresh, create new.
    """
    base_out_dir = Path(config["out_dir"])
    resume = config.get("resume", False)
    
    # Ensure base directory exists
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Look for existing run subdirectories
    run_dirs = sorted(base_out_dir.glob("run_*"), reverse=True)
    
    # Filter for valid runs with checkpoints
    valid_runs = []
    for run_dir in run_dirs:
        if (run_dir / "training_state_latest.pt").exists():
            valid_runs.append(run_dir)
            
    # 2. Interactive Selection if runs exist
    if valid_runs and resume:
        print("\n" + "=" * 80)
        print("RESUME TRAINING: Select a run to resume from")
        print("=" * 80)
        
        # Add "Start New Run" as option 0
        print(f"  [0] START NEW RUN (Create new timestamped directory)")
        
        for idx, run_dir in enumerate(valid_runs, start=1):
            latest_ckpt = run_dir / "training_state_latest.pt"
            # Try to read epoch/step info
            ckpt_info = "checkpoint available"
            try:
                import torch
                # Load cpu to peek
                state = torch.load(latest_ckpt, map_location="cpu") 
                epoch = state.get("epoch", "?")
                step = state.get("global_step", "?")
                ckpt_info = f"epoch={epoch}, step={step}"
            except:
                pass
                
            print(f"  [{idx}] {run_dir.name} ({ckpt_info})")
            
        print("=" * 80)
        
        while True:
            try:
                choice = input(f"Enter your choice [0-{len(valid_runs)}] or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    print("Exiting.")
                    sys.exit(0)
                
                idx = int(choice)
                if idx == 0:
                    # User chose new run
                    break
                elif 1 <= idx <= len(valid_runs):
                    selected_run = valid_runs[idx - 1]
                    print(f"Resuming from: {selected_run}")
                    return str(selected_run)
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
                
    # 3. Create New Run (Fallback or explicit choice)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_out_dir = base_out_dir / f"run_{timestamp}"
    new_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created new run directory: {new_out_dir}")
    
    # If we created a new run, force resume=False in config logic (implied)
    # But effectively, the trainer checks for the file. 
    # If the folder is empty, Trainer starts fresh.
    
    return str(new_out_dir)



def main():
    """
    Main training entry point.
    
    Modify the config in get_training_config() to customize training.
    """
    
    # Get comprehensive configuration
    config = get_training_config()
    
    # Setup output directory based on resume flag
    config["out_dir"] = setup_output_directory(config)
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║              QUICK DEBUG MODE TOGGLE (OVERRIDE)                ║
    # ╚════════════════════════════════════════════════════════════════╝
    # Uncomment any line below to override debug settings:
    # 
    # config["debug_mode"] = True        # Enable debug logging
    # config["debug_level"] = 2          # Set level: 1=INFO, 2=DEBUG, 3=TRACE
    # config["debug_modules"] = []       # Filter: [] = all, ["trainer"] = only trainer
    # 
    # Quick examples:
    #   config["debug_mode"] = True; config["debug_level"] = 2  # Full debug
    #   config["debug_mode"] = True; config["debug_level"] = 3  # Trace mode
    #   config["debug_modules"] = ["trainer", "dataset"]        # Specific modules
    # ────────────────────────────────────────────────────────────────
    
    # ==================== Quick Configuration Overrides ====================
    # Uncomment and modify these for quick experiments without editing the full config
    
    # Quick test (fast, minimal data)
    # config["max_samples"] = 10
    # config["epochs"] = 2
    # config["batch_size"] = 1

    
    # Large model
    # config["model_id"] = "Qwen/Qwen2.5-3B"
    # config["batch_size"] = 1
    # config["grad_accum"] = 8
    # config["fp16"] = True
    
    # High capacity VAT
    # config["vat_queries"] = 768
    # config["vat_layers"] = 6
    # config["vat_heads"] = 12
    # config["vision_queries"] = 2304
    # config["vision_layers"] = 6
    # config["vision_heads"] = 12
    
    # Custom learning rates
    # config["lr_vat"] = 1e-3
    # config["lr_vision_vat"] = 1e-3
    # config["lr_lora"] = 5e-4
    # config["warmup_steps"] = 500
    
    # Custom LoRA configuration
    # Example 1: Only tune attention layers in LLM
    # config["llm_lora_targets"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Example 2: Only tune MLP layers in LLM
    # config["llm_lora_targets"] = ["gate_proj", "up_proj", "down_proj"]
    
    # Example 3: Higher rank LoRA for more capacity
    # config["llm_lora_r"] = 16
    # config["llm_lora_alpha"] = 32
    
    # Example 4: Custom CLIP LoRA targets (attention only)
    # config["clip_lora_target_modules"] = ["qkv_proj", "out_proj"]
    
    # Example 5: Disable CLIP LoRA (freeze CLIP completely)
    # config["clip_lora_enabled"] = False
    
    # Debug mode
    # config["debug_shapes"] = True
    # config["max_samples"] = 5
    # config["epochs"] = 1
    
    

    
    # ==================== Print Configuration ====================
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"\n{'='*30} I/O {'='*30}")

    print(f"JSON files: {config['jsons']}")
    print(f"Output dir: {config['out_dir']}")
    print(f"Max samples: {config['max_samples']}")
    
    print(f"\n{'='*30} Training {'='*30}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['grad_accum']}")
    print(f"Effective batch size: {config['batch_size'] * config['grad_accum']}")
    print(f"Mixed Precision: {config['mixed_precision']}")
    print(f"Resume: {config['resume']} (Mode: {'BEST' if config.get('resume_from_best', False) else 'LATEST'})")
    print(f"Seed: {config['seed']}")
    
    print(f"\n{'='*30} Model {'='*30}")
    print(f"Base model: {config['model_id']}")
    print(f"Use vision: {config['use_vision']}")

    if config['use_vision']:
        print(f"  Vision pipeline enabled")
    print(f"\nLoRA Configuration:")
    print(f"  Tuning Mode: {config.get('tuning_mode', 'qlora')}")
    print(f"  Rank: {config['llm_lora_r']}, Alpha: {config['llm_lora_alpha']}, Dropout: {config['llm_lora_dropout']}")
    print(f"  LLM target modules: {config.get('llm_lora_targets', 'default')}")
    if config['use_vision']:
        clip_targets = config.get('clip_lora_target_modules', None)
        clip_enabled = config.get('clip_lora_enabled', False)
        print(f"  CLIP LoRA enabled: {clip_enabled}")
        if clip_enabled:
            print(f"  CLIP target modules: {clip_targets if clip_targets is not None else 'auto-detect'}")
    
    print(f"\n{'='*30} Optimization {'='*30}")

    print(f"LR LoRA: {config['lr_lora']}")
    if config['use_vision']:
        print(f"LR Vision VAT: {config['lr_vision_vat']}")
        print(f"LR Vision: {config['lr_vision']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Warmup steps: {config['warmup_steps']}")
    print(f"Gradient clip norm: {config['clip_norm']}")
    
    print(f"\n{'='*30} Validation {'='*30}")
    print(f"Val split: {config['val_split']*100:.1f}%")
    print(f"Validate every: {config['validate_every']} epochs")
    print(f"System prompt: {config.get('system_prompt', 'None')}")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    
    # ==================== Create Trainer and Run ====================
    try:
        trainer = Trainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed with error: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()

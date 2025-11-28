# Training Pipeline Code Review - Potential Errors & Bugs

This document provides a comprehensive analysis of potential errors, bugs, and issues found in the LiDAR-Vision-VQA training pipeline. The review covers checkpointing, resuming, evaluation, architecture, dimensions, datatypes, data loading, and hardware utilization.

---

## Table of Contents
1. [Checkpointing & Resuming Issues](#1-checkpointing--resuming-issues)
2. [Architectural & Data Flow Issues](#2-architectural--data-flow-issues)
3. [Dimension Handling Issues](#3-dimension-handling-issues)
4. [Datatype Handling Issues](#4-datatype-handling-issues)
5. [Data Loading Issues](#5-data-loading-issues)
6. [Hardware Utilization Issues](#6-hardware-utilization-issues)
7. [Evaluation & Inference Issues](#7-evaluation--inference-issues)
8. [Memory Management Issues](#8-memory-management-issues)
9. [Distributed Training Issues](#9-distributed-training-issues)
10. [Configuration & Validation Issues](#10-configuration--validation-issues)
11. [Error Handling Issues](#11-error-handling-issues)
12. [Minor Code Quality Issues](#12-minor-code-quality-issues)

---

## 1. Checkpointing & Resuming Issues

### 1.1 **CRITICAL: GradScaler State Not Restored for bf16 Mode**
**Location**: `trainer.py` line 185-188
```python
self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and mixed_prec == 'fp16')
```
**Issue**: The GradScaler is only enabled when `mixed_prec == 'fp16'`, but the scaler state is still saved and attempted to be restored for bf16. While bf16 doesn't need scaling, this inconsistency could cause confusion during resume if mixed_precision mode changes between runs.

**Recommendation**: Add validation to ensure mixed_precision mode doesn't change between checkpoint saves and restores.

---

### 1.2 **Missing LoRA Adapter Validation on Resume**
**Location**: `trainer.py` lines 394-449
**Issue**: When loading LoRA adapters from checkpoint, there's no validation that the LoRA configuration (rank, alpha, target modules) matches between the saved checkpoint and current config. If a user changes `lora_r`, `lora_alpha`, or `lora_target_modules` and resumes, the model will fail silently or produce corrupted weights.

**Code Path**:
```python
adapter_weights_path = lora_path / "adapter_model.safetensors"
# ... loads without checking if config matches
set_peft_model_state_dict(base_model, adapter_state)
```

**Recommendation**: Add LoRA config validation by comparing saved `adapter_config.json` with current config before loading.

---

### 1.3 **Potential Race Condition in Checkpoint Saving**
**Location**: `checkpoints.py` lines 73-113
**Issue**: When saving checkpoints, multiple files are written sequentially (vat_lidar, vat_vision, LLM LoRA, CLIP LoRA, training state). If the process is killed mid-save, the checkpoint will be corrupted.

**Recommendation**: Use atomic write (write to temp files, then rename) or add checkpoint integrity markers.

---

### 1.4 **Optimizer State Device Mismatch on Resume**
**Location**: `trainer.py` line 452
```python
self.optim.load_state_dict(prev_state["optimizer"])
```
**Issue**: The optimizer state is saved on CPU (`map_location="cpu"`) but models may be on GPU. This can cause device mismatch warnings or slowdowns due to CPU-GPU transfers during the first few optimizer steps.

**Recommendation**: After loading optimizer state, call optimizer.state device migration or load with proper device mapping.

---

### 1.5 **No Validation of Checkpoint Epoch/Step Consistency**
**Location**: `trainer.py` lines 459-479
**Issue**: When resuming, the code trusts `prev_state["epoch"]` and `prev_state["global_step"]` without validating they're consistent with `epoch_losses` length or other state.

**Example Corruption**: If `epoch_losses = [0.1, 0.2, 0.3]` but `epoch = 5`, the loss curve will be malformed.

---

### 1.6 **SAM Model Not Saved/Restored Properly**
**Location**: `checkpoints.py` and `trainer.py`
**Issue**: The SAM model's trainable layers (`net_2` and `net_3` - the compression head) have `requires_grad=True` as seen in `deepencoder_infer.py` lines 443-447:
```python
for name, p in self.sam.named_parameters():
    if name.startswith("net_2") or name.startswith("net_3"):
        p.requires_grad = True  # learnable
```
However, the SAM model is NOT saved in checkpoints! Only `vat_lidar`, `vat_vision`, `vision_adapter`, `projector`, and `clip_lora_adapter` are saved.

**Impact**: Training progress on SAM's compression head is lost on resume.

---

## 2. Architectural & Data Flow Issues

### 2.1 **CRITICAL: VisionAdapter Dimension Mismatch with VATVision**
**Location**: `model_setup.py` lines 263-302
**Issue**: 
- `VisionAdapter` outputs `[B, 1536, d_model]` (projects from 2048 to d_model)
- `VATVision` expects input dimension `d_in = d_model` 

But the assertion in `vat_vision.py` line 185 expects:
```python
assert D == self.d_in, f"Expected d_in={self.d_in}, got {D}"
```

This works currently, but the architecture comment in `vat_vision.py` lines 29-34 is misleading:
```python
# Shapes:
#   Input:  [B, N_img_tokens (1536), d_in] where d_in = d_model (from VisionAdapter)
#   After VAT: [B, n_queries (768), d_in]
#   Output: [B, n_queries (768), d_model]
```

The comment says `d_in = d_model` but then says it outputs to a different `d_model`, which is confusing.

---

### 2.2 **Vision Token Order Mismatch Between Training and Validation/Inference**
**Location**: Multiple files
**Issue**: The camera view order is defined in multiple places:
- `dataset.py` line 22-29: `DEFAULT_VIEW_ORDER` (different order)
- `vision_adapter.py` line 24-30: `DEFAULT_VIEW_ORDER` (different order)  
- `deepencoder_infer.py` line 379-386: `DEFAULT_VIEW_ORDER`

```python
# dataset.py / vision_adapter.py:
DEFAULT_VIEW_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",  # Different position!
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
)

# deepencoder_infer.py:
DEFAULT_VIEW_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",  # Different!
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",  # Different position!
)
```

**Impact**: If training uses one order and inference uses another, the vision embeddings will be misaligned per camera, potentially degrading performance.

---

### 2.3 **Inconsistent Special Token Embedding Order**
**Location**: `trainer.py` lines 843-867, `validation.py` lines 143-162
**Issue**: The embedding order is critical but constructed dynamically:
```python
# Training builds: vision_start → vision → vision_end → lidar_start → lidar → lidar_end → text
# But what if only one modality is enabled?
```
The order is correct, but the code relies on list ordering rather than explicit position markers, making it fragile.

---

### 2.4 **VATLiDAR Geometric PE Cache Not Dtype-Aware**
**Location**: `vat_lidar.py` lines 159-164
```python
key = (H, W, device)
if key in self._cache:
    geom, sid = self._cache[key]
    return geom.clone(), sid.clone()
```
**Issue**: The cache key doesn't include dtype. If the model switches from fp32 to bf16 mid-training (via config change), cached tensors will have wrong dtype.

---

### 2.5 **Query Token Scaling Issue**
**Location**: `trainer.py` line 812, `validation.py` line 133
```python
prefix_lidar = self.vat_lidar(bev) * config["prefix_scale"]
```
**Issue**: The `prefix_scale` (default 0.2) is applied after VAT processing. This linear scaling doesn't account for:
1. Different magnitudes between LiDAR and Vision VAT outputs
2. The scale of text embeddings from the LLM

If VAT outputs are already normalized via LayerNorm, this additional scaling may hurt training. If not normalized, the scale relationship to LLM embeddings is unclear.

---

## 3. Dimension Handling Issues

### 3.1 **BEV Feature Shape Assumption**
**Location**: `trainer.py` line 245
```python
probe = ds_full[0]["bev"]
self.c_in = int(probe.shape[0])
```
**Issue**: Assumes all BEV features have the same channel dimension. If the precomputed features have inconsistent shapes (from different PCDet model runs), this will silently use incorrect dimensions.

**Recommendation**: Validate all BEV feature shapes during dataset initialization.

---

### 3.2 **Hardcoded Vision Token Count**
**Location**: `model_setup.py` lines 277-289
```python
n_input_tokens = 6 * 256  # 1536 tokens from VisionAdapter
```
**Issue**: The value 256 is hardcoded and depends on `FIXED_GRID_SIDE = 16` in deepencoder. If grid size changes, this breaks silently.

---

### 3.3 **Vision Compression Factor Calculation Bug**
**Location**: `model_setup.py` lines 281-288
```python
if n_input_tokens % desired_n_queries == 0:
    compression_factor = n_input_tokens // desired_n_queries
else:
    compression_factor = 2  # Fallback
```
**Issue**: The fallback to `compression_factor = 2` can create unexpected query counts. For example:
- If `vision_queries = 100` (not divisible by 1536), compression_factor becomes 2
- Actual queries = 1536 // 2 = 768, not 100

The user might not notice this silent override.

---

### 3.4 **Missing Answer Token Length Validation**
**Location**: `collate.py` lines 46-52
```python
ans_batch = tokenizer(
    answers,
    truncation=True,
    max_length=max_ans_toks,
    ...
)
```
**Issue**: Truncated answers are never logged or flagged. Long answers are silently cut off, potentially losing critical bbox coordinates in grounding tasks.

---

## 4. Datatype Handling Issues

### 4.1 **CRITICAL: Mixed Precision Dtype Inconsistency in Inference**
**Location**: `validation.py` lines 658-659
```python
model_dtype = next(base_model.parameters()).dtype
inputs_embeds = inputs_embeds.to(model_dtype)
```
**Issue**: The code casts `inputs_embeds` to model dtype, but this happens AFTER the embeddings are created. Earlier in the function, embeddings are created under `torch.autocast` context which may produce bf16 or fp16 tensors, then they're cast again. This double casting is inefficient and can cause precision issues.

---

### 4.2 **BEV Feature Float32 Assumption**
**Location**: `dataset.py` line 182
```python
"bev": torch.from_numpy(bev).float(),
```
**Issue**: Forces float32 regardless of training dtype. If using bf16 training, this creates unnecessary dtype conversions in every batch.

---

### 4.3 **Image Tensor Dtype Hardcoded**
**Location**: `deepencoder_infer.py` line 188
```python
t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # Always float32
```
**Issue**: Images are loaded as float32 then converted to target dtype. Should use target dtype directly to save memory.

---

### 4.4 **QLoRA Compute Dtype Not Propagated Everywhere**
**Location**: `model_setup.py` lines 152-164
**Issue**: When using QLoRA, the compute dtype is set for the LLM but VAT models and VisionAdapter may still use a different dtype, causing potential compute overhead from dtype conversions.

---

## 5. Data Loading Issues

### 5.1 **CRITICAL: Dataset Image Loading Failure Not Graceful**
**Location**: `dataset.py` lines 149-163
```python
def _load_camera_images(self, sample_token: str) -> List[Optional[torch.Tensor]]:
    if self.nusc is None:
        return [None] * len(self.view_order)
    image_paths = self._resolve_cam_image_paths(self.nusc, sample_token, self.view_order)
    # No try-except around individual image loads!
```
**Issue**: If `_load_and_preprocess_image` fails for one camera, the entire sample fails. Should return None for that specific camera and continue.

---

### 5.2 **NuScenes Object Not Picklable for Multi-Worker DataLoader**
**Location**: `dataset.py` line 125-128
```python
self.nusc = nusc
```
**Issue**: The NuScenes object is stored in the dataset. With `num_workers > 0`, PyTorch pickles the dataset to send to workers. NuScenes objects may not pickle cleanly (they contain DB connections and file handles), potentially causing "cannot pickle" errors or worker crashes.

**Workaround in Code**: There's a check `if self.nusc is None` but it's unclear if this is sufficient protection.

---

### 5.3 **Token-to-Path Mapping Collision**
**Location**: `data/utils.py` lines 47-48
```python
for npy in r.glob("**/*.npy"):
    token2path.setdefault(npy.stem, str(npy))
```
**Issue**: Uses `setdefault` which keeps the first encountered path. If a sample token has BEV features in multiple directories (e.g., train and val subfolders), only the first is used. No warning is logged.

---

### 5.4 **JSON Loading Memory Issue for Large Datasets**
**Location**: `data/utils.py` lines 11-19
```python
if first == "[":
    for r in json.load(f):  # Loads entire JSON into memory
        yield r
```
**Issue**: For large JSON arrays, the entire file is loaded into memory before iteration. JSONL format properly streams, but JSON array format doesn't.

---

### 5.5 **Missing Image Path Validation in Dataset**
**Location**: `dataset.py` line 155
```python
image_paths = self._resolve_cam_image_paths(self.nusc, sample_token, self.view_order)
```
**Issue**: If `resolve_cam_image_paths` returns paths that don't exist, these are passed to `load_and_preprocess_image` which checks existence. But the check happens in the worker, wasting worker time. Better to validate during dataset construction.

---

### 5.6 **Collate Function Doesn't Handle Missing Images Gracefully**
**Location**: `collate.py` lines 64-70
```python
if load_images and "images" in items[0]:
    batch_images: List[List[Optional[torch.Tensor]]] = []
    for it in items:
        batch_images.append(it["images"])
```
**Issue**: Assumes all items in batch have the same structure. If some items have images and others don't (due to loading failures), this will fail.

---

## 6. Hardware Utilization Issues

### 6.1 **cudnn.benchmark with Variable Input Sizes**
**Location**: `trainer.py` lines 69-71
```python
if config.get("cudnn_benchmark", True) and self.device.type == "cuda":
    torch.backends.cudnn.benchmark = True
```
**Issue**: `cudnn.benchmark = True` assumes fixed input sizes. However:
- BEV features may have variable H×W
- Text sequences have variable lengths
- Image sequences may vary

This can cause slowdowns as cuDNN re-benchmarks for each new shape.

---

### 6.2 **Flash Attention Availability Not Checked Consistently**
**Location**: `vat_blocks.py` lines 67-77
```python
use_flash = (
    _HAS_FLASH_ATTN and 
    q.is_cuda and 
    q.dtype in (torch.float16, torch.bfloat16)
)
```
**Issue**: Flash Attention is checked at runtime for each attention operation. If Flash Attention is available but inputs are float32 (no AMP), it falls back to SDPA every time, logging no warning about the suboptimal path.

---

### 6.3 **Inefficient Gradient Accumulation Implementation**
**Location**: `trainer.py` lines 561-569
```python
if it % self.config["grad_accum"] == 0:
    self._optimizer_step()
    self.global_step += 1
```
**Issue**: The check uses iteration count (`it`) but the DataLoader batch counter starts at 1 (`start=1`). This means:
- `grad_accum=1`: Steps every iteration (correct)
- `grad_accum=2`: Steps on iterations 2, 4, 6... (correct)
- BUT: If the last epoch has fewer batches than `grad_accum`, the final accumulated gradients are never applied.

---

### 6.4 **prefetch_factor Without pin_memory Validation**
**Location**: `trainer.py` lines 262-267
```python
prefetch = self.config.get("prefetch_factor", 2) if num_workers > 0 else None
```
**Issue**: High `prefetch_factor` with `pin_memory=True` can exhaust pinned memory, causing CUDA errors. No validation or warning.

---

### 6.5 **No Memory Optimization for Vision Pipeline**
**Location**: `trainer.py` lines 711-785
**Issue**: The vision pipeline loads all camera images, processes through SAM+CLIP+Projector, then through VisionAdapter and VAT. This keeps all intermediate tensors in memory. No gradient checkpointing is applied to the vision encoding path.

---

### 6.6 **Sync Point After Every Batch**
**Location**: Not explicit, but `torch.autocast` with `device_type=device.type` in training step
**Issue**: Mixed precision training with autocast can introduce sync points. Combined with gradient accumulation, this may not be optimal.

---

## 7. Evaluation & Inference Issues

### 7.1 **CRITICAL: Inference Sampling Uses Different Code Path Than Training**
**Location**: `validation.py` lines 617-654
**Issue**: Inference sampling builds embeddings differently:
- Training: Uses cached special token embeddings (`get_cached_emb`)
- Inference: Uses fresh `emb_token()` calls

While functionally equivalent, any subtle differences in tokenization or embedding lookup could cause train-test mismatch.

---

### 7.2 **Greedy vs Sampling Decision Logic**
**Location**: `validation.py` lines 685-686
```python
use_greedy = not (use_vision_in_inference and use_lidar_in_inference)
```
**Issue**: This enables greedy decoding when EITHER modality is disabled. But a model trained with both modalities will behave differently with greedy vs sampling even when both are enabled. The logic should be:
```python
use_greedy = not use_vision_in_inference and not use_lidar_in_inference  # Only greedy when BOTH disabled
```

---

### 7.3 **Metric Calculation with Empty Predictions**
**Location**: `metrics.py` lines 150-253
**Issue**: If all predictions are empty strings, metrics like BLEU-4 will crash or return undefined behavior. No empty-string filtering before pycocoevalcap.

---

### 7.4 **BertScore Language Hardcoded**
**Location**: `metrics.py` line 248
```python
P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
```
**Issue**: Language is hardcoded to English. If the model is trained on multi-lingual data or non-English nuScenes, scores will be wrong.

---

### 7.5 **Bbox Extraction Regex Too Strict**
**Location**: `metrics.py` lines 21-46
```python
pattern = r'\[([-\d.,\s]+)\]'
```
**Issue**: The regex expects specific format. Model outputs like `[8.4, 10.03, -7.7]` (with spaces after commas) are handled, but `[8.4,  10.03]` (double spaces) or `[8.4 10.03]` (space separator) would fail or produce wrong results.

---

### 7.6 **No Caching of BertScore Model**
**Location**: `metrics.py` line 248
**Issue**: `bert_score()` downloads and loads the RoBERTa model each time metrics are calculated. For frequent inference sampling, this adds significant overhead.

---

## 8. Memory Management Issues

### 8.1 **Special Token Embedding Cache Never Cleared**
**Location**: `trainer.py` line 149
```python
self._special_token_cache = {}
```
**Issue**: The cache grows indefinitely. While small for special tokens, if accidentally used for other tokens, it could leak memory.

---

### 8.2 **Metrics History Unbounded Growth**
**Location**: `trainer.py` lines 153-172
```python
self.caption_metrics_history = {
    "bleu4": [],
    "cider": [],
    ...
}
```
**Issue**: Lists grow with every inference sampling epoch. For very long training runs (100+ epochs), this could consume significant memory. Consider limiting history length or only keeping last N entries.

---

### 8.3 **Vision KV Tensors Not Explicitly Deleted**
**Location**: `trainer.py` lines 711-785
**Issue**: `vision_kv` tensor is created but not explicitly deleted after use. Python GC will handle it, but explicit deletion would free memory faster during the training step.

---

### 8.4 **Pre-encoded Samples Store Tensors in Memory**
**Location**: `validation.py` lines 560-564
```python
sample["_prefix_lidar"] = batch_prefix_lidar[i:i+1]
sample["_prefix_vision"] = batch_prefix_vision[i:i+1]
encoded_samples.append(sample)
```
**Issue**: For inference sampling with many samples, all pre-encoded features are stored in memory before generation. Better to use a generator pattern.

---

## 9. Distributed Training Issues

### 9.1 **Barrier Missing After Checkpoint Save**
**Location**: `trainer.py` line 583
```python
if is_main_process():
    self._save_checkpoint(epoch)
```
**Issue**: Only rank 0 saves checkpoints, but other ranks continue to the next epoch immediately. If checkpoint saving is slow, there's a race condition where rank 0 is still saving while other ranks start epoch N+1.

**Recommendation**: Add `torch.distributed.barrier()` after checkpoint save.

---

### 9.2 **DistributedSampler Epoch Not Synchronized**
**Location**: `trainer.py` line 497
```python
def _set_epoch(self, epoch: int):
    if isinstance(self.sampler_train, (SingleProcessDetSampler, DistributedSampler)):
        self.sampler_train.set_epoch(epoch)
```
**Issue**: This only sets epoch for train sampler. Validation sampler (if using DistributedSampler) doesn't get epoch set, causing potential reproducibility issues in distributed validation.

---

### 9.3 **Model State Not Broadcasted After Resume**
**Location**: `trainer.py` `_try_resume()`
**Issue**: On resume, only rank 0 loads the checkpoint. Other ranks don't receive the restored model state directly - they rely on DDP's initial broadcast which happens before resume. Need explicit state synchronization after resume in multi-GPU scenarios.

---

### 9.4 **Non-Deterministic Operations in Distributed Setting**
**Location**: `trainer.py` lines 717-785 (vision pipeline)
**Issue**: Vision encoding uses `batch_multiview_tokens_from_sample_tokens` which internally may have non-deterministic operations (thread pools in `deepencoder_infer.py`). Different ranks may process batches differently, leading to gradient desync.

---

## 10. Configuration & Validation Issues

### 10.1 **No Schema Validation for Config**
**Location**: `train.py` `get_training_config()`
**Issue**: Config is a plain dict with no type validation. Wrong types (e.g., string instead of int for `epochs`) will fail at runtime, not at config load.

---

### 10.2 **Conflicting Config Options Not Validated**
**Location**: `train.py` and `model_setup.py`
**Issue**: These combinations are not validated:
- `use_vision = False` but `clip_lora_enabled = True` (CLIP LoRA created but never used)
- `use_qlora = True` with `gradient_checkpointing = False` (inefficient)
- `batch_size = 1` with `grad_accum = 1` (very small effective batch)

---

### 10.3 **Model ID Validation Missing**
**Location**: `model_setup.py` line 119
```python
tok = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True)
```
**Issue**: If `model_id` doesn't exist on HuggingFace, error message is cryptic. Should validate model exists early.

---

### 10.4 **Silent Config Override for vision_queries**
**Location**: `model_setup.py` lines 281-288
**Issue**: If `vision_queries` isn't divisible by 1536, it's silently changed. User expects 100 queries but gets 768.

---

### 10.5 **inference_samples_n Must Be Divisible by 4**
**Location**: `validation.py` line 421
```python
assert total_n % 4 == 0, (...)
```
**Issue**: This constraint isn't documented in the config comments. Users will hit an assertion error when trying values like 10, 15, 25.

---

## 11. Error Handling Issues

### 11.1 **Bare Except in Tee Logger**
**Location**: `logging_utils.py` lines 24-31
```python
try:
    self.stdout.write(s)
except Exception:
    pass
```
**Issue**: Silently swallows all exceptions, making debugging difficult if stdout is in a bad state.

---

### 11.2 **Vision Pipeline Fallback Masks Errors**
**Location**: `trainer.py` lines 736-784
```python
except Exception as e:
    debug.warn("trainer", f"Pre-loaded image encoding failed: {e}")
    # vision_kv remains None, will be skipped
```
**Issue**: Vision encoding failures are caught and logged as warnings, but training continues without vision. This could lead to a model that only learned from LiDAR, which user might not notice.

---

### 11.3 **Missing File Exception in save_val_inference_samples**
**Location**: `validation.py` line 271
```python
bev_path = ds_val.dataset.token2path.get(sample_token)
if not bev_path:
    continue
```
**Issue**: If token2path returns None, sample is silently skipped. No logging of how many samples were skipped.

---

### 11.4 **Unsafe PEFT Import**
**Location**: `trainer.py` lines 406-418
```python
from safetensors.torch import load_file
```
**Issue**: Import happens inside the resume block. If safetensors isn't installed, error happens only on resume, not at startup.

---

## 12. Minor Code Quality Issues

### 12.1 **Duplicate DEFAULT_VIEW_ORDER Definitions**
**Location**: `dataset.py`, `vision_adapter.py`, `deepencoder_infer.py`
**Issue**: Same constant defined in 3 places with different orderings. Should be centralized.

---

### 12.2 **Magic Numbers Without Named Constants**
**Location**: Various
- `256` tokens per view (should be `TOKENS_PER_VIEW`)
- `2048` projector dimension (should be `PROJECTOR_DIM`)
- `6` camera views (should use `NUM_VIEWS`)
- `1536` total vision tokens (should be computed from constants)

---

### 12.3 **Inconsistent Debug Logger Import**
**Location**: Multiple files
**Issue**: Some files use:
```python
try:
    from ..utils import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
```
Others use:
```python
from deepencoder.debug import debug
```

This inconsistency makes debugging harder.

---

### 12.4 **Unused Import in train.py**
**Location**: `train.py` line 648-649
```python
import time
time.sleep(3)
```
**Issue**: `time` is imported only for a warning sleep. Should be conditional.

---

### 12.5 **Comments Reference Incorrect Line Numbers**
**Location**: Throughout codebase
**Issue**: Many inline comments reference specific behaviors but don't stay updated when code changes.

---

### 12.6 **TODO Comments Still Present**
**Location**: `train.py` line 7
```python
# TODO :: improve doc string
```
**Issue**: Indicates incomplete documentation.

---

## Summary of Critical Issues

| Priority | Issue | Impact |
|----------|-------|--------|
| 🔴 CRITICAL | SAM model trainable layers not saved | Training progress lost on resume |
| 🔴 CRITICAL | Vision token order mismatch | Train-test performance mismatch |
| 🔴 CRITICAL | LoRA config not validated on resume | Silent model corruption |
| 🔴 CRITICAL | Inference uses different embedding path | Potential train-test gap |
| 🟠 HIGH | Greedy decoding logic inverted | Wrong decoding during ablations |
| 🟠 HIGH | vision_queries silently changed | Unexpected model capacity |
| 🟠 HIGH | NuScenes not picklable for workers | Multi-worker DataLoader crash |
| 🟠 HIGH | Missing distributed barrier | Race condition on checkpoint |
| 🟡 MEDIUM | GradScaler state for bf16 | Confusion on resume |
| 🟡 MEDIUM | BEV dtype always float32 | Inefficient memory usage |
| 🟡 MEDIUM | Final accumulated gradients lost | Training instability at epoch end |

---

## Recommendations

1. **Immediate**: Fix SAM model saving/loading for trainable compression head
2. **Immediate**: Centralize camera view ordering constants
3. **Immediate**: Add LoRA config validation on resume
4. **High Priority**: Fix greedy decoding logic in inference
5. **High Priority**: Add distributed barrier after checkpoint save
6. **Medium Priority**: Add config schema validation
7. **Medium Priority**: Optimize BEV loading dtype
8. **Low Priority**: Clean up duplicate constants and magic numbers

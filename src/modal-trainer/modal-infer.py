#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-Vision-LLM Modal Inference Runner :: src/modal-trainer/modal-infer.py

Runs the same validation-time inference sampling loop (with identical logs)
that Trainer uses between epochs, but as a standalone Modal job. After the
predictions are generated it also archives every sample's BEV .npy, six camera
images, and per-sample metadata JSON for offline inspection.

Usage examples:
    modal run src/modal-trainer/modal-infer.py                       # default config
    modal run src/modal-trainer/modal-infer.py --sample-count 24      # override sample count
    modal run src/modal-trainer/modal-infer.py --dataset-mode both    # override dataset mode
"""

from datetime import datetime
from pathlib import Path

import modal


def _to_jsonable(obj):
    """Convert numpy/scalar outputs to vanilla Python so Modal can serialize without numpy locally."""
    try:
        import numpy as np  # modal container always has numpy; local deserialization then stays pure python
    except ImportError:
        np = None

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    return obj


app = modal.App("lidar-vision-training")
volume = modal.Volume.from_name("lidar-llm", create_if_missing=False)
FLASH_ATTN_CACHE = Path("/data/build_cache/flash-attn")

# ----------------------------------------------------------------------------
# IMAGE DEFINITION: match training/test environment
# ----------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Asia/Kolkata"})
    .run_commands(
        "ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime",
        "echo Asia/Kolkata > /etc/timezone",
    )
    .apt_install(
        "git", "wget", "build-essential", "ninja-build",
        "clang", "llvm-dev", "libopencv-dev", "pkg-config",
    )
    .run_commands(
        "pip3 install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install("spconv-cu126")
    .pip_install(
        "llvmlite", "numba",
    )
    .pip_install(
        "tensorboardX", "easydict", "pyyaml",
        "scikit-image", "tqdm", "SharedArray", "opencv-python", "pyquaternion",
    )
    .add_local_dir(
        local_path="./src/lidar-encoder",
        remote_path="/tmp/lidar-encoder",
        copy=True,
    )
    .run_commands(
        "cd /tmp/lidar-encoder && pip install -e . --no-build-isolation",
        gpu="any",
    )
    .pip_install(
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "open-clip-torch>=2.20.0",
        "pillow>=10.0.0",
        "pycocotools>=2.0.6",
        "pycocoevalcap>=1.2",
        "bert-score>=0.3.13",
        "nuscenes-devkit>=1.1.0",
        "matplotlib>=3.7.0",
        "numpy>=2.0.0,<2.3.0",
        "optimum>=1.15.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "nltk>=3.8.1",
        "sacrebleu>=2.3.0",
        "rouge-score>=0.1.2",
    )
    .run_commands(
        "python -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"wordnet\"); nltk.download(\"omw-1.4\")'"
    )
    .add_local_dir(
        local_path="./src",
        remote_path="/root/src",
        copy=False,
    )
)


def _resolve_latest_run(checkpoints_root: Path) -> Path:
    runs = [d for d in checkpoints_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not runs:
        raise FileNotFoundError(
            f"No run_* directories found under {checkpoints_root}. Train a model before running inference."
        )
    runs.sort(key=lambda p: p.name)
    return runs[-1]


def _load_best_step(run_dir: Path) -> int:
    import torch

    state_path = run_dir / "training_state_latest.pt"
    if not state_path.exists():
        return 0
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        return int(state.get("global_step", 0))
    except Exception as exc:
        print(f"[modal_infer] Warning: failed to read {state_path.name}: {exc}")
        return 0


def _ensure_flash_attn_cached() -> None:
    """Install flash-attn from a cached wheel on the Modal volume, building once if absent."""
    import subprocess
    import sys

    cache_dir = FLASH_ATTN_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _install_from_wheel(wheel_path: Path) -> None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", str(wheel_path), "--no-build-isolation"],
            check=True,
        )

    wheels = sorted(cache_dir.glob("flash_attn*.whl"))
    if wheels:
        wheel = wheels[-1]
        print(f"[modal_infer] Installing cached flash-attn wheel → {wheel.name}")
        _install_from_wheel(wheel)
        return

    print("[modal_infer] flash-attn wheel cache empty; compiling once for reuse")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "flash-attn",
            "--no-build-isolation",
            "-w",
            str(cache_dir),
        ],
        check=True,
    )
    wheels = sorted(cache_dir.glob("flash_attn*.whl"))
    if not wheels:
        raise RuntimeError("flash-attn wheel build failed; no wheel found in cache")
    _install_from_wheel(wheels[-1])


@app.function(
    gpu="L4",
    cpu=32.0,
    memory=131072,
    image=image,
    volumes={"/data": volume},
    timeout=7200,
    max_inputs=1,
)
def run_modal_inference(
    sample_override: int = 0,
    dataset_mode_override: str = "",
    epoch_label: str = "modal",
):
    import json
    import os
    import sys
    import torch

    print("=" * 80)
    print("🚀 MODAL INFERENCE (Random Sampling)")
    print("=" * 80)

    model_cache_dir = "/data/model_cache"
    os.makedirs(f"{model_cache_dir}/huggingface", exist_ok=True)
    os.makedirs(f"{model_cache_dir}/torch", exist_ok=True)
    os.makedirs(f"{model_cache_dir}/clip", exist_ok=True)
    os.makedirs(f"{model_cache_dir}/sam", exist_ok=True)

    os.environ["HF_HOME"] = f"{model_cache_dir}/huggingface"
    os.environ["HF_HUB_CACHE"] = f"{model_cache_dir}/huggingface"
    os.environ["TORCH_HOME"] = f"{model_cache_dir}/torch"
    os.environ["XDG_CACHE_HOME"] = model_cache_dir
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    _ensure_flash_attn_cached()

    src_path = "/root/src"
    encoder_decoder_path = "/root/src/encoder-decoder"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if encoder_decoder_path not in sys.path:
        sys.path.insert(0, encoder_decoder_path)

    from configs.modal_config import get_modal_training_config
    from inference import ModelLoader, InferenceEngine
    from inference.utils import save_inference_artifacts
    from training.data.utils import collect_feature_tokens
    from training.core.validation import run_inference_sampling

    try:
        config = get_modal_training_config()
        if sample_override > 0:
            print(f"[modal_infer] Overriding inference_samples_n → {sample_override}")
            config["inference_samples_n"] = sample_override
        if dataset_mode_override:
            print(f"[modal_infer] Overriding dataset_mode → {dataset_mode_override}")
            config["dataset_mode"] = dataset_mode_override

        checkpoints_root = Path(config["checkpoints_root"]).resolve()
        run_dir = _resolve_latest_run(checkpoints_root)
        print(f"[modal_infer] Using checkpoint run: {run_dir}")

        loader = ModelLoader(str(run_dir), fallback_config=config)
        models = loader.load_all(c_in=config.get("c_in"))
        engine = InferenceEngine(models)

        runtime_config = models["config"]
        override_keys = [
            "dataset_mode",
            "inference_caption_json",
            "inference_grounding_json",
            "inference_samples_n",
            "inference_max_tokens",
            "inference_temperature",
            "inference_top_p",
            "inference_top_k",
            "inference_do_sample",
            "inference_num_beams",
            "inference_batch_size",
            "inference_use_vision",
            "inference_use_lidar",
            "inference_use_system",
            "system_prompt",
            "target_field",
            "validate_image_paths",
            "bev_validation_workers",
            "use_vision",
        ]
        for key in override_keys:
            if key in config:
                runtime_config[key] = config[key]

        advanced_metric_flags = {
            "eval_caption_rougel": True,
            "eval_caption_meteor": True,
            "eval_det_area_rougel": True,
            "eval_det_area_meteor": True,
            "eval_det_object_rougel": True,
            "eval_det_object_meteor": True,
        }
        runtime_config.update(advanced_metric_flags)

        feature_dirs = config.get("feature_dirs", [])
        token2path = collect_feature_tokens(feature_dirs)
        if not token2path:
            raise RuntimeError(f"No BEV features found in {feature_dirs}")
        print(f"[modal_infer] Indexed {len(token2path)} BEV feature files")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = run_dir / "modal_inference" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = output_dir / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        print(f"[modal_infer] Output directory: {output_dir}")

        try:
            rel_output_dir = output_dir.relative_to(Path("/data"))
            rel_output_dir_str = rel_output_dir.as_posix()
        except ValueError:
            rel_output_dir_str = ""

        best_step = _load_best_step(run_dir)
        amp_mode = config.get("mixed_precision", "no").lower()
        amp_dtype = torch.bfloat16 if amp_mode == "bf16" else (
            torch.float16 if amp_mode == "fp16" else torch.float32
        )
        use_amp = amp_mode in ("bf16", "fp16")

        metrics = run_inference_sampling(
            base=models["base_model"],
            vat_lidar=models["vat_lidar"],
            vat_vision=models["vat_vision"],
            vision_adapter=models["vision_adapter"],
            runtime=models["runtime"],
            nusc=models["nusc"],
            tok=models["tokenizer"],
            config=runtime_config,
            out_dir=output_dir,
            epoch=epoch_label,
            device=models["device"],
            token2path=token2path,
            best_step=best_step,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        results_path = output_dir / f"inference_sampling_epoch{epoch_label}.json"
        summary_path = output_dir / f"metrics_summary_epoch{epoch_label}.json"
        if not results_path.exists():
            print(f"[modal_infer] Warning: {results_path} not found; skipping artifact export")
        else:
            with open(results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            samples = payload.get("samples", [])
            print(f"[modal_infer] Archiving {len(samples)} samples to {artifact_root}")
            for idx, sample in enumerate(samples, 1):
                sample_payload = dict(sample)
                sample_payload.setdefault("sequence_id", idx)
                save_inference_artifacts(
                    artifact_dir=str(artifact_root),
                    sample_payload=sample_payload,
                    bev_path=token2path.get(sample.get("sample_token")),
                    engine=engine,
                    vision_requested=runtime_config.get("inference_use_vision", True),
                )
            print(f"[modal_infer] ✓ Artifact export complete")

        return {
            "metrics": _to_jsonable(metrics),
            "results_file": str(results_path),
            "metrics_file": str(summary_path),
            "artifacts_dir": str(artifact_root),
            "output_dir": str(output_dir),
            "volume_rel_output_dir": rel_output_dir_str,
        }
    finally:
        volume.commit()


@app.local_entrypoint()
def main(sample_count: int = 0, dataset_mode: str = "", epoch_label: str = "modal"):
    """Local convenience entrypoint for quick deployments."""
    result = run_modal_inference.remote(
        sample_override=sample_count,
        dataset_mode_override=dataset_mode,
        epoch_label=epoch_label,
    )
    print("=" * 80)
    print("MODAL INFERENCE SUMMARY")
    print("=" * 80)
    print(result)

    _ = (result or {}).get("volume_rel_output_dir")  # kept for future automation hooks


if __name__ == "__main__":
    print("Run with: modal run src/modal-trainer/modal-infer.py")

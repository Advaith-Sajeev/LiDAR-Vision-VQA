#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-Vision-LLM Inference Script
Run inference on trained models with various options
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import sys

from inference import ModelLoader, InferenceEngine
from inference.utils import (
    load_qa_pairs,
    save_predictions,
    calculate_metrics,
    format_output,
    save_inference_artifacts,
    select_random_samples,
)
from training.utils import calculate_sample_level_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with LiDAR-Vision-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Interactive mode (single question)
  python infer.py --checkpoint checkpoints_vat --bev bev_feats/train/sample_token.npy --question "What is ahead?"
  
  # With vision
  python infer.py --checkpoint checkpoints_vat --bev bev_feats/train/sample_token.npy --sample-token abc123 --question "What is ahead?"
  
  # Batch mode (process JSON file)
  python infer.py --checkpoint checkpoints_vat --json Dataset_subset/external/nuCaption.json --feature-dirs bev_feats/train --output predictions.json
  
  # Custom generation parameters
  python infer.py --checkpoint checkpoints_vat --bev sample.npy --question "..." --temperature 0.9 --top-p 0.95 --max-tokens 128
  
  # Greedy decoding
  python infer.py --checkpoint checkpoints_vat --bev sample.npy --question "..." --no-sample --num-beams 3
  
  # Ablation studies (disable components)
  python infer.py --checkpoint checkpoints_vat --bev sample.npy --question "..." --no-vision  # LiDAR only
  python infer.py --checkpoint checkpoints_vat --bev sample.npy --sample-token abc123 --question "..." --no-lidar  # Vision only
  python infer.py --checkpoint checkpoints_vat --bev sample.npy --question "..." --no-system  # No system prompt
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    
    # Input mode: interactive or batch
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--json",
        type=str,
        help="JSON/JSONL file with QA pairs (batch mode)"
    )
    input_group.add_argument(
        "--question", "-q",
        type=str,
        help="Question to answer (interactive mode)"
    )
    
    # Interactive mode arguments
    parser.add_argument(
        "--bev",
        type=str,
        help="Path to BEV feature .npy file (required for interactive mode)"
    )
    parser.add_argument(
        "--sample-token",
        type=str,
        help="nuScenes sample token (optional, for vision)"
    )
    
    # Batch mode arguments
    parser.add_argument(
        "--feature-dirs",
        type=str,
        nargs="+",
        help="Directories containing BEV features (required for batch mode)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for predictions (batch mode)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (batch mode)"
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="answer",
        help="Field name for ground truth answers in JSON (default: 'answer')"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val"),
        default="train",
        help="Dataset split to sample from when --sample-random is set (default: train)"
    )
    parser.add_argument(
        "--sample-random",
        type=int,
        default=0,
        metavar="N",
        help="Randomly pick N samples from the selected split and run sequential inference"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="",
        help="Ground truth answer for interactive mode (saved with artifacts)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or None for auto)"
    )
    parser.add_argument(
        "--c-in",
        type=int,
        default=None,
        help="Number of input channels for BEV (auto-detect if not provided)"
    )
    
    # System prompt
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="System prompt to prepend to all questions (default: empty)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate (default: 64)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold (default: 0.9)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling threshold (default: 50)"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search (default: 1)"
    )
    
    # Inference toggles (for ablation studies)
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision tokens in inference (ablation study)"
    )
    parser.add_argument(
        "--no-lidar",
        action="store_true",
        help="Disable LiDAR tokens in inference (ablation study)"
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Disable system prompt in inference (ablation study)"
    )
    
    # Display options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except answers"
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory to store per-sample artifacts (files + metadata)"
    )
    
    return parser.parse_args()


def interactive_mode(args, engine: InferenceEngine):
    """Run inference in interactive mode (single question)."""
    if not args.bev and not args.no_lidar:
        print("Error: --bev is required unless --no-lidar is set", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print("=" * 80)
        print("INTERACTIVE INFERENCE")
        print("=" * 80)
        print(f"BEV file: {args.bev if not args.no_lidar else 'None (LiDAR disabled)'}")
        print(f"Sample token: {args.sample_token or 'None'}")
        print(f"Question: {args.question}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Temperature: {args.temperature}")
        print("=" * 80)
        print("\nGenerating answer...\n")
    
    # Generate answer
    answer = engine.generate(
        question=args.question,
        bev=None if args.no_lidar else args.bev,
        sample_token=args.sample_token,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sample,
        num_beams=args.num_beams,
        use_vision=not args.no_vision,
        use_lidar=not args.no_lidar,
        use_system=not args.no_system,
    )
    
    if args.quiet:
        print(answer)
    else:
        print(format_output(
            question=args.question,
            prediction=answer,
            sample_token=args.sample_token,
        ))

    sample_payload = {
        "sample_token": args.sample_token,
        "question": args.question,
        "prediction": answer,
        "ground_truth": args.ground_truth or "",
        "dataset_type": engine.config.get("dataset_mode", "caption"),
    }
    sample_payload["metrics"] = calculate_sample_level_metrics(sample_payload, engine.config)

    save_inference_artifacts(
        artifact_dir=args.artifact_dir,
        sample_payload=sample_payload,
        bev_path=None if args.no_lidar else args.bev,
        engine=engine,
        sequence_id=1,
        vision_requested=not args.no_vision,
    )


def batch_mode(args, engine: InferenceEngine):
    """Run inference in batch mode (process JSON file)."""
    if not args.feature_dirs and not args.no_lidar:
        print("Error: --feature-dirs is required unless --no-lidar is set", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print("=" * 80)
        print("BATCH INFERENCE")
        print("=" * 80)
        print(f"Input JSON: {args.json}")
        print(f"Feature dirs: {args.feature_dirs}")
        print(f"Output: {args.output or 'stdout'}")
        print(f"Max samples: {args.max_samples or 'all'}")
        print("=" * 80)
    
    # Load QA pairs
    qa_pairs = load_qa_pairs(args.json)
    if args.verbose:
        print(f"\n[batch] Loaded {len(qa_pairs)} QA pairs")
    
    # Build token -> path mapping
    token2path = {}
    if args.feature_dirs:
        for feat_dir in args.feature_dirs:
            feat_dir = Path(feat_dir)
            for npy_file in feat_dir.glob("*.npy"):
                token = npy_file.stem
                token2path[token] = str(npy_file)
        if args.verbose:
            print(f"[batch] Found {len(token2path)} BEV features")
    
    # Filter samples
    valid_samples = []
    for qa in qa_pairs:
        token = qa.get("sample_token")
        if not token:
            continue
        if not args.no_lidar and token not in token2path:
            continue
        
        question = qa.get("question", "").strip()
        if not question:
            continue
        
        valid_samples.append({
            "sample_token": token,
            "question": question,
            "ground_truth": qa.get(args.target_field, "").strip(),
            "bev_path": None if args.no_lidar else token2path[token],
            "dataset_type": qa.get("dataset_type", "caption"),
        })
    
    if args.max_samples and len(valid_samples) > args.max_samples:
        valid_samples = valid_samples[:args.max_samples]

    sampling_info = None
    selected_samples = valid_samples
    if args.sample_random > 0:
        val_split = float(engine.config.get("val_split", 0.05))
        seed = int(engine.config.get("seed", 42))
        selected_samples, sampling_info = select_random_samples(
            valid_samples,
            count=args.sample_random,
            split=args.split,
            seed=seed,
            val_split=val_split,
        )
        if not selected_samples:
            print(
                f"[batch] No samples available for split='{args.split}' with current filters.",
                file=sys.stderr,
            )
            return

    if not args.quiet:
        print(f"\n[batch] Processing {len(selected_samples)} samples\n")
        if sampling_info:
            print(
                f"[batch] Random sample mode: split={args.split} | requested={sampling_info['requested']} | "
                f"available={sampling_info['available']} | selected={sampling_info['selected']}"
            )
    
    # Run inference
    predictions = []
    iterator = tqdm(selected_samples, desc="Inference") if not args.quiet else selected_samples
    
    processed_count = 0
    for sample in iterator:
        try:
            answer = engine.generate(
                question=sample["question"],
                bev=sample["bev_path"],
                sample_token=sample["sample_token"] if engine.use_vision else None,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=not args.no_sample,
                num_beams=args.num_beams,
                use_vision=not args.no_vision,
                use_lidar=not args.no_lidar,
                use_system=not args.no_system,
            )
            
            prediction_entry = {
                "sample_token": sample["sample_token"],
                "question": sample["question"],
                "prediction": answer,
                "ground_truth": sample["ground_truth"],
                "dataset_type": sample.get("dataset_type", "caption"),
            }
            predictions.append(prediction_entry)

            processed_count += 1
            sample_payload = dict(prediction_entry)
            sample_payload["metrics"] = calculate_sample_level_metrics(sample_payload, engine.config)
            sample_payload["sequence_id"] = processed_count

            save_inference_artifacts(
                artifact_dir=args.artifact_dir,
                sample_payload=sample_payload,
                bev_path=sample["bev_path"],
                engine=engine,
                sequence_id=processed_count,
                vision_requested=not args.no_vision,
            )
            
            if args.verbose:
                print(format_output(
                    question=sample["question"],
                    prediction=answer,
                    ground_truth=sample["ground_truth"],
                    sample_token=sample["sample_token"],
                ))
        
        except Exception as e:
            if args.verbose:
                print(f"[batch] Error processing {sample['sample_token']}: {e}")
            continue
    
    # Calculate metrics
    if not args.quiet:
        metrics = calculate_metrics(predictions)
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Processed: {metrics.get('num_samples', 0)} samples")
        print(f"Avg prediction length: {metrics.get('avg_prediction_length', 0):.1f} tokens")
        if "avg_ground_truth_length" in metrics:
            print(f"Avg ground truth length: {metrics.get('avg_ground_truth_length', 0):.1f} tokens")
        print("=" * 80)
    
    # Save predictions
    if args.output:
        output_format = "jsonl" if args.output.endswith(".jsonl") else "json"
        save_predictions(predictions, args.output, format=output_format)
    else:
        # Print to stdout
        if args.quiet:
            for pred in predictions:
                print(pred["prediction"])
        else:
            print("\nPredictions:")
            print(json.dumps(predictions, indent=2))


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.sample_random > 0 and not args.json:
            print("Error: --sample-random requires --json dataset input", file=sys.stderr)
            sys.exit(1)
        
        # Load models
        if not args.quiet:
            print("\n" + "=" * 80)
            print("LOADING MODELS")
            print("=" * 80)
        
        loader = ModelLoader(args.checkpoint, device=args.device)
        models = loader.load_all(c_in=args.c_in)
        
        # Override system_prompt if provided
        if args.system_prompt:
            models["config"]["system_prompt"] = args.system_prompt
        
        # Create inference engine
        engine = InferenceEngine(models)
        
        if not args.quiet:
            print("\n")
        
        # Run inference
        if args.question:
            interactive_mode(args, engine)
        else:
            batch_mode(args, engine)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

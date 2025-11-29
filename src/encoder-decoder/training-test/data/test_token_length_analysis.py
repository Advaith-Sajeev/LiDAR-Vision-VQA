"""
Token Length Distribution Analysis

This test file analyzes the token length distribution of inputs (questions) and 
outputs (answers) in the dataset. It helps identify potential truncation issues
where max_ans_toks may be too short, causing loss of critical information like
bbox coordinates in grounding tasks.

Issue 3.4: Missing Answer Token Length Validation
- Truncated answers are never logged or flagged
- Long answers are silently cut off, potentially losing critical bbox coordinates

Usage:
    python test_token_length_analysis.py --json-paths /path/to/qa.json --tokenizer Qwen/Qwen2.5-0.5B-Instruct
    
    # With visualization (requires matplotlib)
    python test_token_length_analysis.py --json-paths /path/to/qa.json --plot
    
    # Check against specific max_ans_toks setting
    python test_token_length_analysis.py --json-paths /path/to/qa.json --max-ans-toks 128
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not installed. Plotting disabled. Install with: pip install matplotlib")


def load_json_any(path: str) -> List[Dict]:
    """Load JSON file (handles both JSON array and JSONL formats)"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        content = f.read().strip()
        
    # Try as JSON array first
    if content.startswith('['):
        return json.loads(content)
    
    # Otherwise treat as JSONL
    rows = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def analyze_token_lengths(
    json_paths: List[str],
    tokenizer,
    target_field: str = "answer_lidar",
    system_prompt: str = "",
) -> Dict:
    """
    Analyze token lengths for questions and answers in the dataset.
    
    Args:
        json_paths: List of paths to JSON/JSONL files
        tokenizer: HuggingFace tokenizer
        target_field: Field containing the answer text
        system_prompt: System prompt to use in chat template
        
    Returns:
        Dictionary with analysis results
    """
    if not system_prompt:
        system_prompt = (
            "You are an expert autonomous driving assistant. Analyze the 3D LiDAR point cloud "
            "and camera images to understand the driving scene. Provide accurate, concise "
            "descriptions of objects, their locations, distances, and spatial relationships."
        )
    
    question_lengths = []
    answer_lengths = []
    prompt_lengths = []  # Full prompt with chat template
    
    # Track per-file stats
    file_stats = defaultdict(lambda: {"questions": [], "answers": []})
    
    # Track samples with very long answers (potential truncation victims)
    long_answer_samples = []
    
    total_samples = 0
    skipped_samples = 0
    
    for json_path in json_paths:
        path_name = Path(json_path).name
        print(f"[analyze] Processing: {path_name}")
        
        try:
            rows = load_json_any(json_path)
        except Exception as e:
            print(f"[ERROR] Failed to load {json_path}: {e}")
            continue
        
        for row in rows:
            total_samples += 1
            
            question = row.get("question", "")
            
            # Try multiple answer fields in priority order
            answer = None
            for field in [target_field, "answer", "answer_lidar", "answer_vision"]:
                if field in row and row[field]:
                    answer = row[field]
                    break
            
            if not question or not answer:
                skipped_samples += 1
                continue
            
            # Tokenize question alone
            q_tokens = tokenizer(question, add_special_tokens=False)["input_ids"]
            question_lengths.append(len(q_tokens))
            file_stats[path_name]["questions"].append(len(q_tokens))
            
            # Tokenize answer alone
            a_tokens = tokenizer(answer, add_special_tokens=True)["input_ids"]
            answer_lengths.append(len(a_tokens))
            file_stats[path_name]["answers"].append(len(a_tokens))
            
            # Full prompt with chat template
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            full_prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            p_tokens = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
            prompt_lengths.append(len(p_tokens))
            
            # Track long answers
            if len(a_tokens) > 100:  # Arbitrary threshold for "long"
                long_answer_samples.append({
                    "file": path_name,
                    "sample_token": row.get("sample_token", "N/A"),
                    "answer_tokens": len(a_tokens),
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                })
    
    # Convert to numpy for statistics
    question_lengths = np.array(question_lengths)
    answer_lengths = np.array(answer_lengths)
    prompt_lengths = np.array(prompt_lengths)
    
    return {
        "total_samples": total_samples,
        "analyzed_samples": len(question_lengths),
        "skipped_samples": skipped_samples,
        "question": {
            "min": int(question_lengths.min()) if len(question_lengths) > 0 else 0,
            "max": int(question_lengths.max()) if len(question_lengths) > 0 else 0,
            "mean": float(question_lengths.mean()) if len(question_lengths) > 0 else 0,
            "std": float(question_lengths.std()) if len(question_lengths) > 0 else 0,
            "median": float(np.median(question_lengths)) if len(question_lengths) > 0 else 0,
            "p95": float(np.percentile(question_lengths, 95)) if len(question_lengths) > 0 else 0,
            "p99": float(np.percentile(question_lengths, 99)) if len(question_lengths) > 0 else 0,
            "lengths": question_lengths,
        },
        "answer": {
            "min": int(answer_lengths.min()) if len(answer_lengths) > 0 else 0,
            "max": int(answer_lengths.max()) if len(answer_lengths) > 0 else 0,
            "mean": float(answer_lengths.mean()) if len(answer_lengths) > 0 else 0,
            "std": float(answer_lengths.std()) if len(answer_lengths) > 0 else 0,
            "median": float(np.median(answer_lengths)) if len(answer_lengths) > 0 else 0,
            "p95": float(np.percentile(answer_lengths, 95)) if len(answer_lengths) > 0 else 0,
            "p99": float(np.percentile(answer_lengths, 99)) if len(answer_lengths) > 0 else 0,
            "lengths": answer_lengths,
        },
        "prompt": {
            "min": int(prompt_lengths.min()) if len(prompt_lengths) > 0 else 0,
            "max": int(prompt_lengths.max()) if len(prompt_lengths) > 0 else 0,
            "mean": float(prompt_lengths.mean()) if len(prompt_lengths) > 0 else 0,
            "std": float(prompt_lengths.std()) if len(prompt_lengths) > 0 else 0,
            "median": float(np.median(prompt_lengths)) if len(prompt_lengths) > 0 else 0,
            "p95": float(np.percentile(prompt_lengths, 95)) if len(prompt_lengths) > 0 else 0,
            "p99": float(np.percentile(prompt_lengths, 99)) if len(prompt_lengths) > 0 else 0,
            "lengths": prompt_lengths,
        },
        "file_stats": dict(file_stats),
        "long_answer_samples": sorted(long_answer_samples, key=lambda x: -x["answer_tokens"])[:20],
    }


def check_truncation_risk(
    stats: Dict,
    max_ans_toks: int,
    warn_threshold: str = "mean+std"
) -> Dict:
    """
    Check if max_ans_toks is sufficient for the dataset.
    
    Args:
        stats: Output from analyze_token_lengths
        max_ans_toks: The max_ans_toks value from config
        warn_threshold: "mean+std", "p95", or "p99"
        
    Returns:
        Dictionary with truncation risk analysis
    """
    answer_stats = stats["answer"]
    
    # Calculate threshold based on warn_threshold
    if warn_threshold == "mean+std":
        threshold = answer_stats["mean"] + answer_stats["std"]
        threshold_name = "mean + std"
    elif warn_threshold == "p95":
        threshold = answer_stats["p95"]
        threshold_name = "95th percentile"
    elif warn_threshold == "p99":
        threshold = answer_stats["p99"]
        threshold_name = "99th percentile"
    else:
        threshold = answer_stats["mean"] + answer_stats["std"]
        threshold_name = "mean + std"
    
    # Count samples that would be truncated
    answer_lengths = answer_stats["lengths"]
    truncated_count = int(np.sum(answer_lengths > max_ans_toks))
    truncated_percent = (truncated_count / len(answer_lengths) * 100) if len(answer_lengths) > 0 else 0
    
    # Generate warnings
    warnings = []
    
    if max_ans_toks < threshold:
        warnings.append(
            f"⚠️  WARNING: max_ans_toks={max_ans_toks} is less than {threshold_name}={threshold:.1f}! "
            f"Consider increasing to at least {int(np.ceil(threshold))}."
        )
    
    if truncated_percent > 1:
        warnings.append(
            f"⚠️  WARNING: {truncated_percent:.1f}% of answers ({truncated_count} samples) will be truncated!"
        )
    
    if truncated_percent > 5:
        warnings.append(
            f"🚨 CRITICAL: More than 5% of answers truncated! This may cause significant information loss, "
            f"especially for grounding tasks with bbox coordinates."
        )
    
    # Recommendation
    recommended_max_ans_toks = int(np.ceil(answer_stats["p99"]))
    
    return {
        "max_ans_toks": max_ans_toks,
        "threshold_type": threshold_name,
        "threshold_value": threshold,
        "is_sufficient": max_ans_toks >= threshold,
        "truncated_count": truncated_count,
        "truncated_percent": truncated_percent,
        "recommended_max_ans_toks": recommended_max_ans_toks,
        "warnings": warnings,
    }


def plot_distributions(stats: Dict, output_path: Optional[str] = None):
    """
    Plot token length distributions for questions, answers, and prompts.
    
    Args:
        stats: Output from analyze_token_lengths
        output_path: Path to save the plot (if None, displays interactively)
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib not installed. Cannot create plots.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Token Length Distribution Analysis", fontsize=14, fontweight='bold')
    
    # Question lengths
    ax1 = axes[0, 0]
    q_lengths = stats["question"]["lengths"]
    ax1.hist(q_lengths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(stats["question"]["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["question"]["mean"]:.1f}')
    ax1.axvline(stats["question"]["mean"] + stats["question"]["std"], color='orange', linestyle='--', linewidth=2, label=f'Mean+Std: {stats["question"]["mean"] + stats["question"]["std"]:.1f}')
    ax1.set_xlabel("Token Count")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Question Token Lengths")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Answer lengths
    ax2 = axes[0, 1]
    a_lengths = stats["answer"]["lengths"]
    ax2.hist(a_lengths, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
    ax2.axvline(stats["answer"]["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["answer"]["mean"]:.1f}')
    ax2.axvline(stats["answer"]["mean"] + stats["answer"]["std"], color='orange', linestyle='--', linewidth=2, label=f'Mean+Std: {stats["answer"]["mean"] + stats["answer"]["std"]:.1f}')
    ax2.axvline(stats["answer"]["p95"], color='purple', linestyle=':', linewidth=2, label=f'P95: {stats["answer"]["p95"]:.1f}')
    ax2.set_xlabel("Token Count")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Answer Token Lengths (Critical for Truncation)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Full prompt lengths
    ax3 = axes[1, 0]
    p_lengths = stats["prompt"]["lengths"]
    ax3.hist(p_lengths, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax3.axvline(stats["prompt"]["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["prompt"]["mean"]:.1f}')
    ax3.axvline(stats["prompt"]["mean"] + stats["prompt"]["std"], color='orange', linestyle='--', linewidth=2, label=f'Mean+Std: {stats["prompt"]["mean"] + stats["prompt"]["std"]:.1f}')
    ax3.set_xlabel("Token Count")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Full Prompt Token Lengths (with Chat Template)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [
        ["Metric", "Questions", "Answers", "Prompts"],
        ["Min", f"{stats['question']['min']}", f"{stats['answer']['min']}", f"{stats['prompt']['min']}"],
        ["Max", f"{stats['question']['max']}", f"{stats['answer']['max']}", f"{stats['prompt']['max']}"],
        ["Mean", f"{stats['question']['mean']:.1f}", f"{stats['answer']['mean']:.1f}", f"{stats['prompt']['mean']:.1f}"],
        ["Std Dev", f"{stats['question']['std']:.1f}", f"{stats['answer']['std']:.1f}", f"{stats['prompt']['std']:.1f}"],
        ["Median", f"{stats['question']['median']:.1f}", f"{stats['answer']['median']:.1f}", f"{stats['prompt']['median']:.1f}"],
        ["P95", f"{stats['question']['p95']:.1f}", f"{stats['answer']['p95']:.1f}", f"{stats['prompt']['p95']:.1f}"],
        ["P99", f"{stats['question']['p99']:.1f}", f"{stats['answer']['p99']:.1f}", f"{stats['prompt']['p99']:.1f}"],
    ]
    
    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colColours=['lightgray'] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Summary Statistics", pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[plot] Saved distribution plot to: {output_path}")
    else:
        plt.show()


def print_report(stats: Dict, truncation_check: Optional[Dict] = None):
    """Print a formatted report of the analysis"""
    
    print("\n" + "=" * 80)
    print("TOKEN LENGTH DISTRIBUTION ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples scanned: {stats['total_samples']:,}")
    print(f"  Samples analyzed: {stats['analyzed_samples']:,}")
    print(f"  Samples skipped (missing Q/A): {stats['skipped_samples']:,}")
    
    print("\n" + "-" * 80)
    print("QUESTION TOKEN LENGTHS (Input)")
    print("-" * 80)
    q = stats["question"]
    print(f"  Min:     {q['min']:>6}")
    print(f"  Max:     {q['max']:>6}")
    print(f"  Mean:    {q['mean']:>6.1f}")
    print(f"  Std Dev: {q['std']:>6.1f}")
    print(f"  Median:  {q['median']:>6.1f}")
    print(f"  P95:     {q['p95']:>6.1f}")
    print(f"  P99:     {q['p99']:>6.1f}")
    
    print("\n" + "-" * 80)
    print("ANSWER TOKEN LENGTHS (Output) - CRITICAL FOR TRUNCATION")
    print("-" * 80)
    a = stats["answer"]
    print(f"  Min:     {a['min']:>6}")
    print(f"  Max:     {a['max']:>6}")
    print(f"  Mean:    {a['mean']:>6.1f}")
    print(f"  Std Dev: {a['std']:>6.1f}")
    print(f"  Median:  {a['median']:>6.1f}")
    print(f"  P95:     {a['p95']:>6.1f}")
    print(f"  P99:     {a['p99']:>6.1f}")
    
    print("\n" + "-" * 80)
    print("FULL PROMPT TOKEN LENGTHS (with Chat Template)")
    print("-" * 80)
    p = stats["prompt"]
    print(f"  Min:     {p['min']:>6}")
    print(f"  Max:     {p['max']:>6}")
    print(f"  Mean:    {p['mean']:>6.1f}")
    print(f"  Std Dev: {p['std']:>6.1f}")
    print(f"  Median:  {p['median']:>6.1f}")
    print(f"  P95:     {p['p95']:>6.1f}")
    print(f"  P99:     {p['p99']:>6.1f}")
    
    # Truncation check
    if truncation_check:
        print("\n" + "-" * 80)
        print("TRUNCATION RISK ANALYSIS")
        print("-" * 80)
        print(f"  Current max_ans_toks: {truncation_check['max_ans_toks']}")
        print(f"  Threshold ({truncation_check['threshold_type']}): {truncation_check['threshold_value']:.1f}")
        print(f"  Is sufficient: {'✅ YES' if truncation_check['is_sufficient'] else '❌ NO'}")
        print(f"  Samples to be truncated: {truncation_check['truncated_count']:,} ({truncation_check['truncated_percent']:.2f}%)")
        print(f"  Recommended max_ans_toks (P99): {truncation_check['recommended_max_ans_toks']}")
        
        if truncation_check['warnings']:
            print("\n  WARNINGS:")
            for warning in truncation_check['warnings']:
                print(f"    {warning}")
    
    # Long answer samples
    if stats['long_answer_samples']:
        print("\n" + "-" * 80)
        print("TOP LONGEST ANSWERS (potential truncation victims)")
        print("-" * 80)
        for i, sample in enumerate(stats['long_answer_samples'][:10], 1):
            print(f"\n  {i}. Token: {sample['sample_token']}")
            print(f"     File: {sample['file']}")
            print(f"     Length: {sample['answer_tokens']} tokens")
            print(f"     Preview: {sample['answer_preview'][:100]}...")
    
    print("\n" + "=" * 80)


# ============================================================================
# PYTEST TESTS
# ============================================================================

def test_analyze_token_lengths_mock():
    """Test token length analysis with mock data"""
    from unittest.mock import Mock
    import tempfile
    import os
    
    # Create mock tokenizer
    tokenizer = Mock()
    tokenizer.side_effect = lambda text, **kwargs: {"input_ids": list(range(len(text.split())))}
    tokenizer.apply_chat_template = Mock(return_value="System prompt\nUser: question")
    
    # Create temporary JSON file
    test_data = [
        {"sample_token": "tok1", "question": "What is ahead", "answer_lidar": "A car is ahead at 10 meters"},
        {"sample_token": "tok2", "question": "How many objects", "answer_lidar": "There are three objects visible"},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name
    
    try:
        stats = analyze_token_lengths([temp_path], tokenizer, target_field="answer_lidar")
        
        assert stats["total_samples"] == 2
        assert stats["analyzed_samples"] == 2
        assert "question" in stats
        assert "answer" in stats
        assert "mean" in stats["question"]
        assert "std" in stats["answer"]
        
        print("✅ test_analyze_token_lengths_mock passed!")
    finally:
        os.unlink(temp_path)


def test_check_truncation_risk():
    """Test truncation risk checking"""
    # Mock stats
    mock_stats = {
        "answer": {
            "mean": 50.0,
            "std": 20.0,
            "p95": 80.0,
            "p99": 100.0,
            "lengths": np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
        }
    }
    
    # Test with sufficient max_ans_toks
    result = check_truncation_risk(mock_stats, max_ans_toks=128)
    assert result["is_sufficient"] == True
    assert result["truncated_count"] == 0
    assert len(result["warnings"]) == 0
    
    # Test with insufficient max_ans_toks
    result = check_truncation_risk(mock_stats, max_ans_toks=50)
    assert result["is_sufficient"] == False
    assert result["truncated_count"] > 0
    assert len(result["warnings"]) > 0
    
    print("✅ test_check_truncation_risk passed!")


def test_truncation_warning_threshold():
    """Test that warning is raised when max_ans_toks < mean + std"""
    mock_stats = {
        "answer": {
            "mean": 100.0,
            "std": 30.0,  # mean + std = 130
            "p95": 150.0,
            "p99": 180.0,
            "lengths": np.array([80, 90, 100, 110, 120, 130, 140, 150, 160, 200]),
        }
    }
    
    # Test with max_ans_toks below mean + std (130)
    result = check_truncation_risk(mock_stats, max_ans_toks=128, warn_threshold="mean+std")
    
    assert result["is_sufficient"] == False
    assert any("WARNING" in w for w in result["warnings"])
    
    # Test with max_ans_toks above mean + std
    result = check_truncation_risk(mock_stats, max_ans_toks=200, warn_threshold="mean+std")
    assert result["is_sufficient"] == True
    
    print("✅ test_truncation_warning_threshold passed!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze token length distribution in QA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--json-paths", "-j",
        nargs="+",
        required=True,
        help="Paths to JSON/JSONL files containing QA pairs"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace tokenizer to use (default: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--target-field", "-f",
        default="answer_lidar",
        help="Field containing answer text (default: answer_lidar)"
    )
    parser.add_argument(
        "--max-ans-toks", "-m",
        type=int,
        default=128,
        help="Current max_ans_toks setting to check against (default: 128)"
    )
    parser.add_argument(
        "--warn-threshold", "-w",
        choices=["mean+std", "p95", "p99"],
        default="mean+std",
        help="Threshold for truncation warning (default: mean+std)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate distribution plots"
    )
    parser.add_argument(
        "--output-plot", "-o",
        default=None,
        help="Path to save plot (if not specified, displays interactively)"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run unit tests instead of analysis"
    )
    
    args = parser.parse_args()
    
    if args.run_tests:
        print("Running unit tests...")
        test_analyze_token_lengths_mock()
        test_check_truncation_risk()
        test_truncation_warning_threshold()
        print("\n✅ All tests passed!")
        return
    
    # Load tokenizer
    print(f"[init] Loading tokenizer: {args.tokenizer}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        print("Using simple whitespace tokenizer as fallback...")
        
        class SimpleTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": text.split()}
            def apply_chat_template(self, msgs, **kwargs):
                return f"System: {msgs[0]['content']}\nUser: {msgs[1]['content']}"
        
        tokenizer = SimpleTokenizer()
    
    # Analyze
    print(f"[analyze] Analyzing {len(args.json_paths)} JSON file(s)...")
    stats = analyze_token_lengths(
        args.json_paths,
        tokenizer,
        target_field=args.target_field,
    )
    
    # Check truncation risk
    truncation_check = check_truncation_risk(
        stats,
        max_ans_toks=args.max_ans_toks,
        warn_threshold=args.warn_threshold,
    )
    
    # Print report
    print_report(stats, truncation_check)
    
    # Plot if requested
    if args.plot:
        plot_distributions(stats, args.output_plot)
    
    # Exit with error code if truncation risk is high
    if truncation_check["truncated_percent"] > 5:
        sys.exit(1)


if __name__ == "__main__":
    main()

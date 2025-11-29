"""Data utility functions"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_json_any(path: str) -> Iterable[Dict]:
    """Load JSON from file or JSONL format."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for r in json.load(f):
                yield r
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def collect_feature_tokens(feature_dirs: List[str]) -> Dict[str, str]:
    """
    Collect mapping of sample tokens to feature file paths.
    
    Args:
        feature_dirs: List of directories containing .npy feature files
        
    Returns:
        Dictionary mapping sample_token to feature file path
    """
    import os
    
    def is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0
    
    token2path = {}
    for root in feature_dirs:
        r = Path(root)
        if not r.is_dir():
            if is_main_process():
                print(f"[warn] feature root missing: {root}")
            continue
        # Use recursive glob to find .npy files in subdirectories (e.g., train/, val/)
        for npy in r.glob("**/*.npy"):
            token2path.setdefault(npy.stem, str(npy))
    return token2path


def collect_feature_tokens_with_validation(
    feature_dirs: List[str],
    validate_all: bool = False,
    sample_fraction: float = 0.1,
    min_samples: int = 10,
    max_samples: int = 500,
    num_workers: int = 16,
) -> tuple[Dict[str, str], tuple[int, int, int]]:
    """
    Collect feature tokens AND validate BEV feature shapes are consistent.
    
    This prevents silent failures when BEV features were generated with different
    PCDet model configurations (different channel dimensions, spatial sizes, etc.).
    
    Validation is parallelized for speed - checking 500 files takes ~1-2 seconds.
    
    Args:
        feature_dirs: List of directories containing .npy feature files
        validate_all: If True, validate ALL files (parallelized, but still slower)
        sample_fraction: Fraction of files to sample for validation (default 10%)
        min_samples: Minimum number of files to validate (default 10)
        max_samples: Maximum number of files to validate (default 500)
        num_workers: Number of parallel workers for validation (default 8)
        
    Returns:
        Tuple of:
        - Dictionary mapping sample_token to feature file path
        - Expected BEV shape as (C, H, W) tuple
        
    Raises:
        ValueError: If BEV features have inconsistent shapes
        RuntimeError: If no valid feature files found
        
    Default behavior:
        - Samples 10% of files, bounded to [10, 500] files
        - For 28k files: checks 500 files in parallel (~1-2s)
        - For 100 files: checks all 100 files
    """
    import os
    import random
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0
    
    # First, collect all paths (same as original function)
    token2path = {}
    for root in feature_dirs:
        r = Path(root)
        if not r.is_dir():
            if is_main_process():
                print(f"[warn] feature root missing: {root}")
            continue
        for npy in r.glob("**/*.npy"):
            token2path.setdefault(npy.stem, str(npy))
    
    if not token2path:
        raise RuntimeError(f"No .npy feature files found in: {feature_dirs}")
    
    # Determine which files to validate
    all_paths = list(token2path.values())
    n_total = len(all_paths)
    
    if validate_all:
        paths_to_check = all_paths
    else:
        # Sample a representative subset
        n_samples = max(min_samples, min(max_samples, int(n_total * sample_fraction)))
        n_samples = min(n_samples, n_total)  # Can't sample more than we have
        paths_to_check = random.sample(all_paths, n_samples)
    
    if is_main_process():
        print(f"[BEV validation] Checking {len(paths_to_check)}/{n_total} files with {num_workers} workers...")
    
    # Helper function for parallel shape checking
    def check_shape(path: str) -> tuple[str, tuple | str]:
        """Return (path, shape) or (path, error_string)."""
        try:
            arr = np.load(path, mmap_mode='r')
            return (path, arr.shape)
        except Exception as e:
            return (path, f"LoadError: {e}")
    
    # Parallel validation
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_shape, p): p for p in paths_to_check}
        for future in as_completed(futures):
            results.append(future.result())
    
    # Analyze results
    expected_shape = None
    shape_mismatches = []
    
    for path, shape in results:
        if isinstance(shape, str):  # Error string
            shape_mismatches.append((path, shape))
        elif expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            shape_mismatches.append((path, shape))
    
    if is_main_process() and expected_shape:
        print(f"[BEV validation] Reference shape (C, H, W): {expected_shape}")
    
    # Report mismatches
    if shape_mismatches:
        error_lines = [
            f"BEV feature shape inconsistency detected!",
            f"Expected shape: {expected_shape}",
            f"Found {len(shape_mismatches)} mismatches:"
        ]
        for path, shape in shape_mismatches[:5]:  # Show first 5
            error_lines.append(f"  - {Path(path).name}: {shape}")
        if len(shape_mismatches) > 5:
            error_lines.append(f"  ... and {len(shape_mismatches) - 5} more")
        error_lines.append(
            "\nThis usually means BEV features were generated with different "
            "PCDet model configurations. Re-generate all features with the same model."
        )
        raise ValueError("\n".join(error_lines))
    
    if is_main_process():
        print(f"[BEV validation] All {len(paths_to_check)} checked files have consistent shape {expected_shape}")
    
    return token2path, expected_shape


# =============================================================================
# Additional Validation Functions
# =============================================================================

def validate_json_schema(
    json_paths: List[str],
    required_fields: List[str] = ("sample_token", "question"),
    answer_fields: List[str] = ("answer", "answer_lidar", "answer_vision"),
) -> Dict[str, List[str]]:
    """
    Validate JSON/JSONL files have required schema.
    
    Args:
        json_paths: List of JSON/JSONL file paths
        required_fields: Fields that must be present in every record
        answer_fields: At least one of these must be present (answer content)
        
    Returns:
        Dict with validation results:
        - 'valid_files': list of valid file paths
        - 'issues': list of issue descriptions
        
    Raises:
        ValueError: If critical schema issues found
    """
    import os
    
    def is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0
    
    issues = []
    valid_files = []
    total_records = 0
    records_missing_fields = 0
    records_missing_answer = 0
    
    for jp in json_paths:
        if not Path(jp).exists():
            issues.append(f"JSON file not found: {jp}")
            continue
            
        try:
            file_records = 0
            for record in load_json_any(jp):
                file_records += 1
                total_records += 1
                
                # Check required fields
                missing = [f for f in required_fields if not record.get(f)]
                if missing:
                    records_missing_fields += 1
                    if records_missing_fields <= 3:  # Only report first few
                        issues.append(f"{Path(jp).name}: record missing {missing}")
                
                # Check at least one answer field exists
                has_answer = any(record.get(f) for f in answer_fields)
                if not has_answer:
                    records_missing_answer += 1
            
            valid_files.append(jp)
            if is_main_process():
                print(f"[JSON validation] {Path(jp).name}: {file_records} records")
                
        except Exception as e:
            issues.append(f"Error parsing {jp}: {e}")
    
    if is_main_process():
        print(f"[JSON validation] Total: {total_records} records across {len(valid_files)} files")
        if records_missing_fields > 0:
            print(f"[JSON validation] WARNING: {records_missing_fields} records missing required fields")
        if records_missing_answer > 0:
            print(f"[JSON validation] WARNING: {records_missing_answer} records missing answer content")
    
    return {
        'valid_files': valid_files,
        'issues': issues,
        'total_records': total_records,
        'records_missing_fields': records_missing_fields,
        'records_missing_answer': records_missing_answer,
    }


def validate_token_coverage(
    token2path: Dict[str, str],
    json_paths: List[str],
    target_field: str = "answer_lidar",
) -> Dict[str, any]:
    """
    Validate that JSON sample_tokens have corresponding BEV feature files.
    
    This catches issues where:
    - BEV features were generated for a different dataset split
    - Some samples failed during BEV extraction
    - JSON and features are mismatched
    
    Args:
        token2path: Mapping from sample_token to BEV file path
        json_paths: List of JSON/JSONL files containing sample records
        target_field: Answer field to check
        
    Returns:
        Dict with coverage statistics
    """
    import os
    
    def is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0
    
    total_tokens = 0
    matched_tokens = 0
    unmatched_tokens = []
    
    for jp in json_paths:
        if not Path(jp).exists():
            continue
        for record in load_json_any(jp):
            tok = record.get("sample_token")
            if not tok:
                continue
            total_tokens += 1
            
            if tok in token2path:
                matched_tokens += 1
            else:
                if len(unmatched_tokens) < 10:  # Keep first 10 for reporting
                    unmatched_tokens.append(tok)
    
    coverage_pct = (matched_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    if is_main_process():
        print(f"[Token coverage] {matched_tokens}/{total_tokens} ({coverage_pct:.1f}%) tokens have BEV features")
        if unmatched_tokens:
            print(f"[Token coverage] Missing BEV for: {unmatched_tokens[:5]}...")
    
    return {
        'total_tokens': total_tokens,
        'matched_tokens': matched_tokens,
        'coverage_percent': coverage_pct,
        'unmatched_sample': unmatched_tokens,
    }


def validate_image_paths(
    nusc,
    sample_tokens: List[str],
    view_order: tuple = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                         "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"),
    num_workers: int = 16,
    max_samples: Optional[int] = None,  # None = check ALL samples (entire dataset)
) -> Dict[str, any]:
    """
    Validate that camera image paths exist for sample tokens.
    
    Args:
        nusc: NuScenes object
        sample_tokens: List of sample tokens to check
        view_order: Camera views to validate
        num_workers: Parallel workers
        max_samples: Max samples to check. None = check ALL (default for full validation)
        
    Returns:
        Dict with validation results
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0
    
    if nusc is None:
        return {'status': 'skipped', 'reason': 'NuScenes object not provided'}
    
    # Sample tokens to check (None = check ALL)
    tokens_to_check = sample_tokens if max_samples is None else sample_tokens[:max_samples]
    
    def check_token_images(token: str) -> tuple[str, List[str]]:
        """Return (token, list_of_missing_views)."""
        missing = []
        try:
            sample = nusc.get('sample', token)
            for view in view_order:
                if view not in sample['data']:
                    missing.append(f"{view}:no_data")
                    continue
                sd = nusc.get('sample_data', sample['data'][view])
                img_path = Path(nusc.dataroot) / sd['filename']
                if not img_path.exists():
                    missing.append(f"{view}:not_found")
        except Exception as e:
            missing.append(f"error:{e}")
        return (token, missing)
    
    # Parallel validation
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_token_images, t): t for t in tokens_to_check}
        for future in as_completed(futures):
            results.append(future.result())
    
    # Analyze
    tokens_with_missing = [(t, m) for t, m in results if m]
    
    if is_main_process():
        print(f"[Image validation] Checked {len(tokens_to_check)} samples, {len(view_order)} views each")
        if tokens_with_missing:
            print(f"[Image validation] WARNING: {len(tokens_with_missing)} samples have missing images")
            for tok, missing in tokens_with_missing[:3]:
                print(f"  - {tok}: {missing}")
        else:
            print(f"[Image validation] All checked samples have valid image paths")
    
    return {
        'checked': len(tokens_to_check),
        'tokens_with_missing': len(tokens_with_missing),
        'sample_issues': tokens_with_missing[:10],
    }


def validate_bev_dtype_and_range(
    token2path: Dict[str, str],
    expected_dtype: str = "float32",
    max_samples: int = 50,
    num_workers: int = 16,
) -> Dict[str, any]:
    """
    Validate BEV feature data types and value ranges.
    
    Catches issues like:
    - Wrong dtype (float64 wastes memory, int types lose precision)
    - NaN/Inf values (corrupted features)
    - Unexpected value ranges (normalization issues)
    
    Args:
        token2path: Mapping from sample_token to BEV file path
        expected_dtype: Expected numpy dtype
        max_samples: Max files to check
        num_workers: Parallel workers
        
    Returns:
        Dict with validation results
    """
    import os
    import random
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0
    
    paths = list(token2path.values())
    paths_to_check = random.sample(paths, min(max_samples, len(paths)))
    
    def check_file(path: str) -> Dict:
        """Check dtype, NaN/Inf, and value range."""
        try:
            arr = np.load(path)
            return {
                'path': path,
                'dtype': str(arr.dtype),
                'has_nan': bool(np.isnan(arr).any()),
                'has_inf': bool(np.isinf(arr).any()),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'mean': float(arr.mean()),
            }
        except Exception as e:
            return {'path': path, 'error': str(e)}
    
    # Parallel check
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_file, p): p for p in paths_to_check}
        for future in as_completed(futures):
            results.append(future.result())
    
    # Analyze
    dtype_issues = [r for r in results if r.get('dtype') and r['dtype'] != expected_dtype]
    nan_issues = [r for r in results if r.get('has_nan')]
    inf_issues = [r for r in results if r.get('has_inf')]
    load_errors = [r for r in results if 'error' in r]
    
    # Value range stats
    valid_results = [r for r in results if 'min' in r]
    if valid_results:
        global_min = min(r['min'] for r in valid_results)
        global_max = max(r['max'] for r in valid_results)
        global_mean = sum(r['mean'] for r in valid_results) / len(valid_results)
    else:
        global_min = global_max = global_mean = None
    
    if is_main_process():
        print(f"[BEV dtype/range] Checked {len(paths_to_check)} files")
        if dtype_issues:
            print(f"[BEV dtype/range] WARNING: {len(dtype_issues)} files have unexpected dtype (expected {expected_dtype})")
        if nan_issues:
            print(f"[BEV dtype/range] ERROR: {len(nan_issues)} files contain NaN values!")
        if inf_issues:
            print(f"[BEV dtype/range] ERROR: {len(inf_issues)} files contain Inf values!")
        if load_errors:
            print(f"[BEV dtype/range] ERROR: {len(load_errors)} files failed to load")
        if global_min is not None:
            print(f"[BEV dtype/range] Value range: [{global_min:.3f}, {global_max:.3f}], mean={global_mean:.3f}")
    
    return {
        'checked': len(paths_to_check),
        'dtype_issues': len(dtype_issues),
        'nan_issues': len(nan_issues),
        'inf_issues': len(inf_issues),
        'load_errors': len(load_errors),
        'value_range': (global_min, global_max),
        'mean': global_mean,
    }

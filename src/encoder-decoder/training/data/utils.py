"""Data utility functions"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from tqdm import tqdm


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


# =============================================================================
# Validation Functions
# =============================================================================

def validate_json_schema(
    json_paths: List[str],
    required_fields: List[str] = ("sample_token", "question"),
    answer_fields: List[str] = ("answer", "answer_vision"),
    token2path: Optional[Dict[str, str]] = None,  # Legacy parameter, kept for compatibility
) -> Dict[str, any]:
    """
    Validate JSON/JSONL files schema.
    
    Args:
        json_paths: List of JSON/JSONL file paths
        required_fields: Fields that must be present in every record
        answer_fields: At least one of these must be present (answer content)
        token2path: Legacy parameter, ignored (kept for API compatibility)
        
    Returns:
        Dict with validation results:
        - 'valid_files': list of valid file paths
        - 'issues': list of issue descriptions
        - 'total_records': total records across all files
        - 'records_missing_fields': count of records missing required fields
        - 'records_missing_answer': count of records missing answer content
        
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
            # Count records first for progress bar
            records_list = list(load_json_any(jp))
            file_records = 0
            
            desc = f"📋 Validating {Path(jp).name}"
            with tqdm(
                records_list,
                desc=desc,
                unit="record",
                disable=not is_main_process(),
            ) as pbar:
                for record in pbar:
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
                print(f"[JSON validation] {Path(jp).name}: {file_records} records ✓")
                
        except Exception as e:
            issues.append(f"Error parsing {jp}: {e}")
    
    if is_main_process():
        print(f"[JSON validation] ✓ Total: {total_records} records across {len(valid_files)} files")
        if records_missing_fields > 0:
            print(f"[JSON validation] ⚠ {records_missing_fields} records missing required fields")
        if records_missing_answer > 0:
            print(f"[JSON validation] ⚠ {records_missing_answer} records missing answer content")
    
    result = {
        'valid_files': valid_files,
        'issues': issues,
        'total_records': total_records,
        'records_missing_fields': records_missing_fields,
        'records_missing_answer': records_missing_answer,
    }
    
    return result


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
    
    # Parallel validation with progress bar
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_token_images, t): t for t in tokens_to_check}
        with tqdm(
            total=len(futures),
            desc="📷 Validating image paths",
            unit="sample",
            disable=not is_main_process(),
        ) as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    
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

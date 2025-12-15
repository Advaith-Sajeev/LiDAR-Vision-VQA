# Batch Runner for Multi-View Processing
# Features: 
#   - Processes all 6 camera views in a single API call per artifact
#   - API key batching (splits keys into configurable batches)
#   - API key pooling with immediate reuse
#   - Continuous processing (no batch boundaries)
#   - Zero fixed delays
#   - Smart per-key rate limit handling
# Usage: python batch_runner_multi_view.py

import os
import re
import subprocess
import sys
import time
import json
import threading
import queue
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============== CONFIGURATION ==============
# Batch configuration - splits all API keys into NUM_BATCHES static batches
NUM_BATCHES = 5
BATCH_TO_USE = 1  # Change this to use a different batch (1-5)

# Input/Output directories - automatically determined based on batch number
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "DATA"
ARTIFACTS_DIR = DATA_DIR / f"batch_{BATCH_TO_USE}" / "artifacts"
OUTPUT_DIR = DATA_DIR / f"batch_{BATCH_TO_USE}" / "annotations_multi_view"
LOG_DIR = DATA_DIR / f"batch_{BATCH_TO_USE}" / "logs_multi_view"

# Number of samples to process (set to None to process all)
NUM_SAMPLES = None

# API key pattern to match in .env
API_KEY_PATTERN = r".+_API_\d+$"

# Timeout for individual API calls (seconds)
WORKER_TIMEOUT = 180  # Higher timeout for 6-image processing

# Rate limit backoff (only applied when 429 error detected)
RATE_LIMIT_BACKOFF_INITIAL = 5  # seconds
RATE_LIMIT_BACKOFF_MAX = 60     # maximum backoff
# ===========================================

# Thread-safe tracking
stats_lock = threading.Lock()
backoff_lock = threading.Lock()  # Lock for key_backoff access
stats = {"successful": 0, "failed": 0, "rate_limited": 0}
key_backoff = {}  # Track backoff times per key


def get_api_keys_from_env(pattern: str) -> list[str]:
    """Find all API keys in environment matching the given regex pattern."""
    regex = re.compile(pattern, re.IGNORECASE)
    matching_keys = [key for key in os.environ.keys() if regex.match(key)]
    
    # Sort keys naturally (e.g., API_1, API_2, ... API_10)
    def natural_sort_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    
    matching_keys.sort(key=natural_sort_key)
    return matching_keys


def split_into_batches(items: list, num_batches: int) -> list[list]:
    """Split a list into num_batches roughly equal batches."""
    batch_size = len(items) // num_batches
    remainder = len(items) % num_batches
    
    batches = []
    start = 0
    for i in range(num_batches):
        end = start + batch_size + (1 if i < remainder else 0)
        batches.append(items[start:end])
        start = end
    
    return batches


def get_all_artifact_folders(directory: Path) -> list[Path]:
    """Get ALL artifact subdirectories."""
    if not directory.exists():
        return []
    return [d for d in sorted(directory.iterdir()) if d.is_dir()]


def get_pending_artifacts(all_artifacts: list[Path], output_dir: Path) -> list[Path]:
    """Get artifacts that haven't been processed yet (no output JSON exists)."""
    pending = []
    for artifact in all_artifacts:
        output_file = output_dir / artifact.name / "response.json"
        if not output_file.exists():
            pending.append(artifact)
    return pending


def save_error_log(artifact_name: str, api_key: str, error: str, stdout: str):
    """Save full error details to a log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"error_{timestamp}_{artifact_name[:30]}.log"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Artifact: {artifact_name}\n")
        f.write(f"API Key: {api_key}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 60 + "\n")
        f.write("STDOUT:\n" + stdout + "\n")
        f.write("=" * 60 + "\n")
        f.write("STDERR:\n" + error + "\n")
    
    return log_file


def is_rate_limit_error(stderr: str, stdout: str) -> bool:
    """Check if error is a rate limit (429) error."""
    combined = (stderr + stdout).lower()
    return "429" in combined or "rate limit" in combined or "resource_exhausted" in combined


def run_worker(artifact_path: Path, api_key_name: str, output_path: Path) -> dict:
    """Run the multi_view_worker subprocess for a single artifact."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    worker_script = script_dir / "multi_view_worker.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(worker_script), api_key_name, str(artifact_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT,
            cwd=str(script_dir)  # Run from script directory
        )
        
        if result.returncode == 0:
            return {"artifact": artifact_path, "status": "success", "api_key": api_key_name}
        else:
            # Check for rate limiting
            if is_rate_limit_error(result.stderr, result.stdout):
                return {
                    "artifact": artifact_path, 
                    "status": "rate_limited", 
                    "api_key": api_key_name,
                    "error": result.stderr[:100]
                }
            save_error_log(artifact_path.name, api_key_name, result.stderr, result.stdout)
            return {"artifact": artifact_path, "status": "error", "error": result.stderr[:100], "api_key": api_key_name}
            
    except subprocess.TimeoutExpired:
        return {"artifact": artifact_path, "status": "timeout", "api_key": api_key_name}
    except Exception as e:
        return {"artifact": artifact_path, "status": "error", "error": str(e), "api_key": api_key_name}


def worker_thread(key_queue: queue.Queue, artifact_queue: queue.Queue):
    """
    Worker thread: continuously takes an API key and an artifact,
    processes it, and returns the key to the pool.
    """
    global stats, key_backoff
    
    while True:
        # Get an API key (blocks until available)
        api_key = key_queue.get()
        if api_key is None:  # Shutdown signal
            key_queue.task_done()
            break
        
        # Check if key has backoff time - sleep for exact remaining time
        with backoff_lock:
            if api_key in key_backoff:
                wait_until = key_backoff[api_key]
                now = time.time()
                if now < wait_until:
                    sleep_time = wait_until - now
                else:
                    sleep_time = 0
                del key_backoff[api_key]
            else:
                sleep_time = 0
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Try to get an artifact
        try:
            artifact_path = artifact_queue.get_nowait()
        except queue.Empty:
            # No more artifacts, return key and exit
            key_queue.put(api_key)
            key_queue.task_done()
            break
        
        # Define output path (subdirectory for each artifact)
        output_dir = OUTPUT_DIR / artifact_path.name
        output_path = output_dir / "response.json"
        
        # Quick filesystem check - if output exists, skip
        if output_path.exists():
            key_queue.put(api_key)
            key_queue.task_done()
            artifact_queue.task_done()
            continue
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log start
        artifact_name = artifact_path.name[:40]
        print(f"  🔄 [{api_key}] Processing: {artifact_name}")
        
        # Process the artifact
        result = run_worker(artifact_path, api_key, output_path)
        
        # Handle result
        if result["status"] == "success":
            with stats_lock:
                stats["successful"] += 1
            print(f"  ✅ [{api_key}] {artifact_name}")
        elif result["status"] == "rate_limited":
            with stats_lock:
                stats["rate_limited"] += 1
            # Apply exponential backoff to this key
            with backoff_lock:
                current_backoff = key_backoff.get(f"{api_key}_backoff", RATE_LIMIT_BACKOFF_INITIAL)
                key_backoff[api_key] = time.time() + current_backoff
                key_backoff[f"{api_key}_backoff"] = min(current_backoff * 2, RATE_LIMIT_BACKOFF_MAX)
            print(f"  ⚠️  [{api_key}] Rate limited, backoff {current_backoff}s")
            # Re-queue the artifact for retry
            artifact_queue.put(artifact_path)
        else:
            with stats_lock:
                stats["failed"] += 1
            print(f"  ❌ [{api_key}] {artifact_name} ({result['status']})")
        
        # Return key to pool immediately
        key_queue.put(api_key)
        key_queue.task_done()
        artifact_queue.task_done()


def format_duration(seconds: float) -> str:
    """Format seconds into human readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


def main():
    global stats
    start_time = time.time()
    
    print("=" * 70)
    print("🚀 Multi-View Batch Annotation - Maximum Throughput Mode")
    print(f"⏱️  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Get all API keys and split into batches
    all_keys = get_api_keys_from_env(API_KEY_PATTERN)
    
    if not all_keys:
        print("❌ No API keys found in .env!")
        return
    
    print(f"📊 Total API keys found: {len(all_keys)}")
    
    # Split into batches
    batches = split_into_batches(all_keys, NUM_BATCHES)
    
    # Validate batch selection
    if BATCH_TO_USE < 1 or BATCH_TO_USE > NUM_BATCHES:
        print(f"❌ BATCH_TO_USE must be between 1 and {NUM_BATCHES}")
        return
    
    # Get the selected batch
    api_keys = batches[BATCH_TO_USE - 1]
    
    print(f"📦 Split into {NUM_BATCHES} batches:")
    for i, batch in enumerate(batches, 1):
        marker = " <-- SELECTED" if i == BATCH_TO_USE else ""
        print(f"    Batch {i}: {len(batch)} keys{marker}")
    
    print(f"\n🔑 Using Batch {BATCH_TO_USE}: {len(api_keys)} API keys")
    
    if len(api_keys) == 0:
        print("❌ Selected batch has no API keys!")
        return
    
    # Get all artifacts
    all_artifacts = get_all_artifact_folders(ARTIFACTS_DIR)
    print(f"📁 Total artifacts in folder: {len(all_artifacts)}")
    
    if not all_artifacts:
        print("❌ No artifact folders found!")
        return
    
    # Apply sample limit if configured
    if NUM_SAMPLES is not None:
        all_artifacts = all_artifacts[:NUM_SAMPLES]
        print(f"📌 Limited to first {NUM_SAMPLES} samples")
    
    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get pending artifacts
    pending = get_pending_artifacts(all_artifacts, OUTPUT_DIR)
    
    if not pending:
        print("✅ All artifacts already processed!")
        return
    
    print(f"⏳ Pending: {len(pending)} artifacts")
    print(f"✅ Already done: {len(all_artifacts) - len(pending)} artifacts")
    print("=" * 70)
    print(f"🚀 Starting {len(api_keys)} parallel workers")
    print("=" * 70)
    
    # Create API key pool (queue)
    key_queue = queue.Queue()
    for key in api_keys:
        key_queue.put(key)
    
    # Create artifact queue
    artifact_queue = queue.Queue()
    for artifact in pending:
        artifact_queue.put(artifact)
    
    # Start worker threads (one per API key for maximum parallelism)
    num_workers = len(api_keys)
    threads = []
    
    for _ in range(num_workers):
        t = threading.Thread(
            target=worker_thread,
            args=(key_queue, artifact_queue),
            daemon=True
        )
        t.start()
        threads.append(t)
    
    # Progress monitoring
    last_update = time.time()
    initial_pending = len(pending)
    
    while not artifact_queue.empty() or any(t.is_alive() for t in threads):
        time.sleep(2)
        
        # Progress update every 10 seconds
        if time.time() - last_update >= 10:
            elapsed = time.time() - start_time
            with stats_lock:
                done = stats["successful"]
                failed = stats["failed"]
                rate_limited = stats["rate_limited"]
            
            remaining = initial_pending - done - failed
            rate = done / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0
            
            print(f"\n📊 Progress: {done}/{initial_pending} done | "
                  f"❌ {failed} failed | ⚠️ {rate_limited} rate-limited | "
                  f"⏱️ {format_duration(elapsed)} | "
                  f"ETA: {format_duration(eta)}")
            
            last_update = time.time()
    
    # Wait for all threads to complete
    for t in threads:
        t.join(timeout=5)
    
    # Final statistics
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"✅ Successful: {stats['successful']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⚠️  Rate limited retries: {stats['rate_limited']}")
    print(f"⏱️  Total time: {format_duration(total_duration)}")
    
    if stats['successful'] > 0 and total_duration > 0:
        print(f"📈 Avg per artifact: {format_duration(total_duration / stats['successful'])}")
        print(f"🚀 Throughput: {stats['successful'] / total_duration * 60:.1f} artifacts/min")
    
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    
    # Save timing report
    report = {
        "completed_at": datetime.now().isoformat(),
        "batch_used": BATCH_TO_USE,
        "total_batches": NUM_BATCHES,
        "total_artifacts": len(all_artifacts),
        "total_successful": stats["successful"],
        "total_failed": stats["failed"],
        "rate_limited_retries": stats["rate_limited"],
        "total_duration_seconds": total_duration,
        "throughput_per_minute": stats['successful'] / total_duration * 60 if total_duration > 0 else 0,
        "api_keys_used": len(api_keys)
    }
    
    if stats['successful'] > 0:
        report["avg_per_artifact_seconds"] = total_duration / stats['successful']
    
    report_file = LOG_DIR / f"timing_report_batch{BATCH_TO_USE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Timing report: {report_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user. Progress saved.")
        print(f"📁 Check {OUTPUT_DIR.absolute()} for completed annotations.")

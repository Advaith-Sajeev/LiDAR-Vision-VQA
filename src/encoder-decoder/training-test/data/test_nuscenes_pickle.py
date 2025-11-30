"""
Test NuScenes pickling compatibility for multi-worker DataLoader.

Issue 5.2: NuScenes Object Not Picklable for Multi-Worker DataLoader
-------------------------------------------------------------------
With num_workers > 0, PyTorch pickles the dataset to send to workers.
NuScenes objects may contain DB connections, file handles, or other
non-picklable objects, causing "cannot pickle" errors or worker crashes.

This test verifies:
1. Whether NuScenes can be pickled at all
2. Whether a dataset containing NuScenes can be pickled
3. Whether multi-worker DataLoader works with NuScenes in the dataset

Run this test on a machine with NuScenes installed and data available:
    python -m pytest src/encoder-decoder/training-test/data/test_nuscenes_pickle.py -v

Or run directly:
    python src/encoder-decoder/training-test/data/test_nuscenes_pickle.py
"""

import pickle
import sys
import tempfile
from pathlib import Path

import pytest
from torch.utils.data import Dataset


# Module-level class so it can be pickled (local classes can't be pickled)
class MockDatasetWithoutNusc(Dataset):
    """Mimics MixedNuDataset with nusc=None."""
    def __init__(self):
        self.nusc = None  # Safe: no NuScenes object
        self.samples = [{"token": f"tok_{i}"} for i in range(10)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.nusc is None:
            # This is what _load_camera_images does when nusc is None
            images = [None] * 6
        return {"idx": idx, "token": self.samples[idx]["token"]}


# Module-level class so it can be pickled (local classes can't be pickled)
class MockDatasetWithNusc(Dataset):
    """Mimics MixedNuDataset structure with NuScenes object."""
    def __init__(self, nusc):
        self.nusc = nusc
        self.samples = list(nusc.sample)[:10]  # Just use 10 samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Simulate what _load_camera_images does
        if self.nusc is not None:
            # Access nusc.get() like the real code does
            _ = self.nusc.get("sample", sample["token"])
        return {"idx": idx, "token": sample["token"]}


def test_nuscenes_pickle_basic():
    """Test if NuScenes object can be pickled."""
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        pytest.skip("nuscenes not installed")
    
    # Try to find NuScenes data
    possible_paths = [
        "/data/Datasets/nuScenes",     # Modal volume mount (from modal_config.py)
        "/data/nuscenes",              # Alternative lowercase
        "/data/Datasets/nuscenes",     # Alternative lowercase
        "/home/j_bindu/fyp-26-grp-38/Datasets/nuscenes",
        Path.home() / "data" / "nuscenes",
    ]
    
    dataroot = None
    for p in possible_paths:
        p = Path(p)
        if p.exists() and (p / "v1.0-mini").exists():
            dataroot = str(p)
            break
        if p.exists() and (p / "v1.0-trainval").exists():
            dataroot = str(p)
            break
    
    if dataroot is None:
        pytest.skip("No NuScenes data found at common paths")
    
    # Try mini first, then trainval
    version = "v1.0-mini" if (Path(dataroot) / "v1.0-mini").exists() else "v1.0-trainval"
    
    print(f"\n[test] Loading NuScenes from {dataroot} (version={version})...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    print(f"[test] Loaded {len(nusc.sample)} samples")
    
    # Test 1: Basic pickle
    print("\n[test] Test 1: Basic pickle of NuScenes object...")
    try:
        pickled = pickle.dumps(nusc)
        unpickled = pickle.loads(pickled)
        print(f"[test] ✅ NuScenes CAN be pickled! Size: {len(pickled):,} bytes")
        print(f"[test] Unpickled has {len(unpickled.sample)} samples")
        pickle_works = True
    except Exception as e:
        print(f"[test] ❌ NuScenes CANNOT be pickled: {type(e).__name__}: {e}")
        pickle_works = False
    
    # Test 2: Pickle to file (simulates what DataLoader does)
    print("\n[test] Test 2: Pickle to tempfile...")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(nusc, f)
            temp_path = f.name
        
        with open(temp_path, "rb") as f:
            unpickled = pickle.load(f)
        
        Path(temp_path).unlink()
        print(f"[test] ✅ File pickle works! Unpickled has {len(unpickled.sample)} samples")
    except Exception as e:
        print(f"[test] ❌ File pickle failed: {type(e).__name__}: {e}")
    
    # Test 3: Check internal state that might cause issues
    print("\n[test] Test 3: Checking NuScenes internal state...")
    attrs_to_check = ['_token2ind', 'table_names', 'dataroot', 'version']
    for attr in attrs_to_check:
        if hasattr(nusc, attr):
            val = getattr(nusc, attr)
            val_type = type(val).__name__
            if isinstance(val, dict):
                print(f"[test]   {attr}: dict with {len(val)} entries")
            elif isinstance(val, list):
                print(f"[test]   {attr}: list with {len(val)} entries")
            else:
                print(f"[test]   {attr}: {val_type}")
    
    assert pickle_works, "NuScenes should be picklable for multi-worker DataLoader"


def test_dataset_with_nuscenes_pickle():
    """Test if a dataset containing NuScenes can be pickled."""
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        pytest.skip("nuscenes not installed")
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    # Find NuScenes data
    possible_paths = [
        "/data/Datasets/nuScenes",     # Modal volume mount (from modal_config.py)
        "/data/nuscenes",
        "/data/Datasets/nuscenes",
        "/home/j_bindu/fyp-26-grp-38/Datasets/nuscenes",
    ]
    
    dataroot = None
    for p in possible_paths:
        p = Path(p)
        if p.exists():
            dataroot = str(p)
            break
    
    if dataroot is None:
        pytest.skip("No NuScenes data found")
    
    version = "v1.0-mini" if (Path(dataroot) / "v1.0-mini").exists() else "v1.0-trainval"
    
    print(f"\n[test] Creating mock dataset with NuScenes...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    # Use module-level MockDatasetWithNusc class (local classes can't be pickled)
    dataset = MockDatasetWithNusc(nusc)
    
    # Test 1: Pickle the dataset
    print("\n[test] Test 1: Pickle dataset with NuScenes...")
    try:
        pickled = pickle.dumps(dataset)
        unpickled = pickle.loads(pickled)
        print(f"[test] ✅ Dataset with NuScenes CAN be pickled! Size: {len(pickled):,} bytes")
        dataset_pickle_works = True
    except Exception as e:
        print(f"[test] ❌ Dataset with NuScenes CANNOT be pickled: {type(e).__name__}: {e}")
        pytest.fail(f"Dataset with NuScenes should be picklable: {e}")
    
    # Test 2: DataLoader with num_workers=0 (no pickling)
    print("\n[test] Test 2: DataLoader with num_workers=0...")
    try:
        loader = DataLoader(dataset, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        print(f"[test] ✅ num_workers=0 works: got batch with {len(batch['idx'])} items")
    except Exception as e:
        print(f"[test] ❌ num_workers=0 failed: {type(e).__name__}: {e}")
    
    # Test 3: DataLoader with num_workers=2 (requires pickling)
    print("\n[test] Test 3: DataLoader with num_workers=2 (requires pickle)...")
    try:
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        batch = next(iter(loader))
        print(f"[test] ✅ num_workers=2 works: got batch with {len(batch['idx'])} items")
        multiworker_works = True
    except Exception as e:
        print(f"[test] ❌ num_workers=2 failed: {type(e).__name__}: {e}")
        multiworker_works = False
    
    assert multiworker_works, "Multi-worker DataLoader should work with NuScenes dataset"


def test_dataset_without_nuscenes_pickle():
    """Test that dataset works when nusc=None (the safe path)."""
    from torch.utils.data import DataLoader
    
    print("\n[test] Testing dataset with nusc=None (safe path)...")
    
    # Use module-level MockDatasetWithoutNusc class (local classes can't be pickled)
    dataset = MockDatasetWithoutNusc()
    
    # Test pickle
    print("[test] Test 1: Pickle dataset with nusc=None...")
    try:
        pickled = pickle.dumps(dataset)
        unpickled = pickle.loads(pickled)
        print(f"[test] ✅ Dataset with nusc=None CAN be pickled! Size: {len(pickled):,} bytes")
        pickle_works = True
    except Exception as e:
        print(f"[test] ❌ Unexpected: nusc=None dataset failed pickle: {e}")
        pickle_works = False
    
    assert pickle_works, "Dataset with nusc=None should be picklable"
    
    # Test multi-worker
    print("[test] Test 2: DataLoader with num_workers=2...")
    try:
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        batch = next(iter(loader))
        print(f"[test] ✅ num_workers=2 works with nusc=None")
        multiworker_works = True
    except Exception as e:
        print(f"[test] ❌ num_workers=2 failed: {e}")
        multiworker_works = False
    
    assert multiworker_works, "Multi-worker DataLoader should work with nusc=None"


def main():
    """Run all tests and summarize when running as a script."""
    print("=" * 70)
    print("NuScenes Pickle Compatibility Tests")
    print("=" * 70)
    print("\nThis tests whether NuScenes can be used with multi-worker DataLoader.")
    print("Issue: With num_workers > 0, PyTorch pickles the dataset to workers.")
    print("=" * 70)
    
    results = {"basic_pickle": "unknown", "dataset_pickle": "unknown", "without_nusc": "unknown"}
    
    # Test 1: Basic NuScenes pickle
    print("\n" + "=" * 70)
    print("TEST 1: Basic NuScenes Pickle")
    print("=" * 70)
    try:
        test_nuscenes_pickle_basic()
        results["basic_pickle"] = "pass"
    except pytest.skip.Exception as e:
        results["basic_pickle"] = f"skip: {e}"
    except AssertionError as e:
        results["basic_pickle"] = f"fail: {e}"
    except Exception as e:
        results["basic_pickle"] = f"error: {e}"
    
    # Test 2: Dataset with NuScenes
    print("\n" + "=" * 70)
    print("TEST 2: Dataset with NuScenes Pickle")
    print("=" * 70)
    try:
        test_dataset_with_nuscenes_pickle()
        results["dataset_pickle"] = "pass"
    except pytest.skip.Exception as e:
        results["dataset_pickle"] = f"skip: {e}"
    except AssertionError as e:
        results["dataset_pickle"] = f"fail: {e}"
    except Exception as e:
        results["dataset_pickle"] = f"error: {e}"
    
    # Test 3: Dataset without NuScenes (safe path)
    print("\n" + "=" * 70)
    print("TEST 3: Dataset without NuScenes (nusc=None)")
    print("=" * 70)
    try:
        test_dataset_without_nuscenes_pickle()
        results["without_nusc"] = "pass"
    except AssertionError as e:
        results["without_nusc"] = f"fail: {e}"
    except Exception as e:
        results["without_nusc"] = f"error: {e}"
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, status in results.items():
        if status == "pass":
            emoji = "✅"
        elif status.startswith("skip"):
            emoji = "⏭️"
        else:
            emoji = "❌"
        print(f"{emoji} {name}: {status}")
    
    # Recommendation
    print("\n" + "-" * 70)
    print("RECOMMENDATION:")
    print("-" * 70)
    
    all_pass = all(s == "pass" for s in results.values())
    any_skip = any(s.startswith("skip") if isinstance(s, str) else False for s in results.values())
    
    if all_pass:
        print("✅ NuScenes IS picklable and works with multi-worker DataLoader!")
        print("   Issue 5.2 is NOT a problem for this NuScenes version.")
        print("   You can safely use num_workers > 0 with load_images=True.")
    elif any_skip:
        print("⚠️  Could not fully test - NuScenes not available.")
        print("   Run this test on your training server.")
    else:
        print("❌ NuScenes has pickling issues!")
        print("   WORKAROUNDS:")
        print("   1. Use num_workers=0 when load_images=True")
        print("   2. Pre-resolve image paths during __init__ instead of in __getitem__")
        print("   3. Store only dataroot/version and recreate NuScenes in workers")
    
    return results


if __name__ == "__main__":
    main()

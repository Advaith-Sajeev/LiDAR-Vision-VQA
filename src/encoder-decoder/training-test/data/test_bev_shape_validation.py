"""
Tests for BEV feature shape validation.

Tests Issue 3.1: Validates that inconsistent BEV shapes are detected during
dataset initialization, preventing silent failures from mixed PCDet model outputs.
"""

import os
import sys
import tempfile
import numpy as np
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so imports work from any location
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent                          # .../training-test/data
PROJECT_ROOT = THIS_DIR.parent.parent                               # .../encoder-decoder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.utils import (
    collect_feature_tokens,
    collect_feature_tokens_with_validation,
)


class TestCollectFeatureTokens:
    """Tests for the original collect_feature_tokens function."""
    
    def test_empty_dirs(self, tmp_path):
        """Should return empty dict for empty directories."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = collect_feature_tokens([str(empty_dir)])
        assert result == {}
    
    def test_missing_dir(self, tmp_path):
        """Should handle missing directories gracefully."""
        missing = str(tmp_path / "nonexistent")
        result = collect_feature_tokens([missing])
        assert result == {}
    
    def test_finds_npy_files(self, tmp_path):
        """Should find and index .npy files."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create test files
        for i in range(3):
            token = f"sample_token_{i}"
            arr = np.random.randn(256, 128, 128).astype(np.float32)
            np.save(feat_dir / f"{token}.npy", arr)
        
        result = collect_feature_tokens([str(feat_dir)])
        
        assert len(result) == 3
        assert "sample_token_0" in result
        assert "sample_token_1" in result
        assert "sample_token_2" in result
    
    def test_nested_directories(self, tmp_path):
        """Should find .npy files in nested directories."""
        train_dir = tmp_path / "features" / "train"
        val_dir = tmp_path / "features" / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        
        # Create test files
        np.save(train_dir / "train_token.npy", np.random.randn(256, 128, 128))
        np.save(val_dir / "val_token.npy", np.random.randn(256, 128, 128))
        
        result = collect_feature_tokens([str(tmp_path / "features")])
        
        assert len(result) == 2
        assert "train_token" in result
        assert "val_token" in result


class TestCollectFeatureTokensWithValidation:
    """Tests for the new validated collection function."""
    
    def test_empty_dirs_raises(self, tmp_path):
        """Should raise RuntimeError for empty directories."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(RuntimeError, match="No .npy feature files found"):
            collect_feature_tokens_with_validation([str(empty_dir)])
    
    def test_consistent_shapes_pass(self, tmp_path):
        """Should succeed when all BEV features have consistent shapes."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create files with CONSISTENT shapes
        expected_shape = (256, 128, 128)
        for i in range(10):
            token = f"sample_{i}"
            arr = np.random.randn(*expected_shape).astype(np.float32)
            np.save(feat_dir / f"{token}.npy", arr)
        
        token2path, shape = collect_feature_tokens_with_validation(
            [str(feat_dir)], 
            validate_all=True
        )
        
        assert len(token2path) == 10
        assert shape == expected_shape
    
    def test_inconsistent_shapes_raises(self, tmp_path):
        """Should raise ValueError when BEV features have inconsistent shapes."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create files with INCONSISTENT shapes (simulates different PCDet models)
        shapes = [
            (256, 128, 128),  # VoxelNeXt
            (256, 128, 128),
            (512, 64, 64),    # Different model!
            (256, 128, 128),
        ]
        
        for i, shape in enumerate(shapes):
            token = f"sample_{i}"
            arr = np.random.randn(*shape).astype(np.float32)
            np.save(feat_dir / f"{token}.npy", arr)
        
        with pytest.raises(ValueError, match="BEV feature shape inconsistency"):
            collect_feature_tokens_with_validation([str(feat_dir)], validate_all=True)
    
    def test_channel_mismatch_detected(self, tmp_path):
        """Should detect channel dimension mismatches."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Spatial dimensions same, but channel dimension different
        np.save(feat_dir / "token_a.npy", np.random.randn(256, 128, 128))
        np.save(feat_dir / "token_b.npy", np.random.randn(512, 128, 128))  # Different C!
        
        with pytest.raises(ValueError, match="BEV feature shape inconsistency"):
            collect_feature_tokens_with_validation([str(feat_dir)], validate_all=True)
    
    def test_spatial_mismatch_detected(self, tmp_path):
        """Should detect spatial dimension mismatches."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Channel dimension same, but spatial different
        np.save(feat_dir / "token_a.npy", np.random.randn(256, 128, 128))
        np.save(feat_dir / "token_b.npy", np.random.randn(256, 64, 64))  # Different H,W!
        
        with pytest.raises(ValueError, match="BEV feature shape inconsistency"):
            collect_feature_tokens_with_validation([str(feat_dir)], validate_all=True)
    
    def test_returns_expected_shape(self, tmp_path):
        """Should return the expected shape tuple."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        expected = (384, 96, 96)  # Custom shape
        for i in range(5):
            np.save(feat_dir / f"token_{i}.npy", np.random.randn(*expected))
        
        _, shape = collect_feature_tokens_with_validation([str(feat_dir)], validate_all=True)
        
        assert shape == expected
        assert isinstance(shape, tuple)
        assert len(shape) == 3
    
    def test_sampling_mode(self, tmp_path):
        """Should sample subset when validate_all=False."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create many files
        for i in range(100):
            np.save(feat_dir / f"token_{i}.npy", np.random.randn(256, 128, 128))
        
        # With sampling, should still validate (and pass)
        token2path, shape = collect_feature_tokens_with_validation(
            [str(feat_dir)],
            validate_all=False,
            sample_fraction=0.1,
            min_samples=5,
            max_samples=20
        )
        
        assert len(token2path) == 100  # All tokens indexed
        assert shape == (256, 128, 128)  # Shape still validated
    
    def test_sampling_catches_issues(self, tmp_path):
        """Sampling should catch shape issues if they're in the sample."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create mostly consistent files
        for i in range(10):
            np.save(feat_dir / f"token_{i}.npy", np.random.randn(256, 128, 128))
        
        # Add one bad file
        np.save(feat_dir / "bad_token.npy", np.random.randn(512, 64, 64))
        
        # With validate_all=True, should definitely catch it
        with pytest.raises(ValueError, match="BEV feature shape inconsistency"):
            collect_feature_tokens_with_validation([str(feat_dir)], validate_all=True)
    
    def test_corrupted_file_reported(self, tmp_path):
        """Should report corrupted/invalid files in error."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create one valid file
        np.save(feat_dir / "valid_token.npy", np.random.randn(256, 128, 128))
        
        # Create one corrupted file
        with open(feat_dir / "corrupted.npy", "w") as f:
            f.write("not a numpy file")
        
        # Should report the load error
        with pytest.raises(ValueError):
            collect_feature_tokens_with_validation([str(feat_dir)], validate_all=True)
    
    def test_multiple_directories(self, tmp_path):
        """Should validate across multiple directories."""
        dir1 = tmp_path / "features1"
        dir2 = tmp_path / "features2"
        dir1.mkdir()
        dir2.mkdir()
        
        # Same shape in both dirs
        np.save(dir1 / "token_a.npy", np.random.randn(256, 128, 128))
        np.save(dir2 / "token_b.npy", np.random.randn(256, 128, 128))
        
        token2path, shape = collect_feature_tokens_with_validation(
            [str(dir1), str(dir2)],
            validate_all=True
        )
        
        assert len(token2path) == 2
        assert shape == (256, 128, 128)
    
    def test_multiple_directories_mismatch(self, tmp_path):
        """Should catch mismatches across directories."""
        dir1 = tmp_path / "features1"
        dir2 = tmp_path / "features2"
        dir1.mkdir()
        dir2.mkdir()
        
        # Different shapes in different dirs (common mistake)
        np.save(dir1 / "token_a.npy", np.random.randn(256, 128, 128))
        np.save(dir2 / "token_b.npy", np.random.randn(512, 64, 64))
        
        with pytest.raises(ValueError, match="BEV feature shape inconsistency"):
            collect_feature_tokens_with_validation([str(dir1), str(dir2)], validate_all=True)


class TestDatasetBEVValidation:
    """Integration tests for BEV validation in MixedNuDataset."""
    
    @pytest.fixture
    def minimal_json(self, tmp_path):
        """Create minimal JSON with sample tokens."""
        import json
        json_path = tmp_path / "samples.json"
        samples = [
            {"sample_token": f"token_{i}", "question": "test?", "answer_lidar": "yes"}
            for i in range(3)
        ]
        with open(json_path, "w") as f:
            json.dump(samples, f)
        return str(json_path)
    
    def test_dataset_validates_shapes(self, tmp_path, minimal_json):
        """MixedNuDataset should validate BEV shapes by default."""
        from training.data.dataset import MixedNuDataset
        
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create consistent features
        for i in range(3):
            np.save(feat_dir / f"token_{i}.npy", np.random.randn(256, 128, 128))
        
        ds = MixedNuDataset(
            json_paths=[minimal_json],
            feature_dirs=[str(feat_dir)],
            validate_bev_shapes=True
        )
        
        assert ds.bev_shape == (256, 128, 128)
        assert ds.bev_channels == 256
        assert ds.bev_spatial_shape == (128, 128)
    
    def test_dataset_catches_mismatch(self, tmp_path, minimal_json):
        """MixedNuDataset should raise on shape mismatch."""
        from training.data.dataset import MixedNuDataset
        
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create inconsistent features
        np.save(feat_dir / "token_0.npy", np.random.randn(256, 128, 128))
        np.save(feat_dir / "token_1.npy", np.random.randn(512, 64, 64))  # Different!
        np.save(feat_dir / "token_2.npy", np.random.randn(256, 128, 128))
        
        with pytest.raises(ValueError, match="BEV feature shape inconsistency"):
            MixedNuDataset(
                json_paths=[minimal_json],
                feature_dirs=[str(feat_dir)],
                validate_bev_shapes=True,
                validate_all_bev=True
            )
    
    def test_dataset_validation_disabled(self, tmp_path, minimal_json):
        """Should skip validation when validate_bev_shapes=False."""
        from training.data.dataset import MixedNuDataset
        
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create inconsistent features (would normally fail)
        np.save(feat_dir / "token_0.npy", np.random.randn(256, 128, 128))
        np.save(feat_dir / "token_1.npy", np.random.randn(512, 64, 64))
        np.save(feat_dir / "token_2.npy", np.random.randn(256, 128, 128))
        
        # Should NOT raise because validation disabled
        ds = MixedNuDataset(
            json_paths=[minimal_json],
            feature_dirs=[str(feat_dir)],
            validate_bev_shapes=False
        )
        
        assert ds.bev_shape is None
        assert ds.bev_channels is None
    
    def test_runtime_shape_check(self, tmp_path, minimal_json):
        """Runtime should catch shapes not found during sampling."""
        from training.data.dataset import MixedNuDataset
        
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        
        # Create files - sampling might miss the bad one
        for i in range(3):
            np.save(feat_dir / f"token_{i}.npy", np.random.randn(256, 128, 128))
        
        ds = MixedNuDataset(
            json_paths=[minimal_json],
            feature_dirs=[str(feat_dir)],
            validate_bev_shapes=True
        )
        
        # Now corrupt one file after init
        np.save(feat_dir / "token_1.npy", np.random.randn(512, 64, 64))
        
        # First item should work
        item0 = ds[0]
        assert item0["bev"].shape == (256, 128, 128)
        
        # Second item (corrupted) should raise at runtime
        with pytest.raises(ValueError, match="BEV shape mismatch at runtime"):
            _ = ds[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

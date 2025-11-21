"""Tests for data utilities"""

import pytest
import json
import tempfile
from pathlib import Path
from training.data.utils import load_json_any, collect_feature_tokens


class TestLoadJsonAny:
    """Tests for load_json_any function"""
    
    def test_load_json_array(self):
        """Test loading JSON array format"""
        data = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = list(load_json_any(temp_path))
            assert len(result) == 2
            assert result[0]["id"] == 1
            assert result[1]["name"] == "test2"
        finally:
            Path(temp_path).unlink()
    
    def test_load_jsonl(self):
        """Test loading JSONL format"""
        lines = [
            '{"id": 1, "name": "test1"}\n',
            '{"id": 2, "name": "test2"}\n',
            '{"id": 3, "name": "test3"}\n'
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.writelines(lines)
            temp_path = f.name
        
        try:
            result = list(load_json_any(temp_path))
            assert len(result) == 3
            assert result[0]["id"] == 1
            assert result[2]["name"] == "test3"
        finally:
            Path(temp_path).unlink()
    
    def test_load_jsonl_with_empty_lines(self):
        """Test loading JSONL with empty lines"""
        lines = [
            '{"id": 1}\n',
            '\n',
            '{"id": 2}\n',
            '  \n',
            '{"id": 3}\n'
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.writelines(lines)
            temp_path = f.name
        
        try:
            result = list(load_json_any(temp_path))
            assert len(result) == 3
            assert [r["id"] for r in result] == [1, 2, 3]
        finally:
            Path(temp_path).unlink()
    
    def test_load_empty_json_array(self):
        """Test loading empty JSON array"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            temp_path = f.name
        
        try:
            result = list(load_json_any(temp_path))
            assert len(result) == 0
        finally:
            Path(temp_path).unlink()


class TestCollectFeatureTokens:
    """Tests for collect_feature_tokens function"""
    
    def test_collect_from_single_directory(self):
        """Test collecting features from a single directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some .npy files
            (Path(tmpdir) / "token1.npy").touch()
            (Path(tmpdir) / "token2.npy").touch()
            (Path(tmpdir) / "token3.npy").touch()
            
            result = collect_feature_tokens([tmpdir])
            
            assert len(result) == 3
            assert "token1" in result
            assert "token2" in result
            assert "token3" in result
            assert result["token1"].endswith("token1.npy")
    
    def test_collect_from_multiple_directories(self):
        """Test collecting from multiple directories"""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                (Path(tmpdir1) / "token1.npy").touch()
                (Path(tmpdir1) / "token2.npy").touch()
                (Path(tmpdir2) / "token3.npy").touch()
                (Path(tmpdir2) / "token4.npy").touch()
                
                result = collect_feature_tokens([tmpdir1, tmpdir2])
                
                assert len(result) == 4
                assert all(k in result for k in ["token1", "token2", "token3", "token4"])
    
    def test_collect_with_subdirectories(self):
        """Test collecting from nested subdirectories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories
            train_dir = Path(tmpdir) / "train"
            val_dir = Path(tmpdir) / "val"
            train_dir.mkdir()
            val_dir.mkdir()
            
            (train_dir / "token1.npy").touch()
            (train_dir / "token2.npy").touch()
            (val_dir / "token3.npy").touch()
            
            result = collect_feature_tokens([tmpdir])
            
            assert len(result) == 3
            assert "token1" in result
            assert "token3" in result
    
    def test_collect_handles_missing_directory(self):
        """Test that missing directories are handled gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "token1.npy").touch()
            
            # Include a non-existent directory
            result = collect_feature_tokens([tmpdir, "/nonexistent/path"])
            
            # Should still get the valid token
            assert len(result) == 1
            assert "token1" in result
    
    def test_collect_ignores_non_npy_files(self):
        """Test that non-.npy files are ignored"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "token1.npy").touch()
            (Path(tmpdir) / "token2.txt").touch()
            (Path(tmpdir) / "token3.json").touch()
            (Path(tmpdir) / "token4.npy").touch()
            
            result = collect_feature_tokens([tmpdir])
            
            assert len(result) == 2
            assert "token1" in result
            assert "token4" in result
            assert "token2" not in result
            assert "token3" not in result
    
    def test_collect_no_duplicates(self):
        """Test that duplicate tokens use first occurrence"""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                path1 = Path(tmpdir1) / "token1.npy"
                path2 = Path(tmpdir2) / "token1.npy"
                path1.touch()
                path2.touch()
                
                result = collect_feature_tokens([tmpdir1, tmpdir2])
                
                # Should have only one entry for token1
                assert len(result) == 1
                assert "token1" in result
                # Should use the first occurrence
                assert result["token1"] == str(path1)
    
    def test_collect_empty_directory(self):
        """Test collecting from empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_feature_tokens([tmpdir])
            assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

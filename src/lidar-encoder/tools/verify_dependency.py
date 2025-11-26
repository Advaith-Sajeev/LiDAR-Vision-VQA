#!/usr/bin/env python3
"""
Dependency Verification Script for PCDet

This script verifies that all required dependencies (except visualization tools)
are properly installed and working. It tests:
- Core Python packages
- PyTorch and CUDA availability
- PCDet modules
- Dataset processing capabilities
- Model building capabilities
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    print(f"  {text}")

# Track overall status
all_passed = True

# 1. Test Core Python Packages
print_header("Testing Core Python Packages")

required_packages = {
    'numpy': 'NumPy',
    'torch': 'PyTorch',
    'numba': 'Numba',
    'llvmlite': 'LLVM Lite',
    'easydict': 'EasyDict',
    'yaml': 'PyYAML',
    'skimage': 'scikit-image',
    'tqdm': 'TQDM',
    'cv2': 'OpenCV',
    'pyquaternion': 'PyQuaternion',
    'tensorboardX': 'TensorBoardX',
    'SharedArray': 'SharedArray',
}

for module_name, display_name in required_packages.items():
    try:
        if module_name == 'yaml':
            import yaml
            version = getattr(yaml, '__version__', 'unknown')
        elif module_name == 'skimage':
            import skimage
            version = skimage.__version__
        elif module_name == 'cv2':
            import cv2
            version = cv2.__version__
        else:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
        
        print_success(f"{display_name:20s} version {version}")
    except ImportError as e:
        print_error(f"{display_name:20s} NOT FOUND")
        all_passed = False

# 2. Test PyTorch and CUDA
print_header("Testing PyTorch and CUDA")

try:
    import torch
    print_success(f"PyTorch version: {torch.__version__}")
    print_info(f"Python version: {sys.version.split()[0]}")
    
    if torch.cuda.is_available():
        print_success(f"CUDA available: True")
        print_info(f"CUDA version: {torch.version.cuda}")
        print_info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print_info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test CUDA tensor creation
        try:
            test_tensor = torch.randn(10, 10).cuda()
            print_success("CUDA tensor creation: OK")
        except Exception as e:
            print_error(f"CUDA tensor creation failed: {e}")
            all_passed = False
    else:
        print_warning("CUDA not available - will run on CPU only")
        
except Exception as e:
    print_error(f"PyTorch test failed: {e}")
    all_passed = False

# 3. Test spconv (important for point cloud processing)
print_header("Testing spconv (Sparse Convolution)")

try:
    import spconv
    version = getattr(spconv, '__version__', 'unknown')
    print_success(f"spconv version: {version}")
    
    # Try to import pytorch module
    try:
        from spconv.pytorch import SparseConvTensor
        print_success("spconv.pytorch module: OK")
    except Exception as e:
        print_error(f"spconv.pytorch import failed: {e}")
        all_passed = False
        
except ImportError:
    print_error("spconv NOT FOUND - this is critical for point cloud processing")
    all_passed = False

# 4. Test PCDet modules
print_header("Testing PCDet Modules")

# Add parent directory to path to import pcdet
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

pcdet_modules = [
    ('pcdet.config', 'Config module'),
    ('pcdet.datasets', 'Datasets module'),
    ('pcdet.models', 'Models module'),
    ('pcdet.utils', 'Utils module'),
    ('pcdet.ops', 'Ops module'),
]

for module_name, display_name in pcdet_modules:
    try:
        mod = __import__(module_name, fromlist=[''])
        print_success(f"{display_name:25s} OK")
    except ImportError as e:
        print_error(f"{display_name:25s} FAILED: {e}")
        all_passed = False

# 5. Test PCDet-specific imports
print_header("Testing PCDet Core Functionality")

try:
    from pcdet.config import cfg, cfg_from_yaml_file
    print_success("Config loading functions: OK")
except ImportError as e:
    print_error(f"Config loading failed: {e}")
    all_passed = False

try:
    from pcdet.datasets import DatasetTemplate
    print_success("DatasetTemplate: OK")
except ImportError as e:
    print_error(f"DatasetTemplate import failed: {e}")
    all_passed = False

try:
    from pcdet.models import build_network, load_data_to_gpu
    print_success("Model building functions: OK")
except ImportError as e:
    print_error(f"Model functions import failed: {e}")
    all_passed = False

try:
    from pcdet.utils import common_utils
    print_success("Common utils: OK")
except ImportError as e:
    print_error(f"Common utils import failed: {e}")
    all_passed = False

# 6. Test CUDA extensions (if built)
print_header("Testing CUDA Extensions")

cuda_extensions = [
    'pcdet.ops.iou3d_nms',
    'pcdet.ops.roiaware_pool3d',
    'pcdet.ops.pointnet2.pointnet2_stack',
    'pcdet.ops.pointnet2.pointnet2_batch',
]

extensions_built = False
for ext_name in cuda_extensions:
    try:
        mod = __import__(ext_name, fromlist=[''])
        print_success(f"{ext_name:40s} OK")
        extensions_built = True
    except ImportError as e:
        print_warning(f"{ext_name:40s} Not built (run 'pip install -e . --no-build-isolation')")

if not extensions_built:
    print_warning("No CUDA extensions built yet. Run 'pip install -e . --no-build-isolation' to build them.")

# 7. Test config file loading
print_header("Testing Config File Loading")

tools_dir = repo_root / 'tools'
config_files = list((tools_dir / 'cfgs').glob('**/*.yaml'))

if config_files:
    test_cfg = config_files[0]
    try:
        # Temporarily change to tools directory for config loading
        original_cwd = os.getcwd()
        os.chdir(tools_dir)
        
        from pcdet.config import cfg, cfg_from_yaml_file
        cfg_from_yaml_file(str(test_cfg.relative_to(tools_dir)), cfg)
        print_success(f"Config loading test: OK (tested with {test_cfg.name})")
        print_info(f"Found {len(config_files)} config files in cfgs/")
        
        os.chdir(original_cwd)
    except Exception as e:
        os.chdir(original_cwd)
        print_error(f"Config loading test failed: {e}")
        all_passed = False
else:
    print_warning("No config files found in tools/cfgs/")

# 8. Summary
print_header("Verification Summary")

if all_passed:
    print_success("All core dependencies are properly installed and working!")
    print_info("\nNext steps:")
    print_info("1. If CUDA extensions are not built, run:")
    print_info("   pip install -e . --no-build-isolation")
    print_info("2. Run demo_new.py to test inference:")
    print_info("   python demo_new.py")
    sys.exit(0)
else:
    print_error("Some dependencies are missing or not working correctly.")
    print_info("\nPlease install missing dependencies before proceeding.")
    sys.exit(1)

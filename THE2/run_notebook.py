#!/usr/bin/env python3
"""
Script to execute the notebook cells programmatically
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
print("Testing imports...")
import numpy as np
print(f"✓ NumPy {np.__version__}")

import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

import torchvision
print(f"✓ Torchvision {torchvision.__version__}")

import matplotlib
print(f"✓ Matplotlib {matplotlib.__version__}")

print("\nAll imports successful! Environment is ready.")

# Test Task 1 - Deformable Convolution
print("\n" + "="*50)
print("Testing Task 1: Deformable Convolution")
print("="*50)

# Import the implementation from cell 4
exec(open('test_deformable_conv.py').read())
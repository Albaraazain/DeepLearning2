#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1 support..."
echo "This will download approximately 800MB, please be patient..."

# Option 1: For CUDA 12.1 (recommended for your setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If the above fails, try Option 2: For CUDA 11.8 (smaller download)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
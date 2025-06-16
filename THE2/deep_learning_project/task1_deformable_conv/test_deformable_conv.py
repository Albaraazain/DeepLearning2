"""
Test script for deformable convolution implementation
"""

import numpy as np
import torch
from torchvision.ops import deform_conv2d
from .deformable_conv import deform_conv2d_np


def test_deformable_conv():
    """Test the deformable convolution implementation against PyTorch's reference"""
    
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Define parameters ---
    N, C_in, H_in, W_in = 1, 3, 5, 5
    C_out, K_h, K_w = 2, 3, 3
    stride, padding, dilation = 1, 0, 1

    # Input
    np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
    torch_input = torch.tensor(np_input, dtype=torch.float32)

    # Offset: random offsets
    np_offset = np.random.rand(N, 2 * K_h * K_w, 3, 3).astype(np.float32)
    torch_offset = torch.tensor(np.copy(np_offset), dtype=torch.float32)

    # Mask: ones (no modulation)
    np_mask = np.ones((N, K_h * K_w, 3, 3), dtype=np.float32)
    torch_mask = torch.tensor(np_mask, dtype=torch.float32)

    # Weight: random weights
    np_weight = np.random.rand(C_out, C_in, K_h, K_w).astype(np.float32)
    torch_weight = torch.tensor(np.copy(np_weight), dtype=torch.float32)

    # Bias: zero
    np_bias = np.zeros((C_out), dtype=np.float32)
    torch_bias = torch.tensor(np.copy(np_bias), dtype=torch.float32)

    # --- Run NumPy Deformable Conv ---
    np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight,
                                 stride=stride, padding=padding, dilation=dilation)

    # --- Run PyTorch Deformable Conv ---
    torch_output = deform_conv2d(
        input=torch_input,
        offset=torch_offset,
        weight=torch_weight,
        bias=torch_bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=torch_mask
    )

    # --- Compare outputs ---
    print("NumPy Output:\n", np_output[0, 0])
    print("PyTorch Output:\n", torch_output[0, 0].detach().numpy())

    grade = 0
    if (np.allclose(np_output, torch_output.detach().numpy(), atol=1e-4) == True):
        grade = 30
    else:
        grade = 0
    print(f"\nYour grade is {grade}/30.")
    
    # Additional diagnostics
    max_diff = np.abs(np_output - torch_output.detach().numpy()).max()
    print(f"Maximum difference: {max_diff}")
    
    return grade == 30


if __name__ == "__main__":
    test_deformable_conv()
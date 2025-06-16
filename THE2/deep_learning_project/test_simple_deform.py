import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np

# Test with zero offsets first
def test_zero_offset():
    print("Testing with ZERO offsets (should match regular convolution)")
    print("="*60)
    
    # Simple case
    N, C_in, H_in, W_in = 1, 1, 4, 4
    C_out, K_h, K_w = 1, 2, 2
    
    # Create simple input
    np_input = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    torch_input = torch.tensor(np_input)
    
    # Create simple weight
    np_weight = np.ones((1, 1, 2, 2), dtype=np.float32)
    torch_weight = torch.tensor(np_weight)
    
    # Zero offsets
    H_out = H_in - K_h + 1
    W_out = W_in - K_w + 1
    np_offset = np.zeros((N, 2 * K_h * K_w, H_out, W_out), dtype=np.float32)
    torch_offset = torch.tensor(np_offset)
    
    # Ones mask
    np_mask = np.ones((N, K_h * K_w, H_out, W_out), dtype=np.float32)
    torch_mask = torch.tensor(np_mask)
    
    print("Input:")
    print(np_input[0, 0])
    print("\nWeight (all ones 2x2):")
    print(np_weight[0, 0])
    
    # Run our implementation
    np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight)
    
    # Run PyTorch
    torch_output = deform_conv2d(torch_input, torch_offset, torch_weight, None,
                                stride=1, padding=0, dilation=1, mask=torch_mask)
    
    print("\nNumPy output:")
    print(np_output[0, 0])
    print("\nPyTorch output:")
    print(torch_output[0, 0].detach().numpy())
    
    # Regular conv2d for comparison
    regular_output = torch.nn.functional.conv2d(torch_input, torch_weight)
    print("\nRegular Conv2d output (for comparison):")
    print(regular_output[0, 0].detach().numpy())
    
    print(f"\nDifference: {np.abs(np_output - torch_output.detach().numpy()).max()}")
    
test_zero_offset()

# Test with small offset
def test_small_offset():
    print("\n\nTesting with SMALL offsets")
    print("="*60)
    
    # Simple case
    N, C_in, H_in, W_in = 1, 1, 4, 4
    C_out, K_h, K_w = 1, 2, 2
    
    # Create simple input
    np_input = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    torch_input = torch.tensor(np_input)
    
    # Create simple weight - only top-left corner has weight
    np_weight = np.zeros((1, 1, 2, 2), dtype=np.float32)
    np_weight[0, 0, 0, 0] = 1.0
    torch_weight = torch.tensor(np_weight)
    
    # Small offset only for first kernel position
    H_out = H_in - K_h + 1
    W_out = W_in - K_w + 1
    np_offset = np.zeros((N, 2 * K_h * K_w, H_out, W_out), dtype=np.float32)
    # Set y-offset = 0.5 for first kernel position
    np_offset[0, 0, :, :] = 0.5  # y-offset for position (0,0) of kernel
    torch_offset = torch.tensor(np_offset)
    
    # Ones mask
    np_mask = np.ones((N, K_h * K_w, H_out, W_out), dtype=np.float32)
    torch_mask = torch.tensor(np_mask)
    
    print("Input:")
    print(np_input[0, 0])
    print("\nWeight (only top-left = 1):")
    print(np_weight[0, 0])
    print("\nOffset: y=0.5 for first kernel position")
    
    # Run our implementation
    np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight)
    
    # Run PyTorch
    torch_output = deform_conv2d(torch_input, torch_offset, torch_weight, None,
                                stride=1, padding=0, dilation=1, mask=torch_mask)
    
    print("\nNumPy output:")
    print(np_output[0, 0])
    print("\nPyTorch output:")
    print(torch_output[0, 0].detach().numpy())
    
    print(f"\nDifference: {np.abs(np_output - torch_output.detach().numpy()).max()}")
    
    # Manual calculation for first output position
    # With y-offset=0.5, we sample between rows 0 and 1
    # At position (0,0), we sample between input[0,0]=0 and input[1,0]=4
    # Expected: 0.5 * 0 + 0.5 * 4 = 2.0
    print(f"\nExpected first output value: 0.5 * {np_input[0,0,0,0]} + 0.5 * {np_input[0,0,1,0]} = {0.5 * np_input[0,0,0,0] + 0.5 * np_input[0,0,1,0]}")

test_small_offset()
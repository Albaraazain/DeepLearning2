import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np, bilinear_interpolate

# Test with minimal example
N, C_in, H_in, W_in = 1, 1, 3, 3
C_out, K_h, K_w = 1, 2, 2

# Simple input
np_input = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
print("Input:")
print(np_input[0, 0])

# Simple weight - all ones
np_weight = np.ones((1, 1, 2, 2), dtype=np.float32)
print("\nWeight:")
print(np_weight[0, 0])

# Zero offsets
H_out = 2
W_out = 2
np_offset = np.zeros((N, 2 * K_h * K_w, H_out, W_out), dtype=np.float32)
np_mask = np.ones((N, K_h * K_w, H_out, W_out), dtype=np.float32)

print("\nManual calculation for output[0,0]:")
print("Should sum: input[0:2, 0:2] = [[0,1], [3,4]]")
print("Expected sum: 0+1+3+4 = 8")

# Let's trace through our implementation manually
print("\nTracing through implementation:")

# For output position (0,0)
h_out, w_out = 0, 0
h_start = h_out * 1  # stride=1
w_start = w_out * 1
print(f"Output position ({h_out},{w_out}), start position: ({h_start},{w_start})")

value = 0.0
for kh in range(K_h):
    for kw in range(K_w):
        k = kh * K_w + kw
        
        # Get offsets
        delta_y = np_offset[0, 2 * k, h_out, w_out]
        delta_x = np_offset[0, 2 * k + 1, h_out, w_out]
        m_k = np_mask[0, k, h_out, w_out]
        
        # Sample position
        sample_y = h_start + kh + delta_y
        sample_x = w_start + kw + delta_x
        
        # Get value
        interpolated = bilinear_interpolate(np_input[0, 0], sample_y, sample_x)
        contribution = np_weight[0, 0, kh, kw] * m_k * interpolated
        
        print(f"  Kernel pos ({kh},{kw}): sample at ({sample_y},{sample_x}) = {interpolated}, contribution = {contribution}")
        value += contribution

print(f"Total value: {value}")

# Now run the full implementation
np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight)
print(f"\nImplementation output[0,0]: {np_output[0,0,0,0]}")

# Compare with PyTorch
torch_input = torch.tensor(np_input)
torch_weight = torch.tensor(np_weight)
torch_offset = torch.tensor(np_offset)
torch_mask = torch.tensor(np_mask)

torch_output = deform_conv2d(torch_input, torch_offset, torch_weight, None,
                            stride=1, padding=0, dilation=1, mask=torch_mask)
print(f"PyTorch output[0,0]: {torch_output[0,0,0,0].item()}")

# Also check with regular conv2d
regular_output = torch.nn.functional.conv2d(torch_input, torch_weight)
print(f"Regular conv2d output[0,0]: {regular_output[0,0,0,0].item()}")
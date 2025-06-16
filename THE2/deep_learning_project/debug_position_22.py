import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np, bilinear_interpolate

# Use EXACT same setup as the failing test
np.random.seed(0)
torch.manual_seed(0)

N, C_in, H_in, W_in = 1, 3, 5, 5
C_out, K_h, K_w = 2, 3, 3
stride, padding, dilation = 1, 0, 1

# Input
np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
torch_input = torch.tensor(np_input, dtype=torch.float32)

# Offset
np_offset = np.random.rand(N, 2 * K_h * K_w, 3, 3).astype(np.float32)
torch_offset = torch.tensor(np.copy(np_offset), dtype=torch.float32)

# Mask
np_mask = np.ones((N, K_h * K_w, 3, 3), dtype=np.float32)
torch_mask = torch.tensor(np_mask, dtype=torch.float32)

# Weight
np_weight = np.random.rand(C_out, C_in, K_h, K_w).astype(np.float32)
torch_weight = torch.tensor(np.copy(np_weight), dtype=torch.float32)

# Focus on the failing position: output[0, 0, 2, 2]
h_out, w_out = 2, 2
c_out = 0

print(f"Debugging position output[0, {c_out}, {h_out}, {w_out}]")
print("="*60)

h_start = h_out * stride  # = 2
w_start = w_out * stride  # = 2

print(f"Base position in input: ({h_start}, {w_start})")
print("\nKernel sampling positions:")

# Compute manually
manual_value = 0.0

for kh in range(K_h):
    for kw in range(K_w):
        k = kh * K_w + kw
        
        delta_y = np_offset[0, 2 * k, h_out, w_out]
        delta_x = np_offset[0, 2 * k + 1, h_out, w_out]
        m_k = np_mask[0, k, h_out, w_out]
        
        sample_y = h_start + kh * dilation + delta_y
        sample_x = w_start + kw * dilation + delta_x
        
        print(f"\nKernel position ({kh},{kw}), index {k}:")
        print(f"  Offset: y={delta_y:.4f}, x={delta_x:.4f}")
        print(f"  Sample position: ({sample_y:.4f}, {sample_x:.4f})")
        
        # Check if out of bounds
        if sample_y < 0 or sample_y > 4 or sample_x < 0 or sample_x > 4:
            print(f"  WARNING: Sample position out of bounds [0,4]!")
        
        # Sum contributions from all input channels
        kernel_sum = 0.0
        for c_in in range(C_in):
            interpolated = bilinear_interpolate(np_input[0, c_in, :, :], sample_y, sample_x)
            weight = np_weight[c_out, c_in, kh, kw]
            contribution = weight * m_k * interpolated
            kernel_sum += contribution
            
            if c_in == 0:  # Print details for first channel
                print(f"  Channel 0: interp={interpolated:.4f}, weight={weight:.4f}")
        
        print(f"  Total contribution from this kernel position: {kernel_sum:.4f}")
        manual_value += kernel_sum

print(f"\nManual calculation result: {manual_value:.6f}")

# Get actual outputs
np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight,
                             stride=stride, padding=padding, dilation=dilation)

torch_output = deform_conv2d(
    input=torch_input,
    offset=torch_offset,
    weight=torch_weight,
    bias=None,
    stride=stride,
    padding=padding,
    dilation=dilation,
    mask=torch_mask
)

print(f"\nNumPy implementation: {np_output[0, c_out, h_out, w_out]:.6f}")
print(f"PyTorch: {torch_output[0, c_out, h_out, w_out].item():.6f}")
print(f"Difference: {np_output[0, c_out, h_out, w_out] - torch_output[0, c_out, h_out, w_out].item():.6f}")

# Check if manual matches our implementation
print(f"\nManual matches NumPy: {np.isclose(manual_value, np_output[0, c_out, h_out, w_out])}")

# Let's check the corner case - position (2,2) with kernel (2,2) samples at (4.x, 4.x)
print("\n" + "="*60)
print("Checking boundary handling for bottom-right kernel position:")
k = 8  # kernel position (2,2)
delta_y = np_offset[0, 2 * k, h_out, w_out]
delta_x = np_offset[0, 2 * k + 1, h_out, w_out]
sample_y = 2 + 2 + delta_y  # 4 + delta_y
sample_x = 2 + 2 + delta_x  # 4 + delta_x
print(f"Sampling at ({sample_y:.4f}, {sample_x:.4f})")
print(f"This is beyond the boundary (4,4) of a 5x5 image")

# Test what happens with out-of-bounds sampling
test_y, test_x = 4.8, 4.8
our_interp = bilinear_interpolate(np_input[0, 0], test_y, test_x)
print(f"\nOur interpolation at ({test_y}, {test_x}): {our_interp}")
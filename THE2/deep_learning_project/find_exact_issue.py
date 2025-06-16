import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np

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

# Run both
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

print("Shapes:")
print(f"Input: {np_input.shape}")
print(f"Weight: {np_weight.shape}")
print(f"Offset: {np_offset.shape}")
print(f"Output: {np_output.shape}")

# Find positions with largest differences
diff = np.abs(np_output - torch_output.detach().numpy())
max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
print(f"\nMax difference at position {max_diff_idx}: {diff[max_diff_idx]}")

# Let's trace this specific position
n, c_out, h_out, w_out = max_diff_idx
print(f"\nTracing position ({h_out}, {w_out}) for output channel {c_out}:")

# Check first that offsets and weights are identical
print("\nVerifying inputs are identical:")
print(f"Offset tensor equal: {np.allclose(np_offset, torch_offset.numpy())}")
print(f"Weight tensor equal: {np.allclose(np_weight, torch_weight.numpy())}")
print(f"Input tensor equal: {np.allclose(np_input, torch_input.numpy())}")

# Now let's compute just the first output channel at position (0,0) to see if it matches
print("\n" + "="*50)
print("Computing output[0,0,0,0] step by step")

# Our computation
h_start = 0
w_start = 0
value = 0.0

for kh in range(K_h):
    for kw in range(K_w):
        k = kh * K_w + kw
        delta_y = np_offset[0, 2 * k, 0, 0]
        delta_x = np_offset[0, 2 * k + 1, 0, 0]
        m_k = np_mask[0, k, 0, 0]
        
        sample_y = h_start + kh * dilation + delta_y
        sample_x = w_start + kw * dilation + delta_x
        
        # Sum over input channels
        for c_in in range(C_in):
            from deformable_conv import bilinear_interpolate
            interpolated = bilinear_interpolate(np_input[0, c_in, :, :], sample_y, sample_x)
            contribution = np_weight[0, c_in, kh, kw] * m_k * interpolated
            value += contribution
            if c_in == 0 and (kh == 0 and kw == 0):  # First position, first channel
                print(f"First contribution: weight={np_weight[0, c_in, kh, kw]:.4f}, "
                      f"interp={interpolated:.4f}, contrib={contribution:.4f}")

print(f"\nOur output[0,0,0,0]: {value}")
print(f"NumPy impl output[0,0,0,0]: {np_output[0,0,0,0]}")
print(f"PyTorch output[0,0,0,0]: {torch_output[0,0,0,0].item()}")

# The values should match between our manual calculation and NumPy implementation
print(f"\nManual matches NumPy impl: {np.isclose(value, np_output[0,0,0,0])}")

# Let's check if the issue is related to floating point precision
print("\n" + "="*50)
print("Checking data types:")
print(f"NumPy offset dtype: {np_offset.dtype}")
print(f"NumPy weight dtype: {np_weight.dtype}")
print(f"NumPy input dtype: {np_input.dtype}")
print(f"PyTorch offset dtype: {torch_offset.dtype}")
print(f"PyTorch weight dtype: {torch_weight.dtype}")
print(f"PyTorch input dtype: {torch_input.dtype}")
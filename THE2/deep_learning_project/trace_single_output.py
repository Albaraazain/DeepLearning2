import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np, bilinear_interpolate

# Let's trace a single output position in detail
np.random.seed(0)
torch.manual_seed(0)

# Simplified case: 1 channel input, 1 channel output
N, C_in, H_in, W_in = 1, 1, 5, 5
C_out, K_h, K_w = 1, 3, 3

# Simple input
np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
torch_input = torch.tensor(np_input, dtype=torch.float32)

print("Input:")
print(np_input[0, 0])

# Simple weight - all ones to sum all contributions
np_weight = np.ones((C_out, C_in, K_h, K_w), dtype=np.float32)
torch_weight = torch.tensor(np.copy(np_weight), dtype=torch.float32)

# Use the same random offsets
np_offset = np.random.rand(N, 2 * K_h * K_w, 3, 3).astype(np.float32)
torch_offset = torch.tensor(np.copy(np_offset), dtype=torch.float32)

np_mask = np.ones((N, K_h * K_w, 3, 3), dtype=np.float32)
torch_mask = torch.tensor(np_mask, dtype=torch.float32)

# Let's manually compute output[0,0,0,0]
h_out, w_out = 0, 0
h_start = 0
w_start = 0

print(f"\nManually computing output at position ({h_out},{w_out}):")
print("="*50)

manual_sum = 0.0
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
        
        # Bilinear interpolation
        interpolated = bilinear_interpolate(np_input[0, 0], sample_y, sample_x)
        
        # Contribution
        contribution = np_weight[0, 0, kh, kw] * m_k * interpolated
        manual_sum += contribution
        
        print(f"Kernel pos ({kh},{kw}): offset=({delta_y:.4f},{delta_x:.4f})")
        print(f"  Sample at ({sample_y:.4f},{sample_x:.4f}) = {interpolated:.4f}")
        print(f"  Weight={np_weight[0,0,kh,kw]}, mask={m_k}, contribution={contribution:.4f}")

print(f"\nManual sum: {manual_sum:.6f}")

# Now run both implementations
np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight)
torch_output = deform_conv2d(torch_input, torch_offset, torch_weight, None,
                            stride=1, padding=0, dilation=1, mask=torch_mask)

print(f"\nNumPy implementation output[0,0,0,0]: {np_output[0,0,0,0]:.6f}")
print(f"PyTorch output[0,0,0,0]: {torch_output[0,0,0,0].item():.6f}")

# Check if our manual calculation matches our implementation
print(f"\nDoes manual match NumPy? {np.isclose(manual_sum, np_output[0,0,0,0])}")
print(f"Does manual match PyTorch? {np.isclose(manual_sum, torch_output[0,0,0,0].item())}")

# Let's also check with torch's grid_sample to understand interpolation
print("\n" + "="*50)
print("Testing PyTorch's grid_sample interpolation")

# Create a grid for a single sample point
test_y, test_x = 0.5488, 0.3834  # First offset values
grid = torch.zeros(1, 1, 1, 2)
# grid_sample expects coordinates in [-1, 1] range
# Convert from pixel coordinates to normalized coordinates
norm_x = (test_x / (W_in - 1)) * 2 - 1
norm_y = (test_y / (H_in - 1)) * 2 - 1
grid[0, 0, 0, 0] = norm_x
grid[0, 0, 0, 1] = norm_y

# Add batch and channel dimensions to input
input_4d = torch_input.unsqueeze(0)  # Now [1, 1, 1, 5, 5]
sampled = torch.nn.functional.grid_sample(input_4d[0:1], grid, align_corners=True, padding_mode='border')
print(f"PyTorch grid_sample at ({test_y:.4f},{test_x:.4f}): {sampled[0,0,0,0].item():.4f}")

# Our bilinear interpolation
our_interp = bilinear_interpolate(np_input[0, 0], test_y, test_x)
print(f"Our bilinear interpolation: {our_interp:.4f}")
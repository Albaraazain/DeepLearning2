import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np

# Let's trace through what happens with the actual test case
np.random.seed(0)
torch.manual_seed(0)

# Same setup as the test
N, C_in, H_in, W_in = 1, 3, 5, 5
C_out, K_h, K_w = 2, 3, 3
stride, padding, dilation = 1, 0, 1

# Input
np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
torch_input = torch.tensor(np_input, dtype=torch.float32)

# Let's use VERY SMALL offsets first
np_offset = np.random.rand(N, 2 * K_h * K_w, 3, 3).astype(np.float32) * 0.1  # Scale down by 10x
torch_offset = torch.tensor(np.copy(np_offset), dtype=torch.float32)

np_mask = np.ones((N, K_h * K_w, 3, 3), dtype=np.float32)
torch_mask = torch.tensor(np_mask, dtype=torch.float32)

np_weight = np.random.rand(C_out, C_in, K_h, K_w).astype(np.float32)
torch_weight = torch.tensor(np.copy(np_weight), dtype=torch.float32)

print("Testing with SMALL offsets (scaled by 0.1)")
print("Max offset value:", np_offset.max())

# Run both
np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight)
torch_output = deform_conv2d(torch_input, torch_offset, torch_weight, None,
                            stride=stride, padding=padding, dilation=dilation, mask=torch_mask)

diff = np.abs(np_output - torch_output.detach().numpy()).max()
print(f"Max difference with small offsets: {diff}")

# Now let's check what happens at boundaries
print("\n" + "="*50)
print("Checking boundary handling")

# Create offset that samples near boundaries
np_offset2 = np.zeros((N, 2 * K_h * K_w, 3, 3), dtype=np.float32)

# For output position (2, 2) - bottom right
# When kernel is at position (2,2) with kernel size 3x3,
# the bottom-right kernel position naturally samples at (4,4) which is the boundary
# Let's add a small positive offset to push it slightly out of bounds

# Bottom-right kernel position is index 8
np_offset2[0, 2*8, 2, 2] = 0.5    # y-offset 
np_offset2[0, 2*8+1, 2, 2] = 0.5  # x-offset

torch_offset2 = torch.tensor(np.copy(np_offset2), dtype=torch.float32)

print("\nTesting with offset at boundary position (2,2):")
print("Bottom-right kernel position will sample at (4.5, 4.5) - out of bounds")

np_output2 = deform_conv2d_np(np_input, np_offset2, np_mask, np_weight)
torch_output2 = deform_conv2d(torch_input, torch_offset2, torch_weight, None,
                             stride=stride, padding=padding, dilation=dilation, mask=torch_mask)

print("NumPy output[0,0,2,2]:", np_output2[0, 0, 2, 2])
print("PyTorch output[0,0,2,2]:", torch_output2[0, 0, 2, 2].item())

# Let's also manually check what bilinear interpolation gives
from deformable_conv import bilinear_interpolate

# For first channel, sampling at (4.5, 4.5)
manual_interp = bilinear_interpolate(np_input[0, 0], 4.5, 4.5)
print(f"\nManual bilinear interpolation at (4.5, 4.5): {manual_interp}")
print("This should be same as input[4,4] since we clip to bounds:", np_input[0, 0, 4, 4])

# Check if PyTorch does something different at boundaries
print("\n" + "="*50)
print("Testing edge case: Large offsets")

# Use original random offsets (which are up to ~1.0)
np.random.seed(0)
torch.manual_seed(0)
np_offset3 = np.random.rand(N, 2 * K_h * K_w, 3, 3).astype(np.float32)

# Let's trace specific positions that have large differences
# From earlier debug, we know position (0,2) has issues
print("\nChecking specific position (0,2) with large offsets:")
print("Offsets for each kernel position at output (0,2):")
for k in range(9):
    y_off = np_offset3[0, 2*k, 0, 2]
    x_off = np_offset3[0, 2*k+1, 0, 2]
    kh = k // 3
    kw = k % 3
    # For output (0,2), input base position is (0,2)
    sample_y = 0 + kh + y_off
    sample_x = 2 + kw + x_off
    print(f"  Kernel ({kh},{kw}): offset=({y_off:.3f},{x_off:.3f}), samples at ({sample_y:.3f},{sample_x:.3f})")
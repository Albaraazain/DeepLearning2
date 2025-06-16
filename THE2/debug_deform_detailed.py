import numpy as np
import torch
from torchvision.ops import deform_conv2d

def bilinear_interpolate(a_l, q_y, q_x):
    """
    Perform bilinear interpolation on the input activation map at the given (fractional) coordinates.
    """
    H, W = a_l.shape
    
    # Check bounds - if outside, return 0
    if q_y < 0 or q_y > H - 1 or q_x < 0 or q_x > W - 1:
        return 0.0
    
    # Get the four nearest integer positions
    p_lt_y = int(np.floor(q_y))  # left top y
    p_lt_x = int(np.floor(q_x))  # left top x
    p_rb_y = int(np.ceil(q_y))   # right bottom y
    p_rb_x = int(np.ceil(q_x))   # right bottom x
    
    # Ensure coordinates are within bounds
    p_lt_y = np.clip(p_lt_y, 0, H - 1)
    p_lt_x = np.clip(p_lt_x, 0, W - 1)
    p_rb_y = np.clip(p_rb_y, 0, H - 1)
    p_rb_x = np.clip(p_rb_x, 0, W - 1)
    
    # Get the four corner values
    v_lt = a_l[p_lt_y, p_lt_x]  # left top
    v_rt = a_l[p_lt_y, p_rb_x] if p_rb_x < W else 0  # right top
    v_lb = a_l[p_rb_y, p_lt_x] if p_rb_y < H else 0  # left bottom
    v_rb = a_l[p_rb_y, p_rb_x] if p_rb_y < H and p_rb_x < W else 0  # right bottom
    
    # Calculate fractional parts
    frac_y = q_y - np.floor(q_y)
    frac_x = q_x - np.floor(q_x)
    
    # Calculate weights
    w_lt = (1 - frac_x) * (1 - frac_y)
    w_rt = frac_x * (1 - frac_y)
    w_lb = (1 - frac_x) * frac_y
    w_rb = frac_x * frac_y
    
    # Compute interpolated value
    out = v_lt * w_lt + v_rt * w_rt + v_lb * w_lb + v_rb * w_rb
    
    return out

# Test with simple case
np.random.seed(0)
torch.manual_seed(0)

# Smaller test case to debug
N, C_in, H_in, W_in = 1, 1, 4, 4
C_out, K_h, K_w = 1, 2, 2
stride, padding, dilation = 1, 0, 1

np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
print("Input:\n", np_input[0, 0])

# Zero offsets for easier debugging
np_offset = np.zeros((N, 2 * K_h * K_w, 3, 3), dtype=np.float32)
# Add small offset to see effect
np_offset[0, 0, 0, 2] = 0.5  # dy for first kernel position at output (0,2)
np_offset[0, 1, 0, 2] = 0.5  # dx for first kernel position at output (0,2)

torch_offset = torch.tensor(np_offset, dtype=torch.float32)

np_mask = np.ones((N, K_h * K_w, 3, 3), dtype=np.float32)
torch_mask = torch.tensor(np_mask, dtype=torch.float32)

np_weight = np.ones((C_out, C_in, K_h, K_w), dtype=np.float32)
torch_weight = torch.tensor(np_weight, dtype=torch.float32)

torch_bias = torch.zeros(C_out)

# Calculate expected output dimensions
H_out = (H_in - dilation * (K_h - 1) - 1) // stride + 1
W_out = (W_in - dilation * (K_w - 1) - 1) // stride + 1
print(f"\nOutput dimensions: {H_out} x {W_out}")

# Manual calculation for position (0, 2)
print("\nManual calculation for output position (0, 2):")
print("Base position: (0, 2)")
print("Kernel positions sampled:")
print("  k=0 (0,0): sample at (0+0.5, 2+0.5) = (0.5, 2.5)")
print("  k=1 (0,1): sample at (0, 3)")
print("  k=2 (1,0): sample at (1, 2)")
print("  k=3 (1,1): sample at (1, 3)")

# Test bilinear interpolation
print("\nBilinear interpolation at (0.5, 2.5):")
val = bilinear_interpolate(np_input[0, 0], 0.5, 2.5)
print(f"  Value: {val}")
print(f"  Expected: {(np_input[0,0,0,2] + np_input[0,0,0,3] + np_input[0,0,1,2] + np_input[0,0,1,3])/4}")

# Run PyTorch
torch_output = deform_conv2d(
    input=torch.tensor(np_input),
    offset=torch_offset,
    weight=torch_weight,
    bias=torch_bias,
    stride=stride,
    padding=padding,
    dilation=dilation,
    mask=torch_mask
)

print("\nPyTorch output:\n", torch_output[0, 0].detach().numpy())
import numpy as np
import torch
from torchvision.ops import deform_conv2d

# Set seeds
np.random.seed(0)
torch.manual_seed(0)

# Simple test case
N, C_in, H_in, W_in = 1, 1, 3, 3
C_out, K_h, K_w = 1, 2, 2
stride, padding, dilation = 1, 0, 1

# Simple input
np_input = np.array([[[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]]], dtype=np.float32)
torch_input = torch.tensor(np_input)

# Zero offset (no deformation)
np_offset = np.zeros((N, 2 * K_h * K_w, 2, 2), dtype=np.float32)
torch_offset = torch.tensor(np_offset)

# Ones mask
np_mask = np.ones((N, K_h * K_w, 2, 2), dtype=np.float32)
torch_mask = torch.tensor(np_mask)

# Simple weight
np_weight = np.ones((C_out, C_in, K_h, K_w), dtype=np.float32)
torch_weight = torch.tensor(np_weight)

# Zero bias
torch_bias = torch.zeros(C_out)

print("Input shape:", np_input.shape)
print("Input:\n", np_input[0, 0])
print("\nOffset shape:", np_offset.shape)
print("Mask shape:", np_mask.shape)
print("Weight shape:", np_weight.shape)
print("Weight:\n", np_weight[0, 0])

# Test bilinear interpolation
def bilinear_interpolate(a_l, q_y, q_x):
    H, W = a_l.shape
    
    p_lt_y = int(np.floor(q_y))
    p_lt_x = int(np.floor(q_x))
    p_rb_y = int(np.ceil(q_y))
    p_rb_x = int(np.ceil(q_x))
    
    p_lt_y = np.clip(p_lt_y, 0, H - 1)
    p_lt_x = np.clip(p_lt_x, 0, W - 1)
    p_rb_y = np.clip(p_rb_y, 0, H - 1)
    p_rb_x = np.clip(p_rb_x, 0, W - 1)
    
    v_lt = a_l[p_lt_y, p_lt_x]
    v_rt = a_l[p_lt_y, p_rb_x]
    v_lb = a_l[p_rb_y, p_lt_x]
    v_rb = a_l[p_rb_y, p_rb_x]
    
    w_lt = (1 - abs(p_lt_x - q_x)) * (1 - abs(p_lt_y - q_y))
    w_rt = (1 - abs(p_rb_x - q_x)) * (1 - abs(p_lt_y - q_y))
    w_lb = (1 - abs(p_lt_x - q_x)) * (1 - abs(p_rb_y - q_y))
    w_rb = (1 - abs(p_rb_x - q_x)) * (1 - abs(p_rb_y - q_y))
    
    out = v_lt * w_lt + v_rt * w_rt + v_lb * w_lb + v_rb * w_rb
    
    return out

# Test interpolation at integer positions
test_img = np_input[0, 0]
print("\nTesting bilinear interpolation:")
print(f"At (0, 0): {bilinear_interpolate(test_img, 0, 0)} (expected: {test_img[0, 0]})")
print(f"At (1, 1): {bilinear_interpolate(test_img, 1, 1)} (expected: {test_img[1, 1]})")
print(f"At (0.5, 0.5): {bilinear_interpolate(test_img, 0.5, 0.5)} (expected: {(1+2+4+5)/4})")

# PyTorch reference
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

print("\nPyTorch output shape:", torch_output.shape)
print("PyTorch output:\n", torch_output[0, 0].detach().numpy())
print("\nExpected (manual calculation for top-left):")
print("Position (0,0): 1*1 + 2*1 + 4*1 + 5*1 = 12")
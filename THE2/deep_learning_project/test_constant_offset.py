import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np

# Test with CONSTANT small offsets to isolate the issue
np.random.seed(0)
torch.manual_seed(0)

# --- Define parameters ---
N, C_in, H_in, W_in = 1, 3, 5, 5
C_out, K_h, K_w = 2, 3, 3
stride, padding, dilation = 1, 0, 1

# Input
np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
torch_input = torch.tensor(np_input, dtype=torch.float32)

# CONSTANT offset - all positions have same offset
np_offset = np.zeros((N, 2 * K_h * K_w, 3, 3), dtype=np.float32)
# Set all y-offsets to 0.2 and x-offsets to 0.3
for k in range(K_h * K_w):
    np_offset[0, 2*k, :, :] = 0.2    # y-offset
    np_offset[0, 2*k+1, :, :] = 0.3  # x-offset
    
torch_offset = torch.tensor(np.copy(np_offset), dtype=torch.float32)

print("Using constant offsets: y=0.2, x=0.3 for all positions")

# Mask
np_mask = np.ones((N, K_h * K_w, 3, 3), dtype=np.float32)
torch_mask = torch.tensor(np_mask, dtype=torch.float32)

# Weight
np_weight = np.random.rand(C_out, C_in, K_h, K_w).astype(np.float32)
torch_weight = torch.tensor(np.copy(np_weight), dtype=torch.float32)

# Bias
np_bias = np.zeros((C_out), dtype=np.float32)
torch_bias = torch.tensor(np.copy(np_bias), dtype=torch.float32)

# --- Run both implementations ---
np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight,
                             stride=stride, padding=padding, dilation=dilation)

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

print("\nNumPy Output channel 0:")
print(np_output[0, 0])
print("\nPyTorch Output channel 0:")
print(torch_output[0, 0].detach().numpy())

diff = np.abs(np_output - torch_output.detach().numpy()).max()
print(f"\nMax difference: {diff}")

if diff < 1e-4:
    print("✅ SUCCESS with constant offsets!")
else:
    print("❌ Still failing with constant offsets")
    
# Now test with ZERO offsets
print("\n" + "="*60)
print("Testing with ZERO offsets")

np_offset_zero = np.zeros((N, 2 * K_h * K_w, 3, 3), dtype=np.float32)
torch_offset_zero = torch.tensor(np_offset_zero)

np_output_zero = deform_conv2d_np(np_input, np_offset_zero, np_mask, np_weight,
                                  stride=stride, padding=padding, dilation=dilation)

torch_output_zero = deform_conv2d(
    input=torch_input,
    offset=torch_offset_zero,
    weight=torch_weight,
    bias=torch_bias,
    stride=stride,
    padding=padding,
    dilation=dilation,
    mask=torch_mask
)

print("\nWith zero offsets - NumPy output:")
print(np_output_zero[0, 0])
print("\nWith zero offsets - PyTorch output:")
print(torch_output_zero[0, 0].detach().numpy())

diff_zero = np.abs(np_output_zero - torch_output_zero.detach().numpy()).max()
print(f"\nMax difference with zero offsets: {diff_zero}")

if diff_zero < 1e-4:
    print("✅ SUCCESS with zero offsets!")
else:
    print("❌ Failing even with zero offsets")
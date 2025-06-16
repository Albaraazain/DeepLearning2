import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np

np.random.seed(0)
torch.manual_seed(0)

# --- Define parameters ---
N, C_in, H_in, W_in = 1, 3, 5, 5
C_out, K_h, K_w = 2, 3, 3
stride, padding, dilation = 1, 0, 1

# Input
np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
torch_input = torch.tensor(np_input, dtype=torch.float32)

# Offset
np_offset = np.random.rand(N, 2 * K_h * K_w, 3, 3).astype(np.float32)
torch_offset = torch.tensor(np.copy(np_offset), dtype=torch.float32)

# Check offset shapes and values
print("Offset shape:", np_offset.shape)
print("Expected shape: (1, 18, 3, 3)")
print("2 * K_h * K_w =", 2 * K_h * K_w)

# Let's check if offsets are the same for different output positions
print("\nChecking offsets for output position (0,0):")
for k in range(9):
    y_offset = np_offset[0, 2*k, 0, 0]
    x_offset = np_offset[0, 2*k+1, 0, 0]
    print(f"  Kernel pos {k}: y_offset={y_offset:.4f}, x_offset={x_offset:.4f}")

print("\nChecking offsets for output position (0,1):")
for k in range(9):
    y_offset = np_offset[0, 2*k, 0, 1]
    x_offset = np_offset[0, 2*k+1, 0, 1]
    print(f"  Kernel pos {k}: y_offset={y_offset:.4f}, x_offset={x_offset:.4f}")

print("\nChecking offsets for output position (0,2):")
for k in range(9):
    y_offset = np_offset[0, 2*k, 0, 2]
    x_offset = np_offset[0, 2*k+1, 0, 2]
    print(f"  Kernel pos {k}: y_offset={y_offset:.4f}, x_offset={x_offset:.4f}")

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

# Check differences per position
diff = np_output - torch_output.detach().numpy()
print("\nDifference per position (channel 0):")
print(diff[0, 0])
print("\nMax abs difference per column:")
for col in range(3):
    print(f"  Column {col}: {np.abs(diff[0, 0, :, col]).max()}")
import numpy as np
import torch
from torchvision.ops import deform_conv2d
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np

# Test offset ordering
def test_offset_order():
    # Simple test case
    input_tensor = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    weight = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    
    # Test with y-offset only
    offset = torch.zeros(1, 2, 5, 5, dtype=torch.float32)
    offset[0, 0, :, :] = 1.0  # y offset = 1
    offset[0, 1, :, :] = 0.0  # x offset = 0
    
    mask = torch.ones(1, 1, 5, 5, dtype=torch.float32)
    
    output = deform_conv2d(input_tensor, offset, weight, None, 
                          stride=1, padding=0, dilation=1, mask=mask)
    
    print("PyTorch deform_conv2d with y-offset=1, x-offset=0:")
    print("Input:\n", input_tensor[0, 0])
    print("Output:\n", output[0, 0])
    print("Expected: values from next row")
    
test_offset_order()

# Now test our implementation
np.random.seed(0)
torch.manual_seed(0)

# --- Define parameters ---
N, C_in, H_in, W_in = 1, 3, 5, 5
C_out, K_h, K_w = 2, 3, 3
stride, padding, dilation = 1, 0, 1

# Input
np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
torch_input = torch.tensor(np_input, dtype=torch.float32)

# Offset: random offsets
np_offset = np.random.rand(N, 2 * K_h * K_w, 3, 3).astype(np.float32)
torch_offset = torch.tensor(np.copy(np_offset), dtype=torch.float32)

# Mask: ones (no modulation)
np_mask = np.ones((N, K_h * K_w, 3, 3), dtype=np.float32)
torch_mask = torch.tensor(np_mask, dtype=torch.float32)

# Weight: random weights
np_weight = np.random.rand(C_out, C_in, K_h, K_w).astype(np.float32)
torch_weight = torch.tensor(np.copy(np_weight), dtype=torch.float32)

# Bias: zero
np_bias = np.zeros((C_out), dtype=np.float32)
torch_bias = torch.tensor(np.copy(np_bias), dtype=torch.float32)

print("\n" + "="*50)
print("Testing with random offsets")
print("="*50)

# Check offset values
print("First few offset values:")
print(f"Offset[0,0,0,0] (should be y): {np_offset[0,0,0,0]}")
print(f"Offset[0,1,0,0] (should be x): {np_offset[0,1,0,0]}")
print(f"Offset[0,2,0,0] (should be y): {np_offset[0,2,0,0]}")
print(f"Offset[0,3,0,0] (should be x): {np_offset[0,3,0,0]}")

# --- Run NumPy Deformable Conv ---
np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight,
                             stride=stride, padding=padding, dilation=dilation)

# --- Run PyTorch Deformable Conv ---
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

# --- Compare outputs ---
print("\nNumPy Output:\n", np_output[0, 0])
print("PyTorch Output:\n", torch_output[0, 0].detach().numpy())
print("Max difference:", np.abs(np_output - torch_output.detach().numpy()).max())
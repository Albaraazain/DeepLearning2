import torch
from torchvision.ops import deform_conv2d
import numpy as np

# Let's understand PyTorch's offset interpretation more deeply
def test_pytorch_offset_behavior():
    """Test to understand exactly how PyTorch interprets offsets"""
    
    # Create a simple 5x5 input with distinct values
    input_tensor = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    print("Input tensor:")
    print(input_tensor[0, 0])
    
    # 3x3 kernel with only center position having weight
    weight = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    weight[0, 0, 1, 1] = 1.0  # Only center of kernel has weight
    print("\nWeight (only center=1):")
    print(weight[0, 0])
    
    # Test 1: No offset - should sample from center of each 3x3 region
    offset = torch.zeros(1, 2 * 3 * 3, 3, 3, dtype=torch.float32)
    mask = torch.ones(1, 3 * 3, 3, 3, dtype=torch.float32)
    
    output = deform_conv2d(input_tensor, offset, weight, None, 
                          stride=1, padding=0, dilation=1, mask=mask)
    
    print("\nTest 1 - No offset (center kernel only):")
    print("Output:")
    print(output[0, 0])
    print("This samples from positions: (1,1)=6, (1,2)=7, (1,3)=8, etc.")
    
    # Test 2: Offset only the center position
    offset = torch.zeros(1, 2 * 3 * 3, 3, 3, dtype=torch.float32)
    # Center position is kernel index 4 (for 3x3 kernel: 0,1,2,3,4,5,6,7,8)
    # Set y-offset for center
    offset[0, 2*4, :, :] = 1.0     # y-offset = 1
    offset[0, 2*4+1, :, :] = 0.0   # x-offset = 0
    
    output2 = deform_conv2d(input_tensor, offset, weight, None, 
                           stride=1, padding=0, dilation=1, mask=mask)
    
    print("\nTest 2 - Center position offset by (y=1, x=0):")
    print("Output:")
    print(output2[0, 0])
    print("This should sample from one row below: (2,1)=11, (2,2)=12, (2,3)=13")
    
    # Test 3: Let's verify the kernel indexing
    print("\n\nKernel position indexing (3x3):")
    for kh in range(3):
        for kw in range(3):
            k = kh * 3 + kw
            print(f"Position ({kh},{kw}) -> index {k}")

# Run the test
test_pytorch_offset_behavior()

# Now let's test with our implementation to see the difference
print("\n" + "="*60)
print("Testing our implementation with same setup")
print("="*60)

import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import deform_conv2d_np

# Same test but with numpy
np_input = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
np_weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
np_weight[0, 0, 1, 1] = 1.0

# No offset
np_offset = np.zeros((1, 2 * 3 * 3, 3, 3), dtype=np.float32)
np_mask = np.ones((1, 3 * 3, 3, 3), dtype=np.float32)

np_output = deform_conv2d_np(np_input, np_offset, np_mask, np_weight)
print("NumPy output (no offset):")
print(np_output[0, 0])

# With center offset
np_offset[0, 2*4, :, :] = 1.0     # y-offset = 1
np_offset[0, 2*4+1, :, :] = 0.0   # x-offset = 0

np_output2 = deform_conv2d_np(np_input, np_offset, np_mask, np_weight)
print("\nNumPy output (center offset y=1):")
print(np_output2[0, 0])
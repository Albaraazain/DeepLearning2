import numpy as np
import torch
from torchvision.ops import deform_conv2d

# Let's test with a simple case to understand the offset ordering
def test_offset_ordering():
    """Test to definitively determine offset ordering in PyTorch"""
    
    # Create a simple 5x5 input with values 0-24
    input_tensor = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    print("Input tensor:")
    print(input_tensor[0, 0])
    
    # Create a 1x1 kernel with weight 1.0
    weight = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    
    # Create offset that shifts by (1, 0) - one pixel down
    # If offset[0] is y and offset[1] is x, this should sample from row 1
    offset = torch.zeros(1, 2, 5, 5, dtype=torch.float32)
    offset[0, 0, :, :] = 1.0  # First channel
    offset[0, 1, :, :] = 0.0  # Second channel
    
    # No mask modulation
    mask = torch.ones(1, 1, 5, 5, dtype=torch.float32)
    
    output = deform_conv2d(input_tensor, offset, weight, None, 
                          stride=1, padding=0, dilation=1, mask=mask)
    
    print("\nWith offset first_channel=1, second_channel=0:")
    print("Output:")
    print(output[0, 0])
    print("If first channel is Y: should sample from next row (5,6,7,8,9)")
    print("If first channel is X: should sample from next column (1,2,3,4,0)")
    
test_offset_ordering()
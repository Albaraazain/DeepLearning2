import numpy as np
import torch
from torchvision.ops import deform_conv2d

# First, let's understand PyTorch's deformable convolution behavior
def test_simple_case():
    """Test with simple, predictable values to understand the behavior"""
    
    # Simple 3x3 input, 2x2 kernel
    input_tensor = torch.tensor([[[[1., 2.],
                                   [3., 4.]]]], dtype=torch.float32)
    
    # No offset (should behave like regular convolution)
    offset = torch.zeros(1, 2 * 2 * 2, 1, 1, dtype=torch.float32)  # 2x2 kernel -> 8 offset values
    
    # Weight: identity-like
    weight = torch.ones(1, 1, 2, 2, dtype=torch.float32)
    
    # No mask modulation
    mask = torch.ones(1, 2 * 2, 1, 1, dtype=torch.float32)
    
    # No bias
    bias = torch.zeros(1, dtype=torch.float32)
    
    output = deform_conv2d(input_tensor, offset, weight, bias, 
                          stride=1, padding=0, dilation=1, mask=mask)
    
    print("Simple test - no offset:")
    print("Input:\n", input_tensor[0, 0])
    print("Output:", output[0, 0].item())
    print("Expected (sum of all elements):", 1+2+3+4)
    
    # Now test with small offset
    offset[0, 0, 0, 0] = 0.5  # x offset for first kernel position
    offset[0, 1, 0, 0] = 0.0  # y offset for first kernel position
    
    output_with_offset = deform_conv2d(input_tensor, offset, weight, bias, 
                                      stride=1, padding=0, dilation=1, mask=mask)
    
    print("\nWith offset (0.5, 0) for first kernel position:")
    print("Output:", output_with_offset[0, 0].item())
    
    return input_tensor, weight, offset, mask

def check_offset_ordering():
    """Check how PyTorch orders offsets"""
    N, C_in, H_in, W_in = 1, 1, 5, 5
    C_out, K_h, K_w = 1, 3, 3
    
    # Create simple input
    input_tensor = torch.arange(N * C_in * H_in * W_in, dtype=torch.float32).reshape(N, C_in, H_in, W_in)
    
    # Create offset tensor with specific pattern
    offset = torch.zeros(N, 2 * K_h * K_w, 3, 3, dtype=torch.float32)
    
    # Set offset for first kernel position (0, 0) at output position (0, 0)
    # Try x offset
    offset[0, 0, 0, 0] = 1.0  # This should be x offset
    offset[0, 1, 0, 0] = 0.0  # This should be y offset
    
    # Weight: only first position has weight
    weight = torch.zeros(C_out, C_in, K_h, K_w, dtype=torch.float32)
    weight[0, 0, 0, 0] = 1.0
    
    # No mask modulation
    mask = torch.ones(N, K_h * K_w, 3, 3, dtype=torch.float32)
    
    output = deform_conv2d(input_tensor, offset, weight, None, 
                          stride=1, padding=0, dilation=1, mask=mask)
    
    print("\nOffset ordering test:")
    print("Input at (0,0):", input_tensor[0, 0, 0, 0].item())
    print("Input at (0,1):", input_tensor[0, 0, 0, 1].item())
    print("Input at (1,0):", input_tensor[0, 0, 1, 0].item())
    print("With x-offset=1, y-offset=0:")
    print("Output at (0,0):", output[0, 0, 0, 0].item())
    print("This should sample from position (0, 1) if offset[0] is x-offset")
    
    # Reset and try y offset
    offset.zero_()
    offset[0, 0, 0, 0] = 0.0  # x offset
    offset[0, 1, 0, 0] = 1.0  # y offset
    
    output2 = deform_conv2d(input_tensor, offset, weight, None, 
                           stride=1, padding=0, dilation=1, mask=mask)
    
    print("\nWith x-offset=0, y-offset=1:")
    print("Output at (0,0):", output2[0, 0, 0, 0].item())
    print("This should sample from position (1, 0) if offset[1] is y-offset")

def bilinear_interpolate_torch(img, y, x):
    """PyTorch-style bilinear interpolation for comparison"""
    H, W = img.shape
    
    # Get integer coordinates
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    # Clip coordinates
    x0 = torch.clamp(x0, 0, W-1)
    x1 = torch.clamp(x1, 0, W-1)
    y0 = torch.clamp(y0, 0, H-1)
    y1 = torch.clamp(y1, 0, H-1)
    
    # Get pixel values
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]
    
    # Calculate weights
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())
    
    return Ia * wa + Ib * wb + Ic * wc + Id * wd

# Run tests
print("=" * 50)
print("Testing PyTorch Deformable Convolution Behavior")
print("=" * 50)

test_simple_case()
check_offset_ordering()

# Now let's check if the issue is with bilinear interpolation
print("\n" + "=" * 50)
print("Testing Bilinear Interpolation")
print("=" * 50)

# Create test image
img = torch.arange(9, dtype=torch.float32).reshape(3, 3)
print("Test image:")
print(img)

# Test at fractional coordinates
x, y = torch.tensor(0.5), torch.tensor(0.5)
result = bilinear_interpolate_torch(img, y, x)
print(f"\nBilinear interpolation at ({x}, {y}): {result}")
print("Expected (average of 0,1,3,4):", (0+1+3+4)/4)
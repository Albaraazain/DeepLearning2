import numpy as np
import torch
from torchvision.ops import deform_conv2d

def bilinear_interpolate(a_l, q_y, q_x):
    """
    Perform bilinear interpolation on the input activation map at the given (fractional) coordinates.
    PyTorch uses zero-padding for out-of-bounds values.
    """
    H, W = a_l.shape
    
    # PyTorch uses zero padding for out-of-bounds
    if q_y < 0 or q_y >= H or q_x < 0 or q_x >= W:
        # Check if we're completely out of bounds
        if q_y < -1 or q_y >= H or q_x < -1 or q_x >= W:
            return 0.0
    
    # Get the four nearest integer positions
    y0 = int(np.floor(q_y))
    x0 = int(np.floor(q_x))
    y1 = y0 + 1
    x1 = x0 + 1
    
    # Get values at corners (0 if out of bounds)
    def get_pixel_value(y, x):
        if 0 <= y < H and 0 <= x < W:
            return a_l[y, x]
        return 0.0
    
    v_00 = get_pixel_value(y0, x0)
    v_01 = get_pixel_value(y0, x1)
    v_10 = get_pixel_value(y1, x0)
    v_11 = get_pixel_value(y1, x1)
    
    # Calculate weights
    fy = q_y - y0
    fx = q_x - x0
    
    # Bilinear interpolation
    value = (v_00 * (1 - fx) * (1 - fy) +
             v_01 * fx * (1 - fy) +
             v_10 * (1 - fx) * fy +
             v_11 * fx * fy)
    
    return value

# Test the interpolation
test_img = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.float32)

print("Testing bilinear interpolation:")
print(f"At (0, 0): {bilinear_interpolate(test_img, 0, 0)} (expected: 1)")
print(f"At (1, 1): {bilinear_interpolate(test_img, 1, 1)} (expected: 5)")
print(f"At (2.5, 2.5): {bilinear_interpolate(test_img, 2.5, 2.5)} (expected: partial)")
print(f"At (3, 3): {bilinear_interpolate(test_img, 3, 3)} (expected: 0)")
print(f"At (-0.5, -0.5): {bilinear_interpolate(test_img, -0.5, -0.5)} (expected: partial)")

def deform_conv2d_np(a_l, delta, mask, weight, stride=1, padding=0, dilation=1):
    """
    Deformable Conv2D v2 operation (forward pass) implemented in NumPy.
    """
    N, C_in, H_in, W_in = a_l.shape
    C_out, _, K_h, K_w = weight.shape
    
    if padding > 0:
        a_l = np.pad(a_l, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        H_in += 2 * padding
        W_in += 2 * padding
    
    H_out = (H_in - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W_in - dilation * (K_w - 1) - 1) // stride + 1
    
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    
    for n in range(N):
        for c_out in range(C_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    w_start = w_out * stride
                    
                    value = 0.0
                    
                    for kh in range(K_h):
                        for kw in range(K_w):
                            k = kh * K_w + kw
                            
                            delta_y = delta[n, 2 * k, h_out, w_out]
                            delta_x = delta[n, 2 * k + 1, h_out, w_out]
                            m_k = mask[n, k, h_out, w_out]
                            
                            sample_y = h_start + kh * dilation + delta_y
                            sample_x = w_start + kw * dilation + delta_x
                            
                            for c_in in range(C_in):
                                interpolated = bilinear_interpolate(
                                    a_l[n, c_in, :, :], sample_y, sample_x
                                )
                                
                                value += weight[c_out, c_in, kh, kw] * m_k * interpolated
                    
                    out[n, c_out, h_out, w_out] = value
    
    return out

# Run the full test
np.random.seed(0)
torch.manual_seed(0)

N, C_in, H_in, W_in = 1, 3, 5, 5
C_out, K_h, K_w = 2, 3, 3
stride, padding, dilation = 1, 0, 1

np_input = np.arange(N * C_in * H_in * W_in, dtype=np.float32).reshape(N, C_in, H_in, W_in)
torch_input = torch.tensor(np_input, dtype=torch.float32)

np_offset = np.random.rand(N, 2 * K_h * K_w, 3, 3).astype(np.float32)
torch_offset = torch.tensor(np.copy(np_offset), dtype=torch.float32)

np_mask = np.ones((N, K_h * K_w, 3, 3), dtype=np.float32)
torch_mask = torch.tensor(np_mask, dtype=torch.float32)

np_weight = np.random.rand(C_out, C_in, K_h, K_w).astype(np.float32)
torch_weight = torch.tensor(np.copy(np_weight), dtype=torch.float32)

np_bias = np.zeros((C_out), dtype=np.float32)
torch_bias = torch.tensor(np.copy(np_bias), dtype=torch.float32)

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

print("\nNumPy Output:\n", np_output[0, 0])
print("PyTorch Output:\n", torch_output[0, 0].detach().numpy())

grade = 0
if (np.allclose(np_output, torch_output.detach().numpy(), atol=1e-4) == True):
    grade = 30
else:
    grade = 0
print(" Your grade is ", grade, "/30.")

# Debug specific positions
print("\nDebugging position (0,2):")
print(f"Offset at (0,2) for k=0: dy={np_offset[0, 0, 0, 2]}, dx={np_offset[0, 1, 0, 2]}")
print(f"NumPy: {np_output[0, 0, 0, 2]}")
print(f"PyTorch: {torch_output[0, 0, 0, 2].item()}")
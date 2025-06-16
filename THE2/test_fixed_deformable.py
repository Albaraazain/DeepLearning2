import numpy as np
import torch
from torchvision.ops import deform_conv2d

# Fixed bilinear interpolation
def bilinear_interpolate(a_l, q_y, q_x):
    """
    Perform bilinear interpolation on the input activation map at the given (fractional) coordinates.
    """
    H, W = a_l.shape
    
    # Ensure coordinates are within bounds first
    q_y = np.clip(q_y, 0, H - 1)
    q_x = np.clip(q_x, 0, W - 1)
    
    # Get the four nearest integer positions
    p_lt_y = int(np.floor(q_y))  # left top y
    p_lt_x = int(np.floor(q_x))  # left top x
    p_rb_y = int(np.ceil(q_y))   # right bottom y
    p_rb_x = int(np.ceil(q_x))   # right bottom x
    
    # Handle edge case where q_y or q_x is exactly an integer
    if p_lt_y == p_rb_y:
        p_rb_y = min(p_lt_y + 1, H - 1)
    if p_lt_x == p_rb_x:
        p_rb_x = min(p_lt_x + 1, W - 1)
    
    # Get the four corner values
    v_lt = a_l[p_lt_y, p_lt_x]  # left top
    v_rt = a_l[p_lt_y, p_rb_x]  # right top
    v_lb = a_l[p_rb_y, p_lt_x]  # left bottom
    v_rb = a_l[p_rb_y, p_rb_x]  # right bottom
    
    # Calculate fractional parts
    frac_y = q_y - p_lt_y
    frac_x = q_x - p_lt_x
    
    # Calculate weights
    w_lt = (1 - frac_x) * (1 - frac_y)
    w_rt = frac_x * (1 - frac_y)
    w_lb = (1 - frac_x) * frac_y
    w_rb = frac_x * frac_y
    
    # Compute interpolated value
    out = v_lt * w_lt + v_rt * w_rt + v_lb * w_lb + v_rb * w_rb
    
    return out

# Test the fixed interpolation
test_img = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.float32)

print("Testing fixed bilinear interpolation:")
print(f"At (0, 0): {bilinear_interpolate(test_img, 0, 0)} (expected: {test_img[0, 0]})")
print(f"At (1, 1): {bilinear_interpolate(test_img, 1, 1)} (expected: {test_img[1, 1]})")
print(f"At (0.5, 0.5): {bilinear_interpolate(test_img, 0.5, 0.5)} (expected: {(1+2+4+5)/4})")
print(f"At (1.5, 1.5): {bilinear_interpolate(test_img, 1.5, 1.5)} (expected: {(5+6+8+9)/4})")

# Now let's fix the deformable convolution
def deform_conv2d_np(a_l, delta, mask, weight, stride=1, padding=0, dilation=1):
    """
    Deformable Conv2D v2 operation (forward pass) implemented in NumPy.
    """
    # Step 1: Preparing hyperparameters, pad input, and initialize output
    N, C_in, H_in, W_in = a_l.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Pad the input if necessary
    if padding > 0:
        a_l = np.pad(a_l, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        H_in += 2 * padding
        W_in += 2 * padding
    
    # Calculate output dimensions
    H_out = (H_in - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W_in - dilation * (K_w - 1) - 1) // stride + 1
    
    # Initialize output
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    
    # Step 2-6: Iterate over all coordinates and perform deformable convolution
    for n in range(N):  # batch dimension
        for c_out in range(C_out):  # output channels
            for h_out in range(H_out):  # output height
                for w_out in range(W_out):  # output width
                    # Starting position in input (p_0 in the formulation)
                    h_start = h_out * stride
                    w_start = w_out * stride
                    
                    # Accumulator for this output position
                    value = 0.0
                    
                    # Iterate over kernel positions
                    for kh in range(K_h):
                        for kw in range(K_w):
                            # Kernel position index
                            k = kh * K_w + kw
                            
                            # Step 3: Get delta (offset) and mask
                            # Delta contains (dy, dx) for each kernel location
                            delta_y = delta[n, 2 * k, h_out, w_out]
                            delta_x = delta[n, 2 * k + 1, h_out, w_out]
                            m_k = mask[n, k, h_out, w_out]
                            
                            # Step 4: Compute the deformed sampling position
                            # p_0 + p_k + Delta_p_k
                            sample_y = h_start + kh * dilation + delta_y
                            sample_x = w_start + kw * dilation + delta_x
                            
                            # Iterate over input channels
                            for c_in in range(C_in):
                                # Step 5: Bilinear interpolation
                                interpolated = bilinear_interpolate(
                                    a_l[n, c_in, :, :], sample_y, sample_x
                                )
                                
                                # Step 6: Apply convolution weight and modulation mask
                                value += weight[c_out, c_in, kh, kw] * m_k * interpolated
                    
                    out[n, c_out, h_out, w_out] = value
    
    return out

# Test with the original data
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

# Run both implementations
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
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

def bilinear_interpolate(a_l, q_y, q_x):
    """
    Perform bilinear interpolation on the input activation map at the given (fractional) coordinates.
    """
    H, W = a_l.shape
    
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
    v_rt = a_l[p_lt_y, p_rb_x]  # right top
    v_lb = a_l[p_rb_y, p_lt_x]  # left bottom
    v_rb = a_l[p_rb_y, p_rb_x]  # right bottom
    
    # Calculate weights using the formula: G(p, q) = (1 - |px - qx|) * (1 - |py - qy|)
    w_lt = (1 - abs(p_lt_x - q_x)) * (1 - abs(p_lt_y - q_y))
    w_rt = (1 - abs(p_rb_x - q_x)) * (1 - abs(p_lt_y - q_y))
    w_lb = (1 - abs(p_lt_x - q_x)) * (1 - abs(p_rb_y - q_y))
    w_rb = (1 - abs(p_rb_x - q_x)) * (1 - abs(p_rb_y - q_y))
    
    # Compute interpolated value
    out = v_lt * w_lt + v_rt * w_rt + v_lb * w_lb + v_rb * w_rb
    
    return out

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
                            # Delta contains (dx, dy) for each kernel location
                            # FIXED: PyTorch uses (x, y) order, not (y, x)
                            delta_x = delta[n, 2 * k, h_out, w_out]      # x offset first
                            delta_y = delta[n, 2 * k + 1, h_out, w_out]  # y offset second
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

# Test with the same setup as the notebook
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
print("NumPy Output:\n", np_output[0, 0])
print("PyTorch Output:\n", torch_output[0, 0].detach().numpy())
print("Max difference:", np.abs(np_output - torch_output.detach().numpy()).max())

# Let's debug step by step
print("\n=== Debugging ===")
print("First offset pair (x, y):", np_offset[0, 0:2, 0, 0])
print("First mask value:", np_mask[0, 0, 0, 0])

# Check a specific output position
n, c_out, h_out, w_out = 0, 0, 0, 0
print(f"\nFor output position ({h_out}, {w_out}):")

# Manual calculation for first kernel position
kh, kw = 0, 0
k = kh * K_w + kw
h_start = h_out * stride
w_start = w_out * stride

delta_x = np_offset[n, 2 * k, h_out, w_out]
delta_y = np_offset[n, 2 * k + 1, h_out, w_out]
m_k = np_mask[n, k, h_out, w_out]

sample_y = h_start + kh * dilation + delta_y
sample_x = w_start + kw * dilation + delta_x

print(f"Kernel position ({kh}, {kw}): offset=({delta_x:.4f}, {delta_y:.4f}), sample pos=({sample_x:.4f}, {sample_y:.4f})")
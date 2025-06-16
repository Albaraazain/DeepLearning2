"""
Deformable Convolution v2 Implementation in NumPy

This module implements deformable convolution v2 from scratch using only NumPy,
following the paper: https://arxiv.org/abs/1811.11168
"""

import numpy as np


def bilinear_interpolate(a_l, q_y, q_x):
    """
    Perform bilinear interpolation on the input activation map at the given (fractional) coordinates.

    Args:
        a_l (np.ndarray): 2D array of shape (H, W) representing the activation map (feature map) at a certain layer.
        q_y (float): Y-coordinate (row index) where interpolation is to be performed.
        q_x (float): X-coordinate (column index) where interpolation is to be performed.

    Returns:
        out (np.ndarray): Interpolated value at (q_x, q_y).
    """
    H, W = a_l.shape
    
    # Get the four nearest integer positions
    y0 = int(np.floor(q_y))
    x0 = int(np.floor(q_x))
    y1 = y0 + 1
    x1 = x0 + 1
    
    # Get values at corners, using 0 for out-of-bounds
    def get_pixel_value(img, y, x):
        if 0 <= y < H and 0 <= x < W:
            return img[y, x]
        else:
            return 0.0  # Zero padding for out-of-bounds
    
    v_00 = get_pixel_value(a_l, y0, x0)  # top-left
    v_01 = get_pixel_value(a_l, y0, x1)  # top-right
    v_10 = get_pixel_value(a_l, y1, x0)  # bottom-left
    v_11 = get_pixel_value(a_l, y1, x1)  # bottom-right
    
    # Calculate fractional parts
    dy = q_y - y0
    dx = q_x - x0
    
    # Perform bilinear interpolation
    # First interpolate along x-direction
    v_0 = v_00 * (1 - dx) + v_01 * dx  # top edge
    v_1 = v_10 * (1 - dx) + v_11 * dx  # bottom edge
    
    # Then interpolate along y-direction
    out = v_0 * (1 - dy) + v_1 * dy
    
    return out


def deform_conv2d_np(a_l, delta, mask, weight, stride=1, padding=0, dilation=1):
    """
    Deformable Conv2D v2 operation (forward pass) implemented in NumPy.

    Args:
        a_l (np.ndarray): Input feature map of shape (N, C_in, H_in, W_in),
                            where N is the batch size, C_in is the number of
                            input channels, and (H_in, W_in) are the height and
                            width of the input feature map. input corresponds to
                            'a^l' in the above formulation.

        delta (np.ndarray): Learned/estimated offsets of shape (N, 2 * K_h * K_w, H_out, W_out),
                             where K_h and K_w are the kernel height and width,
                             and (H_out, W_out) are the spatial dimensions of
                             the output feature map. The offset tensor corresponds
                             to 'Delta-p' in the above formulation and provides
                             the x and y displacements for each sampled point.

        mask (np.ndarray): Learned modulation masks of shape (N, K_h*K_w, H_out, W_out).
                           Corresponds to 'm' in the above formulation.

        weight (np.ndarray): Convolution kernel of shape (C_out, C_in, K_h, K_w),
                             where C_out is the number of output channels. Corresponds
                             to 'w' in the above formulation.

        stride (int): Stride of the convolution. Determines the spacing between
                               sampled input locations. Default is 1.

        padding (int): Zero-padding added to both sides of the input along height and width.
                                Default is 0.

        dilation (int): Dilation factor for the convolution kernel.
                                 Controls the spacing between kernel elements. Default is 1.

    Returns:
        out (np.ndarray): Output feature map of shape (N, C_out, H_out, W_out),
                          where each position is computed via deformable convolution using
                          bilinearly interpolated input values and learned offsets.
                          Corresponds to 'a^l+1' in the above formulation.
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
                            # IMPORTANT: PyTorch stores offsets with y (height) offset first, then x (width)
                            # offset[2*k] is the y-offset (height direction)
                            # offset[2*k+1] is the x-offset (width direction)
                            delta_y = delta[n, 2 * k, h_out, w_out]      # y offset (height)
                            delta_x = delta[n, 2 * k + 1, h_out, w_out]  # x offset (width)
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
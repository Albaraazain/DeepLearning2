import numpy as np
import sys
sys.path.append('task1_deformable_conv')
from deformable_conv import bilinear_interpolate

# Test bilinear interpolation
img = np.array([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]])

print("Test image:")
print(img)

# Test at integer coordinates (should return exact values)
test_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("\nTesting at integer coordinates:")
for y, x in test_coords:
    result = bilinear_interpolate(img, y, x)
    print(f"  Position ({y},{x}): result = {result}, expected = {img[y,x]}")

# Test at fractional coordinates
print("\nTesting at fractional coordinates:")
result = bilinear_interpolate(img, 0.5, 0.5)
print(f"  Position (0.5,0.5): result = {result}, expected = {(0+1+3+4)/4} = {(0+1+3+4)/4}")

# Debug the function step by step
print("\nDebugging bilinear_interpolate at (0, 1):")
q_y, q_x = 0.0, 1.0
H, W = img.shape

p_lt_y = int(np.floor(q_y))  # 0
p_lt_x = int(np.floor(q_x))  # 1
p_rb_y = int(np.ceil(q_y))   # 0
p_rb_x = int(np.ceil(q_x))   # 1

print(f"  Floor/ceil positions: lt=({p_lt_y},{p_lt_x}), rb=({p_rb_y},{p_rb_x})")

# After clipping
p_lt_y = np.clip(p_lt_y, 0, H - 1)
p_lt_x = np.clip(p_lt_x, 0, W - 1)
p_rb_y = np.clip(p_rb_y, 0, H - 1)
p_rb_x = np.clip(p_rb_x, 0, W - 1)

print(f"  After clipping: lt=({p_lt_y},{p_lt_x}), rb=({p_rb_y},{p_rb_x})")

# Get values
v_lt = img[p_lt_y, p_lt_x]  # img[0,1] = 1
v_rt = img[p_lt_y, p_rb_x]  # img[0,1] = 1
v_lb = img[p_rb_y, p_lt_x]  # img[0,1] = 1
v_rb = img[p_rb_y, p_rb_x]  # img[0,1] = 1

print(f"  Corner values: lt={v_lt}, rt={v_rt}, lb={v_lb}, rb={v_rb}")

# Calculate weights
w_lt = (1 - abs(p_lt_x - q_x)) * (1 - abs(p_lt_y - q_y))  # (1-|1-1|)*(1-|0-0|) = 0*1 = 0
w_rt = (1 - abs(p_rb_x - q_x)) * (1 - abs(p_lt_y - q_y))  # (1-|1-1|)*(1-|0-0|) = 0*1 = 0
w_lb = (1 - abs(p_lt_x - q_x)) * (1 - abs(p_rb_y - q_y))  # (1-|1-1|)*(1-|0-0|) = 0*1 = 0
w_rb = (1 - abs(p_rb_x - q_x)) * (1 - abs(p_rb_y - q_y))  # (1-|1-1|)*(1-|0-0|) = 0*1 = 0

print(f"  Weights: lt={w_lt}, rt={w_rt}, lb={w_lb}, rb={w_rb}")
print(f"  Sum of weights: {w_lt + w_rt + w_lb + w_rb}")

# The issue: when q_x = 1.0 (integer), all weights become 0!
# This is because floor(1.0) = ceil(1.0) = 1, so all corners are the same point
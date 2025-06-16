import torch
import gc

print("Attempting to fix CUDA issues...")

# Force garbage collection
gc.collect()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("✓ CUDA cache cleared")

print("\nTo fully resolve this issue, you should:")
print("1. Restart your Jupyter kernel (Kernel -> Restart)")
print("2. Or kill the existing Python process using:")
print(f"   kill -9 699839")
print("\nAfter restarting, the CUDA error should be resolved.")
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    try:
        # Test basic CUDA operations
        print("\nTesting basic CUDA operations...")
        
        # First clear any existing CUDA context
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try setting manual seed
        print("Setting manual seed...")
        torch.manual_seed(0)
        print("✓ Manual seed set successfully")
        
        # Test basic tensor operations
        print("\nTesting tensor creation...")
        x = torch.randn(2, 2, device='cuda')
        print("✓ CUDA tensor created")
        
        print("\nTesting tensor operations...")
        y = x * 2
        torch.cuda.synchronize()
        print("✓ CUDA operations work")
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        print("\nTrying to reset CUDA...")
        torch.cuda.empty_cache()
        
        # Check if we can at least use CPU
        print("\nTesting CPU operations...")
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 2)
        print("✓ CPU operations work fine")
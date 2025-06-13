import os
import subprocess
import torch

def download_dataset():
    """Download and extract MNIST dataset"""
    if not os.path.exists('mnist_png'):
        print("Downloading dataset...")
        url = "https://github.com/Bedrettin-Cetinkaya/imbalancedMNIST/raw/refs/heads/main/mnist_png.zip"
        subprocess.run(["wget", "-O", "mnist_png.zip", url])
        subprocess.run(["unzip", "mnist_png.zip"])
        print("Dataset downloaded and extracted")
    else:
        print("Dataset already exists")

def run_task1():
    """Run Task 1: MLP from scratch"""
    print("\n=== Task 1: MLP from Scratch ===")
    
    print("\nTesting activation functions...")
    os.system('python task1_activation.py')
    
    print("\nTraining MLP on XOR dataset...")
    os.system('python task1_training.py')

def run_task2():
    """Run Task 2: PyTorch MLP"""
    print("\n=== Task 2: PyTorch MLP ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download dataset if needed
    download_dataset()
    
    print("\nTesting dataloader...")
    os.system('python task2_dataloader.py')
    
    print("\nTesting model architecture...")
    os.system('python task2_model.py')
    
    print("\nStarting training and hyperparameter tuning...")
    os.system('python task2_train.py')

if __name__ == "__main__":
    print("CENG403 - Deep Learning Take Home Exam 1")
    print("========================================")
    
    while True:
        print("\nChoose a task to run:")
        print("1. Task 1 - MLP from Scratch")
        print("2. Task 2 - PyTorch MLP")
        print("3. Run both tasks")
        print("q. Quit")
        
        choice = input("\nEnter your choice (1/2/3/q): ")
        
        if choice == '1':
            run_task1()
        elif choice == '2':
            run_task2()
        elif choice == '3':
            run_task1()
            run_task2()
        elif choice.lower() == 'q':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")
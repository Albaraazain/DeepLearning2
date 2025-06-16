"""
Data loader for CIFAR-100 dataset
"""

from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms


def get_cifar100_loaders(batch_size=128, data_root='./data'):
    """
    Create DataLoaders for CIFAR-100 dataset with 80/20 train/validation split
    
    Args:
        batch_size (int): Batch size for training and validation
        data_root (str): Root directory to download/load the dataset
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),      # Random horizontal flip for augmentation
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
                            std=[0.2675, 0.2565, 0.2761])    # CIFAR-100 std
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
                            std=[0.2675, 0.2565, 0.2761])    # CIFAR-100 std
    ])

    # Define full training set
    full_train_set = CIFAR100(root=data_root, train=True, download=True, transform=transform_train)

    # Calculate split lengths, Perform split
    train_size = int(0.8 * len(full_train_set))  # 80% for training
    val_size = len(full_train_set) - train_size  # 20% for validation

    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    # Update val_set transform to use validation transform
    val_set.dataset.transform = transform_test

    # Define Data loaders for training and validation splits
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Define Test set and loader
    test_set = CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Print Split Ratios
    print(f"Total samples in CIFAR-100 training set: {len(full_train_set)}")
    print(f"Training split: {len(train_set)} samples ({len(train_set)/len(full_train_set)*100:.2f}%)")
    print(f"Validation split: {len(val_set)} samples ({len(val_set)/len(full_train_set)*100:.2f}%)")
    print(f"Test split: {len(test_set)} samples")
    
    return train_loader, val_loader, test_loader
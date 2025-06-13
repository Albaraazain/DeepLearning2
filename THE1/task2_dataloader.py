import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MNISTCustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Get all class folders
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                # Get all images in class folder
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(class_path, img_file)
                        self.images.append(img_path)
                        self.labels.append(int(class_folder))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(batch_size=32):
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create datasets
    train_dataset = MNISTCustomDataset(
        data_dir='mnist_png/training',
        transform=transform_train
    )
    
    val_dataset = MNISTCustomDataset(
        data_dir='mnist_png/validation',
        transform=transform_test
    )
    
    test_dataset = MNISTCustomDataset(
        data_dir='mnist_png/testing',
        transform=transform_test
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test dataloader
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Sample labels:", labels[:10])
"""
Training script for CNN models on CIFAR-100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from .models import CustomCNN, CustomCNNwithBN
from .data_loader import get_cifar100_loaders


def train(model, train_loader, optimizer, loss_function, device):
    """Train the model for one epoch"""
    model.train()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        # Get images and their labels from the batch
        images, labels = batch
        
        # Pass tensors to device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        # Accumulate loss
        total_loss += loss.item() * images.size(0)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Calculate top-1 and top-5 accuracies
        _, top1_pred = outputs.topk(1, dim=1, largest=True, sorted=True)
        top1_correct = top1_pred.eq(labels.view(-1, 1)).sum().item()
        
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        total_top1 += top1_correct
        total_top5 += top5_correct
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_top1 = 100.0 * total_top1 / total_samples
    avg_top5 = 100.0 * total_top5 / total_samples

    return avg_loss, avg_top1, avg_top5


def validate(model, val_loader, loss_function, device):
    """Validate the model"""
    model.eval()

    running_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Extract images and labels from batch
            images, labels = batch
            
            # Pass tensors to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            # Accumulate loss
            running_loss += loss.item() * images.size(0)
            
            # Calculate top-1 accuracy
            _, top1_pred = outputs.topk(1, dim=1, largest=True, sorted=True)
            top1_correct = top1_pred.eq(labels.view(-1, 1)).sum().item()
            
            # Calculate top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            total_top1 += top1_correct
            total_top5 += top5_correct
            total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_top1 = 100.0 * total_top1 / total_samples
    avg_top5 = 100.0 * total_top5 / total_samples

    return avg_loss, avg_top1, avg_top5


def train_model(model_class=CustomCNN, num_epochs=30, learning_rate=0.01, 
                optimizer_type='SGD', batch_size=128):
    """
    Train a model with specified hyperparameters
    
    Args:
        model_class: Model class to instantiate (CustomCNN or CustomCNNwithBN)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        optimizer_type: 'SGD' or 'Adam'
        batch_size: Batch size for training
        
    Returns:
        dict: Training history with losses and accuracies
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_cifar100_loaders(batch_size)
    
    # Initialize model
    model = model_class().to(device)
    print(f"Model: {model_class.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    loss_function = nn.CrossEntropyLoss()
    
    # Optimizer
    weight_decay = 5e-4
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                             momentum=0.9, weight_decay=weight_decay)
    else:  # Adam
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                              weight_decay=weight_decay)
    
    print(f"Optimizer: {optimizer_type}, LR: {learning_rate}")
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_top1': [],
        'train_top5': [],
        'val_top1': [],
        'val_top5': []
    }
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_top1, train_top5 = train(model, train_loader, optimizer, loss_function, device)
        
        # Validate
        val_loss, val_top1, val_top5 = validate(model, val_loader, loss_function, device)
        
        # Store metrics
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_top1'].append(train_top1)
        history['train_top5'].append(train_top5)
        history['val_top1'].append(val_top1)
        history['val_top5'].append(val_top5)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%")
    
    print("Training completed!")
    
    # Test the model
    test_loss, test_top1, test_top5 = validate(model, test_loader, loss_function, device)
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Top-1 Accuracy: {test_top1:.2f}%")
    print(f"  Test Top-5 Accuracy: {test_top5:.2f}%")
    
    history['test_top1'] = test_top1
    history['test_top5'] = test_top5
    history['model'] = model
    
    return history


if __name__ == "__main__":
    # Train the basic model
    history = train_model(model_class=CustomCNN, num_epochs=30, 
                         learning_rate=0.01, optimizer_type='SGD')
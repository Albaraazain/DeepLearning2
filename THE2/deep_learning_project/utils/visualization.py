"""
Visualization utilities for plotting training metrics
"""

import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses):
    """
    Plot training and validation losses
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_accuracies(train_top1, train_top5, val_top1, val_top5):
    """
    Plot training and validation accuracies (top-1 and top-5)
    
    Args:
        train_top1 (list): Training top-1 accuracies per epoch
        train_top5 (list): Training top-5 accuracies per epoch
        val_top1 (list): Validation top-1 accuracies per epoch
        val_top5 (list): Validation top-5 accuracies per epoch
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_top1) + 1)
    
    plt.plot(epochs, train_top1, 'b-', label='Train Top-1')
    plt.plot(epochs, train_top5, 'b--', label='Train Top-5')
    plt.plot(epochs, val_top1, 'r-', label='Val Top-1')
    plt.plot(epochs, val_top5, 'r--', label='Val Top-5')
    
    plt.title('Training and Validation Accuracies', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def report_accuracy(test_loader, model, device):
    """
    Calculate and report test accuracies
    
    Args:
        test_loader: DataLoader for test set
        model: Trained model
        device: Device to run on (CPU/GPU)
        
    Returns:
        tuple: (top1_accuracy, top5_accuracy)
    """
    import torch
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Get images and their labels from batch
            images, labels = batch
            
            # Pass tensors to device (CPU/GPU)
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate top-1 accuracy
            _, top1_pred = outputs.topk(1, dim=1, largest=True, sorted=True)
            correct_top1 += top1_pred.eq(labels.view(-1, 1)).sum().item()
            
            # Calculate top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            total += labels.size(0)

    accuracy_top1 = (correct_top1 / total) * 100
    accuracy_top5 = (correct_top5 / total) * 100
    
    return accuracy_top1, accuracy_top5
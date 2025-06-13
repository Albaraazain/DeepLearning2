import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from task2_dataloader import get_data_loaders
from task2_model import get_model

def train(model, train_loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move to device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_function, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    
    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy

def grid_search(device, batch_size=32):
    # Get dataloaders
    train_loader, val_loader, _ = get_data_loaders(batch_size)
    
    # Grid search parameters
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    dropout_rates = [0.0, 0.1, 0.2, 0.5, 0.8]
    
    best_acc = 0
    best_params = None
    results = []
    
    print("Starting grid search...")
    
    # Try all combinations
    for lr in learning_rates:
        for dropout in dropout_rates:
            print(f"\nTrying lr={lr}, dropout={dropout}")
            
            # Create model
            model = get_model(device)
            model.dropout.p = dropout
            
            # Setup training
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Train for few epochs
            best_val_acc = 0
            for epoch in range(5):
                train_loss = train(model, train_loader, optimizer, loss_function, device)
                val_loss, val_acc = validate(model, val_loader, loss_function, device)
                best_val_acc = max(best_val_acc, val_acc)
            
            results.append({
                'lr': lr,
                'dropout': dropout,
                'accuracy': best_val_acc
            })
            
            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_params = {'lr': lr, 'dropout': dropout}
    
    return best_params, results

def train_final_model(best_params, device, batch_size=32, epochs=20):
    # Get data
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)
    
    # Create model with best params
    model = get_model(device)
    model.dropout.p = best_params['dropout']
    
    # Setup training
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    # Training tracking
    train_losses = []
    val_losses = []
    val_accs = []
    
    print("\nTraining final model...")
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, loss_function, device)
        val_loss, val_acc = validate(model, val_loader, loss_function, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Final evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, loss_function, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return model, test_acc

if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find best parameters
    best_params, results = grid_search(device)
    print("\nBest parameters found:")
    print(f"Learning rate: {best_params['lr']}")
    print(f"Dropout rate: {best_params['dropout']}")
    
    # Train final model
    model, test_acc = train_final_model(best_params, device)
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
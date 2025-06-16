"""
Hyperparameter search for CNN models
"""

import torch
from .models import CustomCNN, CustomCNNwithBN
from .train import train_model
from ..utils.visualization import plot_loss, plot_accuracies


def hyperparameter_search():
    """
    Perform grid search over hyperparameters
    """
    
    # Hyperparameter grid
    learning_rates = [0.0001, 0.001]
    optimizer_classes = ['Adam', 'SGD']
    model_classes = [CustomCNN, CustomCNNwithBN]
    
    # Store results
    results = []
    best_val_top1 = 0
    best_params = None
    
    # Fewer epochs for grid search
    num_epochs_search = 15
    
    for model_class in model_classes:
        for optimizer_type in optimizer_classes:
            for lr in learning_rates:
                # Get model name
                model_name = model_class.__name__
                
                print(f"\n{'='*60}")
                print(f"Training {model_name} with {optimizer_type}, LR={lr}")
                print(f"{'='*60}")
                
                # Train model
                history = train_model(
                    model_class=model_class,
                    num_epochs=num_epochs_search,
                    learning_rate=lr,
                    optimizer_type=optimizer_type
                )
                
                # Get best validation accuracy for this combination
                best_epoch_val_top1 = max(history['val_top1'])
                
                # Store results
                result = {
                    'model': model_name,
                    'optimizer': optimizer_type,
                    'lr': lr,
                    'best_val_top1': best_epoch_val_top1,
                    'history': history
                }
                results.append(result)
                
                # Check if this is the best so far
                if best_epoch_val_top1 > best_val_top1:
                    best_val_top1 = best_epoch_val_top1
                    best_params = result
    
    # Print summary of results
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*60}")
    for result in results:
        print(f"{result['model']:15} | {result['optimizer']:6} | LR: {result['lr']:6} | "
              f"Best Val Top-1: {result['best_val_top1']:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"BEST PARAMETERS:")
    print(f"Model: {best_params['model']}")
    print(f"Optimizer: {best_params['optimizer']}")
    print(f"Learning Rate: {best_params['lr']}")
    print(f"Best Validation Top-1 Accuracy: {best_params['best_val_top1']:.2f}%")
    print(f"{'='*60}")
    
    # Plot results for best model
    print(f"\nPlotting results for best model...")
    history = best_params['history']
    
    plot_loss(history['train_losses'], history['val_losses'])
    plot_accuracies(
        history['train_top1'], 
        history['train_top5'], 
        history['val_top1'], 
        history['val_top5']
    )
    
    # Train best model for more epochs
    print(f"\nTraining best model configuration for 30 epochs...")
    final_history = train_model(
        model_class=CustomCNN if best_params['model'] == 'CustomCNN' else CustomCNNwithBN,
        num_epochs=30,
        learning_rate=best_params['lr'],
        optimizer_type=best_params['optimizer']
    )
    
    print(f"\nFinal Test Results:")
    print(f"Test Top-1 Accuracy: {final_history['test_top1']:.2f}%")
    print(f"Test Top-5 Accuracy: {final_history['test_top5']:.2f}%")
    
    return results, best_params, final_history


if __name__ == "__main__":
    results, best_params, final_history = hyperparameter_search()
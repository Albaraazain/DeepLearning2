#!/bin/bash

# Script to run examples for each task

echo "Deep Learning Project - Example Usage"
echo "====================================="

# Task 1: Test Deformable Convolution
echo -e "\n1. Testing Deformable Convolution Implementation"
echo "------------------------------------------------"
python main.py --task 1

# Task 2: Train CNN (short example)
echo -e "\n2. Training CNN on CIFAR-100 (example with 5 epochs)"
echo "----------------------------------------------------"
# Note: Full training uses 30 epochs, but we use 5 here for demo
python -c "
from task2_cnn.train import train_model
from task2_cnn.models import CustomCNN
history = train_model(model_class=CustomCNN, num_epochs=5, learning_rate=0.01, optimizer_type='SGD')
print(f'\\nExample training completed!')
print(f'Val Top-1 Accuracy: {history[\"val_top1\"][-1]:.2f}%')
"

# Task 3: Test RNN
echo -e "\n3. Testing RNN Implementation"
echo "-----------------------------"
python main.py --task 3

echo -e "\nAll examples completed!"
echo "For full training, use:"
echo "  python main.py --task 2 --train                    # Train CNN for 30 epochs"
echo "  python main.py --task 2 --hyperparameter-search    # Run hyperparameter search"
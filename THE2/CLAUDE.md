# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Learning course take-home exam (THE2) for CENG403 at METU (Middle East Technical University), Spring 2025. The project contains three main tasks implemented in a Jupyter notebook:

1. **Task 1**: Implement deformable convolution v2 forward pass from scratch using NumPy
2. **Task 2**: Implement CNN training pipeline using PyTorch with CIFAR-100 dataset
3. **Task 3**: Implement RNN using torch.autograd for character-level prediction

## Development Commands

Since this is a Jupyter notebook-based project, the main development workflow involves:

- **Run notebook**: Open `CENG403_2025_Spring_THE2.ipynb` in Jupyter Lab/Notebook
- **Execute cells**: Use standard Jupyter notebook execution (Shift+Enter)
- **Validate implementations**: Each task has validation cells that check correctness and provide grades

## Code Architecture

### Task 1: Deformable Convolution (NumPy Implementation)
- `bilinear_interpolate()`: Performs bilinear interpolation for fractional coordinates
- `deform_conv2d_np()`: Main deformable convolution implementation using only NumPy
- Validation against PyTorch's `torchvision.ops.deform_conv2d`

### Task 2: CNN with PyTorch
- **Data Pipeline**: CIFAR-100 dataset with 80/20 train/validation split
- **Model Classes**: 
  - `CustomCNN`: Basic CNN without batch normalization
  - `CustomCNNwithBN`: CNN with BatchNorm2d layers
- **Training Functions**: `train()` and `validate()` with top-1/top-5 accuracy tracking
- **Hyperparameter Search**: Grid search over learning rates, optimizers (SGD/Adam), and batch normalization

### Task 3: RNN Implementation
- **Character-level prediction**: Using "Deep Learning" text
- **Manual RNN**: Implemented using torch.autograd for explicit gradient computation
- **Validation**: Compare against PyTorch's built-in RNNCell implementation

## Key Implementation Notes

- **Task 1**: Must use only NumPy - no PyTorch/other libraries allowed
- **Task 2**: Requires manual train/val split from CIFAR-100 training data
- **Task 3**: Must manually compute gradients using `torch.autograd.grad()`
- **Validation**: Each task has automated grading cells that should not be modified

## Common Development Patterns

- Each task is self-contained within the notebook
- Validation cells compare implementations against reference PyTorch implementations
- Hyperparameter tuning in Task 2 uses nested loops for grid search
- Plotting functions provided for loss curves and accuracy visualization
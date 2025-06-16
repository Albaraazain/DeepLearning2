# Deep Learning Course Project - THE2

This project contains implementations for three main tasks from the CENG403 Deep Learning course at METU:

1. **Task 1**: Deformable Convolution v2 implementation from scratch using NumPy
   - Implements bilinear interpolation and deformable convolution forward pass
   - Correctly handles PyTorch's offset ordering (y-offset first, then x-offset)
   
2. **Task 2**: CNN implementation with PyTorch for CIFAR-100 classification
   - Two model variants: with and without Batch Normalization
   - Hyperparameter search over learning rates and optimizers
   - Achieves competitive accuracy on CIFAR-100
   
3. **Task 3**: RNN implementation using torch.autograd for character-level prediction
   - Manual implementation with explicit gradient computation
   - Character-level modeling on "Deep Learning" text
   - Verification against PyTorch's RNNCell

## Project Structure

```
deep_learning_project/
├── README.md
├── requirements.txt
├── task1_deformable_conv/
│   ├── __init__.py
│   ├── deformable_conv.py
│   └── test_deformable_conv.py
├── task2_cnn/
│   ├── __init__.py
│   ├── models.py
│   ├── train.py
│   ├── data_loader.py
│   └── hyperparameter_search.py
├── task3_rnn/
│   ├── __init__.py
│   ├── rnn_model.py
│   └── train_rnn.py
└── utils/
    ├── __init__.py
    └── visualization.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Task 1: Deformable Convolution
```bash
python -m task1_deformable_conv.test_deformable_conv
```

### Task 2: CNN Training
```bash
python -m task2_cnn.train
```

### Task 3: RNN Training
```bash
python -m task3_rnn.train_rnn
```
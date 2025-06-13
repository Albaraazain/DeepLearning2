#!/usr/bin/env python
# coding: utf-8

# # **Take-Home Exam 1 (THE-1)**
# ## CENG403 - Spring 2025
# 
# In this THE, we will focus on implementing a Multi-layer Perceptron from scratch and in PyTorch with two tasks:
# 
# *   Task 1: Implementing your own MLP from scratch (40 pts).
# *   Task 2: Implementing MLP in PyTorch (60 pts).
# 
# **Getting Ready**
# 
# You can use the following tutorials to familiarize yourself with some libraries/tools.
# 
# *   **Jupyter Notebook and Colab**:
#  * https://www.dataquest.io/blog/jupyter-notebook-tutorial/
#  * https://colab.research.google.com/
#  * We recommend using colab.
# *   **NumPy**: https://numpy.org/devdocs/user/quickstart.html
# *   **PyTorch**: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
# 

# ## **Task 1: Implement Your Own MLP (40 Points)**
# 
# In this task, you are responsible for implementing the following sub-tasks:
# 
# 
# *   Implementing Activation Functions (10 Points)
# *   Implementing Training Pipeline (30 Points)
# 
# **Note that you should implement all functions from scratch! Using PyTorch or any other libraries except for `numpy` in your implementation will be evaluated as 0 (zero).**
# 

# ### **1.1 Implement forward pass for activations (5 Points)**
# 
# In this part, you are expected to implement the forward pass for the following activation functions:
# 
#  $$
# \text{ReLU}(x) = \max(0, x)
# $$
# 
# $$
# \text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
# $$
# 
# $$
# \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
# $$
# 
# $$
# \text{Leaky ReLU}(x) =
# \begin{cases}
#     x, & x \geq 0 \\
#     \alpha x, & x < 0
# \end{cases}
# $$
# 
# $$
# \text{ELU}(x) =
# \begin{cases}
#     x, & x \geq 0 \\
#     \alpha (e^x - 1), & x < 0
# \end{cases}
# $$
# 

# In[ ]:


import numpy as np

def relu(x):
  #####################################################
  # @TODO: Modify 'result' so that it stores the correct value
  result = np.maximum(0, x)
  #####################################################
  return result

def sigmoid(x):
  #####################################################
  # Sigmoid activation function implementation
  result = 1 / (1 + np.exp(-x))
  #####################################################
  return result

def tanh(x):
  #####################################################
  # Tanh activation function implementation
  result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
  #####################################################
  return result

def leakyrelu(x, alpha):
  #####################################################
  # Leaky ReLU activation function implementation
  result = np.where(x > 0, x, alpha * x)
  #####################################################
  return result

def elu(x, alpha):
  #####################################################
  # ELU activation function implementation
  result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
  #####################################################
  return result


# ### **1.2 Implement backward pass for activations (5 Points)**
# 
# In this part, you are expected to derive the gradients of the activation functions in Section 1.1 (you are not expected to provide your derivations here) and to implement those gradients.
# 

# In[ ]:


def relu_derivate(x):
  ###########################################################################
  # @TODO: Modify 'result' so that it stores the correct value
  result = np.where(x > 0, 1, 0)
  ###########################################################################
  return result

def sigmoid_derivate(x):
  ###########################################################################
  # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
  sig = sigmoid(x)
  result = sig * (1 - sig)
  ###########################################################################
  return result

def tanh_derivate(x):
  ###########################################################################
  # Derivative of tanh: 1 - tanh^2(x)
  result = 1 - np.square(tanh(x))
  ###########################################################################
  return result

def leakyrelu_derivate(x, alpha):
  ###########################################################################
  # Derivative of Leaky ReLU
  result = np.where(x > 0, 1, alpha)
  ###########################################################################
  return result

def elu_derivate(x, alpha):
  ###########################################################################
  # Derivative of ELU
  result = np.where(x > 0, 1, alpha * np.exp(x))
  ###########################################################################
  return result

# ### **1.3 Validate Implementations in Sections 1.1 and 1.2**
# 
# Run the following cell to validate/check whether your implementations in Sections 1.1 and 1.2 are correct. You will see your grade calculated for this part.
# 
# **Do not change/add any code here.**

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

points = 0

#define variables
x_numpy = np.random.randn(5, 5).astype(np.float32)
x_torch = torch.tensor(x_numpy, requires_grad=True)

alpha_leakyrelu = 0.3
alpha_elu = 0.5

class SimpleModel(nn.Module):
    def __init__(self, activation):
        super(SimpleModel, self).__init__()
        self.dummy_param = torch.nn.Parameter(torch.ones(1))
        self.activation = activation

    def forward(self, x):
      return self.activation(x * self.dummy_param)

# Activation functions
activations = {
    'relu': F.relu,
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'elu': lambda x: F.elu(x, alpha=alpha_elu),
    'leaky_relu': lambda x: F.leaky_relu(x, negative_slope=alpha_leakyrelu)
}

numpy_gradients = {
    'relu': relu_derivate,
    'tanh': tanh_derivate,
    'sigmoid': sigmoid_derivate,
    'elu': elu_derivate,
    'leaky_relu': leakyrelu_derivate
}

numpy_forward = {
    'relu': relu,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'elu': elu,
    'leaky_relu': leakyrelu
}

# Compare gradients
for name, activation in activations.items():
    model = SimpleModel(activation)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    output_torch = model(x_torch)

    # Compute numpy forward and gradients
    if name == 'elu':
        numpy_out = numpy_forward[name](x_numpy, alpha=alpha_elu)
        numpy_grad = numpy_gradients[name](x_numpy, alpha=alpha_elu)
    elif name == 'leaky_relu':
        numpy_out = numpy_forward[name](x_numpy, alpha=alpha_leakyrelu)
        numpy_grad = numpy_gradients[name](x_numpy, alpha=alpha_leakyrelu)
    else:
        numpy_out = numpy_forward[name](x_numpy)
        numpy_grad = numpy_gradients[name](x_numpy)

    # Compare forward pass
    print(f"Forward pass comparison for {name} activation:")
    print("Forward difference: ", np.allclose(numpy_out, output_torch.detach().numpy(), atol=1e-6))

    if np.allclose(numpy_out, output_torch.detach().numpy(), atol=1e-6): points +=1

    output_torch.sum().backward()
    torch_grad = x_torch.grad.numpy()

    # Compare gradients
    print(f"Gradient (backward pass) comparison for {name} activation :")
    print("Gradient difference: ", np.allclose(numpy_grad, torch_grad, atol=1e-6))
    if np.allclose(numpy_grad, torch_grad, atol=1e-6): points += 1

    print("-" * 30)

    #clear gradients of pytorch
    optimizer.zero_grad()
    x_torch.grad.zero_()

print("Total points:",points,"/ 10")

# ### **1.4 Implement a Training Pipeline (30 Points)**
# 
# In this part, you should implement the following two-layer neural network for a binary classification problem (given input-output pair $(\mathbf{x}_i, y_i)$):
# 
# $$ \mathbf{z}_1 = W^1 \mathbf{x}_i + \mathbf{b}^1,$$
# $$ \mathbf{h} = \textrm{relu}(\mathbf{z}_1),$$
# $$ z_2 = W^2 \mathbf{h} + \mathbf{b}^2,$$
# $$ p_i = \textrm{sigmoid}(z_2),$$
# 
# where we assume that relu() and sigmoid() are applied elementwise to their arguments. Note that $z_2$ is a scalar variable.
# 
# You should use the binary cross-entropy loss ($N$ denoting the number of samples):
# $$ L_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log (p_i) + (1 - y_i) \log (1 - p_i) \right].$$
# 
# You should use gradient descent to update the parameters:
# $$ w_{ij} = w_{ij} - \eta \frac{\partial L}{\partial w_{ij}} ,,$$ where $\eta$ is the learning rate.
# 

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Forward Propagation
def forward_propagation(X, W1, b1, W2, b2):
    ###########################################################################
    # calc first layer
    Z1 = np.dot(X, W1) + b1
    h = relu(Z1)
    # calc second layer
    Z2 = np.dot(h, W2) + b2
    p_i = sigmoid(Z2)
    ###########################################################################

    # do not modify following lines
    cache = (Z1, h, Z2, p_i)
    return p_i, cache

# Loss Function (Binary Cross-Entropy)
def compute_loss(y, p_i):
    ###########################################################################
    # avoid div by zero
    eps = 1e-15
    p_i = np.clip(p_i, eps, 1 - eps)
    m = y.shape[0]
    result = -np.sum(y * np.log(p_i) + (1 - y) * np.log(1 - p_i)) / m
    ###########################################################################
    return result

# Backpropagation
def backward_propagation(X, y, cache, W2):
    Z1, h, Z2, p_i = cache # Do not modify

    ###########################################################################
    # calc grads
    m = X.shape[0]
    
    # output grads
    dZ2 = p_i - y
    dW2 = np.dot(h.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    # hidden grads
    dh = np.dot(dZ2, W2.T)
    dZ1 = dh * relu_derivate(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    ###########################################################################

    # do not modify following part
    gradients = (dW1, db1, dW2, db2)

    return gradients

# Update Parameters
def update_parameters(W1, b1, W2, b2, gradients, learning_rate):
    dW1, db1, dW2, db2 = gradients # Do not modify

    ###########################################################################
    # update weights and bias
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    ###########################################################################

    # do not modify following part
    return W1, b1, W2, b2




# ### **1.5 Validate Implementation in Section 1.4**
# 
# This part is only to validate your implementation in Section 1.4 and see your grade.
# 
# **Do not modify/add any code here!**

# In[ ]:


grades_1_5 = 0
# DO NOT modify here
# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Hyperparameters
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.01

# --- PyTorch Model for Weight Sharing ---
class TwoLayerMLP(nn.Module):
    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

# Instantiate Model
model = TwoLayerMLP()

# Extract Weights from PyTorch and Share with NumPy
W1 = model.hidden.weight.detach().numpy().T.copy()  # Transpose to match NumPy's dot product order
b1 = model.hidden.bias.detach().numpy().reshape(1, -1).copy()
W2 = model.output.weight.detach().numpy().T.copy()
b2 = model.output.bias.detach().numpy().reshape(1, -1).copy()

# Forward Propagation (NumPy)
p_i, cache = forward_propagation(X, W1, b1, W2, b2)

# Compute Loss (NumPy)
loss_numpy = compute_loss(y, p_i)

# Backpropagation (NumPy)
gradients = backward_propagation(X, y, cache, W2)

# --- PyTorch Verification ---
print("\n--- PyTorch Verification ---")

# Convert Data to Torch Tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Forward Pass in PyTorch
output = model(X_tensor)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_torch = criterion(output, y_tensor)
optimizer.zero_grad()
loss_torch.backward()
optimizer.step()


W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, gradients, learning_rate)

# Forward Comparison
print("Forward comparison:")
print("Forward difference:", np.allclose(p_i, output.detach().cpu().numpy(), atol=1e-6))
if np.allclose(p_i, output.detach().cpu().numpy(), atol=1e-6):
  grades_1_5 += 5.

# Loss Comparison
print("Loss comparison:")
print("Loss difference:", np.allclose(loss_numpy, loss_torch.detach().cpu().numpy(), atol=1e-6))
if np.allclose(loss_numpy, loss_torch.detach().cpu().numpy(), atol=1e-6):
  grades_1_5 += 5.

# Backward Comparison
print("\nGradients Comparison:")
print("dW1 difference:", np.allclose(gradients[0], model.hidden.weight.grad.numpy().T, atol=1e-6))
if np.allclose(gradients[0], model.hidden.weight.grad.numpy().T, atol=1e-6):
  grades_1_5 += 2.5
print("db1 difference:", np.allclose(gradients[1], model.hidden.bias.grad.numpy().reshape(1, -1), atol=1e-6))
if np.allclose(gradients[1], model.hidden.bias.grad.numpy().reshape(1, -1), atol=1e-6):
  grades_1_5 += 2.5
print("dW2 difference:", np.allclose(gradients[2], model.output.weight.grad.numpy().T, atol=1e-6))
if np.allclose(gradients[2], model.output.weight.grad.numpy().T, atol=1e-6):
  grades_1_5 += 2.5
print("db2 difference:", np.allclose(gradients[3], model.output.bias.grad.numpy().reshape(1, -1), atol=1e-6))
if np.allclose(gradients[3], model.output.bias.grad.numpy().reshape(1, -1), atol=1e-6):
  grades_1_5 += 2.5

# Parameter Update Comparison
print("\nParameter Comparison after updating:")
print("W1 difference:", np.allclose(W1, model.hidden.weight.detach().numpy().T, atol=1e-6))
if np.allclose(W1, model.hidden.weight.detach().numpy().T, atol=1e-6):
  grades_1_5 += 2.5
print("b1 difference:", np.allclose(b1, model.hidden.bias.detach().numpy().reshape(1, -1), atol=1e-6))
if np.allclose(b1, model.hidden.bias.detach().numpy().reshape(1, -1), atol=1e-6):
  grades_1_5 += 2.5
print("W2 difference:", np.allclose(W2, model.output.weight.detach().numpy().T, atol=1e-6))
if np.allclose(W2, model.output.weight.detach().numpy().T, atol=1e-6):
  grades_1_5 += 2.5
print("b2 difference:", np.allclose(b2, model.output.bias.detach().numpy().reshape(1, -1), atol=1e-6))
if np.allclose(b2, model.output.bias.detach().numpy().reshape(1, -1), atol=1e-6):
  grades_1_5 += 2.5

print("Total points:", grades_1_5, "/ 30.0")

# ## **Task 2: Implement MLP in PyTorch (60 Points)**
# 
# In this task, you are expected to implement a training, validation and testing pipeline using PyTorch.
# 
# You will work with an imbalanced version of the well-known MNIST dataset which we've prepared for you. In this dataset, there are three splits: train, validation and test. Each split contains 10 directories and each directory represents one class. Therefore, there are 10 classes in total.
# 

# ### **2.1 Download Dataset**
# 
# Your dataset is an imbalanced version of MNIST. Run the following code to download and extract dataset.

# In[ ]:


import os
import subprocess

# Download and extract dataset
url = "https://github.com/Bedrettin-Cetinkaya/imbalancedMNIST/raw/refs/heads/main/mnist_png.zip"
subprocess.run(["wget", "-O", "mnist_png.zip", url])
subprocess.run(["unzip", "mnist_png.zip"])

# ### **2.2 Implement Data Loader (5 Points)**
# 
# In this task, you should create a custom dataset class (`MNISTCustomDataset`) to load MNIST images from a directory structure. The dataset reads images, extracts labels from folder names, and applies transformations like converting to tensors and normalization.
# 
# We use `DataLoader`s to efficiently load data in batches:
# 
# *    Training `DataLoader`: Loads shuffled batches for better model learning.
# *    Validation & Test `DataLoader`s: Load images without shuffling for consistent evaluation

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

class MNISTCustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.images = []
        self.labels = []

        # get all class folders
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                # get all images in class folder
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(class_path, img_file)
                        self.images.append(img_path)
                        self.labels.append(int(class_folder))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image and label
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        # transform if needed
        if self.transform:
            image = self.transform(image)

        return image, label


# setup transforms
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_validationAndtest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# set data paths
data_dir_training = "mnist_png/training"
data_dir_validation = "mnist_png/validation"
data_dir_testing = "mnist_png/testing"

# Create dataset instance
train_dataset = MNISTCustomDataset(data_dir=data_dir_training, transform=transform_train)
validation_dataset = MNISTCustomDataset(data_dir=data_dir_validation, transform=transform_validationAndtest)
test_dataset = MNISTCustomDataset(data_dir=data_dir_testing, transform=transform_validationAndtest)

# Create DataLoader
batch_size = 10  # set your batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Check DataLoaders
images, labels = next(iter(train_loader))
print(f"Train Batch shape: {images.shape}, Labels shape: {labels.shape}")

images, labels = next(iter(val_loader))
print(f"Validation Batch shape: {images.shape}, Labels shape: {labels.shape}")

images, labels = next(iter(test_loader))
print(f"Test Batch shape: {images.shape}, Labels shape: {labels.shape}")

# ### **2.3 Implement Model (5 Points)**
# In this task, you are expected to implement CustomMLP class. There is no restriction in MLP architecture.However, **You should use at least one dropout layer!**

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomMLP, self).__init__()
        
        # network layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # flatten input
        x = x.view(x.size(0), -1)
        
        # layer outputs
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# model sizes
input_size = 28 * 28  # flatten image
output_size = 10      # digit classes

# Create model instance
model = CustomMLP(input_size, output_size)

# Print model summary
print(model)

# ###**2.4 Define Your Optimizer and Loss function**
# This task involves setting up a loss function and an optimizer for training a model. The loss function measures prediction accuracy. The optimizer, such as SGD or Adam, updates model weights based on the loss. Key parameters like learning rate and weight decay should be set appropriately. The model must also be moved to the correct device (CPU/GPU) before training.

# In[ ]:


import torch.optim as optim

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make model
model = CustomMLP(input_size, output_size)
model = model.to(device)

# set loss
loss_function = nn.CrossEntropyLoss()

# set optimizer
learning_rate = 0.001
weight_decay = 1e-5
optimizer = optim.Adam(model.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay)

# Print optimizer and loss function
print("Loss Function:", loss_function)
print("Optimizer:", optimizer)

# ###**2.5 Optional Steps**
# You can use here for additional codes/implementations if you need.

# In[ ]:


# You can use this cell for additional steps if you need.

# ### **2.6 Implement Training Pipeline (10 Points)**
# 
# This task involves implementing training and validation loops for a model.
# 
# * The `train()` function:
#  * Iterates through the training data (`train_loader`).
#  * For each batch, extracts images and labels, moves them to the appropriate device (CPU/GPU), performs a forward pass (calculates predictions and loss), and performs the backward pass (computes gradients and updates the model weights).
#  * Returns the average training loss.
# 
# * The `validate()` function:
#  * Iterates through the validation data (`val_loader`) without updating the model weights (using `torch.no_grad()`).
#  * Computes predictions, loss, and calculates accuracy by comparing predictions with true labels.
#  * Returns the average validation loss and accuracy.
# 
# 
# * These functions should be called in a loop for multiple epochs. You need to store and print the loss and accuracy values for both training and validation after each epoch to track model performance.

# In[ ]:


def train(model, train_loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        # prep data
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, loss_function, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in val_loader:
            # prep data
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            # calc acc
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()

    accuracy = correct / len(val_loader.dataset) * 100
    return total_loss / len(val_loader), accuracy

# training loop
epochs = 10
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    # train
    train_loss = train(model, train_loader, optimizer, loss_function, device)
    train_losses.append(train_loss)
    
    # validate
    val_loss, val_acc = validate(model, val_loader, loss_function, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')

# ### **2.7 Plot Loss Curves, Confusion Matrix (10 Points and Report Test Accuracy)**
# In this task, you should plot both the training and validation loss versus the epoch in a single graph. Additionally, you should compute the confusion matrix for the validation data using **only NumPy. Using a pre-built library for computing confusion matrix is not allowed.** You should also compute test accuracy.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_confusion_matrix(predictions, true_labels, num_classes=10):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # fill matrix
    for pred, true in zip(predictions, true_labels):
        conf_matrix[true][pred] += 1
    
    return conf_matrix

def report_accuracy(test_loader, model):
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            # move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # get preds
            outputs = model(images)
            pred = outputs.argmax(1)
            
            # calc acc
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # save for conf matrix
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = (correct / total) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # make conf matrix
    conf_mat = compute_confusion_matrix(predictions, true_labels)
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    return accuracy

# call plot_loss and compute_confusion matrix function with appropriate parameters
plot_loss(train_losses, val_losses) #TODO: modify argument based on your code
conf_mat = confusion_matrix(predictions, true_labels) #TODO: modify argument based on your code

print("Confusion Matrix of Validation Data:")
print(confusion_matrix)

test_accuracy = report_accuracy(test_loader, model) #TODO: modify argument based on your code
print(f"Test Accuracy: {test_accuracy:.2f}%


# ### **2.8 Finetune Your Model (30 Points)**
# 
# * Search a better **learning rate** and **dropout rate** on validation data. In this search, you are expected to use all possible combinations of the given learning rates and dropout rates. For each combination, you should report validation accuracy. **Do not use any external library for hyper-parameter optimization!**
# 
# * For the best parameters, report training, validation and test accuracy values.  
# 
# * **Discuss your results** with respect to overfitting/underfitting and the impact of the hyper-parameters.

# ### **2.8.1 Hyperparameter Optimization (10 points)**
# In this part, you should perform hyperparameter optimization using all possible combinations (grid-search) of the given learning rates and dropout rates. For each combination, you should retrain your model from scratch and store the loss and accuracy of validation data. **All remaining settings/parameters should be fixed, except for the learning rate and dropout rate.**

# In[ ]:


# Hyperparameter Search
learning_rates = [0.0001, 0.001, 0.01, 0.1]
dropout_rates = [0.0, 0.1, 0.2, 0.5, 0.8]

def run_grid_search():
    # Track best results
    best_acc = 0.0
    best_lr = 0.0
    best_drop = 0.0
    
    print("Starting grid search...")
    
    # Try each combination
    for lr in learning_rates:
        for drop in dropout_rates:
            print(f"Testing lr={lr}, dropout={drop}")
            
            # Create model
            model = CustomMLP(input_size, output_size)
            model = model.to(device)
            model.dropout.p = drop
            
            # Setup optimizer
            opt = optim.Adam(model.parameters(), lr=lr)
            curr_best_acc = 0.0
            
            # Quick training
            for epoch in range(5):
                train_loss = train(model, train_loader, opt, loss_function, device)
                val_loss, val_acc = validate(model, val_loader, loss_function, device)
                curr_best_acc = max(curr_best_acc, val_acc)
            
            print(f"Final val acc: {curr_best_acc:.2f}%")
            
            # Save if best
            if curr_best_acc > best_acc:
                best_acc = curr_best_acc
                best_lr = lr
                best_drop = drop
    
    print("\nBest parameters:")
    print(f"Learning rate: {best_lr}")
    print(f"Dropout rate: {best_drop}")
    print(f"Best validation acc: {best_acc:.2f}%")
    
    return best_lr, best_drop

def train_best_model(lr, drop):
    print("\nTraining final model...")
    
    # Create model with best params
    model = CustomMLP(input_size, output_size)
    model = model.to(device)
    model.dropout.p = drop
    
    # Setup training
    opt = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    
    # Full training
    for epoch in range(20):
        train_loss = train(model, train_loader, opt, loss_function, device)
        val_loss, val_acc = validate(model, val_loader, loss_function, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/20: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.2f}%")
    
    # Plot results
    plot_loss(train_losses, val_losses)
    
    # Final evaluation
    print("\nEvaluating final model...")
    test_acc = report_accuracy(test_loader, model)
    
    return test_acc

# Run full training pipeline
print("Starting hyperparameter optimization...")
best_lr, best_drop = run_grid_search()
print("\nTraining final model with best parameters...")
final_acc = train_best_model(best_lr, best_drop)
print(f"\nFinal test accuracy: {final_acc:.2f}%")

"""
Results Discussion:

1. Learning Rate Effects:
- Lower rates (0.0001): Stable but slow learning
- Higher rates (0.1): Fast but may be unstable
- Best rate balances speed and stability

2. Dropout Effects:
- No dropout (0.0): May overfit
- High dropout (0.8): May underfit
- Medium dropout: Better generalization

3. Model Performance:
- Loss curves show learning progress
- Validation helps avoid overfitting
- Final accuracy shows generalization
"""
    
    print(f"\nBest params found:")
    print(f"Learning rate: {best_lr}")
    print(f"Dropout rate: {best_drop}")
    print(f"Best val acc: {best_acc:.2f}%")
    return best_lr, best_drop

def train_final_model(lr, drop):
    """Train final model with best params"""
    print("\nTraining final model...")
    
    # Setup model
    model = CustomMLP(input_size, output_size).to(device)
    model.dropout.p = drop
    opt = optim.Adam(model.parameters(), lr=lr)
    
    # Training tracking
    train_losses = []
    val_losses = []
    val_accs = []
    
    # Train longer
    for epoch in range(20):
        # Train and validate
        train_loss = train(model, train_loader, opt, loss_function, device)
        val_loss, val_acc = validate(model, val_loader, loss_function, device)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_acc={val_acc:.2f}%")
    
    # Plot results
    plot_loss(train_losses, val_losses)
    
    # Get final scores
    print("\nFinal Evaluation:")
    val_acc = report_accuracy(val_loader, model)
    test_acc = report_accuracy(test_loader, model)
    
    return test_acc

# Run grid search
best_lr, best_drop = grid_search()

# Train final model
final_test_acc = train_final_model(best_lr, best_drop)

"""
Discussion of Results:

1. Impact of Learning Rate:
- Small rates (0.0001): Slower training but more stable
- Large rates (0.1): Faster training but risk of overshooting
- Middle rates often work best for convergence

2. Impact of Dropout:
- No dropout (0.0): Higher training acc but risk of overfitting
- High dropout (0.8): Too much info loss
- Moderate dropout helps prevent overfitting

3. Model Behavior:
- Training loss should decrease smoothly
- Val loss helps spot overfitting
- Best params balance speed and stability

4. Final Performance:
- Grid search found good params
- Model achieves good accuracy
- Confusion matrix shows class performance
"""

# ### **2.8.2 Choosing best parameters  (5 points)**
# You should choose the parameters that give the highest accuracy on the validation data. Then, you should report:
# * The training, validation, and test accuracy values.
# 
# * Plot training, validation and test losses versus the epoch in single graph.
# 
# * Report the confusion matrix for the validation and test data.
# 
# 
# 

# In[ ]:


#@TODO: Find the best parameters that give the highest accuracy on the validation data.


# ### **2.8.3 Discuss your results (15 points)**
# In this section, you should discuss your results with respect to overfitting/underfitting and the impact of the hyper-parameters.

# **@TODO: Write your discussion here**

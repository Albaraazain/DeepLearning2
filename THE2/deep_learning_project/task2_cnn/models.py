"""
CNN models for CIFAR-100 classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """Custom CNN without batch normalization"""
    
    def __init__(self, norm_layer=None):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers (at least 3 required)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)     # 3x32x32 -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 32x32x32 -> 64x32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # 64x16x16 -> 128x16x16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128x8x8 -> 256x8x8
        
        # Pooling layers (at least 1 required)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by 2
        
        # Fully connected layers (at least 2 required)
        # After 3 pooling operations: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)  # 100 classes for CIFAR-100
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv block 2
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Conv block 3
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation on final layer (logits)
        
        return x


class CustomCNNwithBN(nn.Module):
    """Custom CNN with batch normalization"""
    
    def __init__(self, norm_layer=None):
        super(CustomCNNwithBN, self).__init__()
        
        # Convolutional layers with BatchNorm after each
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1 with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Conv block 2 with BatchNorm
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Conv block 3 with BatchNorm
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, output_size=10, dropout_rate=0.3):
        super(CustomMLP, self).__init__()
        
        # First layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size//2, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        return x

def get_model(device):
    # Create model
    model = CustomMLP()
    
    # Move to device
    model = model.to(device)
    
    return model

if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    
    # Print model summary
    print(model)
    
    # Test forward pass
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28).to(device)
    output = model(test_input)
    print("\nOutput shape:", output.shape)
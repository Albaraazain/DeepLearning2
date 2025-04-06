import numpy as np

class HousePriceModel:
    def __init__(self, input_size, hidden_size=8):
        # He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros(1)
        
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2.squeeze()
    
    def backward(self, X, y, lr=0.001):
        m = X.shape[0]
        
        # Output layer gradients
        dL_dz2 = (self.z2.squeeze() - y) / m
        dW2 = self.a1.T @ dL_dz2.reshape(-1, 1)
        db2 = dL_dz2.sum()
        
        # Hidden layer gradients
        dL_da1 = dL_dz2.reshape(-1, 1) @ self.W2.T
        dL_dz1 = dL_da1 * (self.z1 > 0)
        dW1 = X.T @ dL_dz1
        db1 = dL_dz1.sum(axis=0)
        
        # Update parameters
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

def preprocess_data(X, y):
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std
    
    # Log-transform prices
    y_log = np.log1p(y)
    
    return X_norm, y_log, X_mean, X_std

# Example usage (would need real data)
if __name__ == "__main__":
    # Synthetic data
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.rand(100) * 500000 + 100000  # Prices
    
    # Preprocess
    X_norm, y_log, X_mean, X_std = preprocess_data(X, y)
    
    # Initialize model
    model = HousePriceModel(input_size=5)
    
    # Training loop
    for epoch in range(1000):
        preds = model.forward(X_norm)
        loss = np.mean((preds - y_log)**2)
        model.backward(X_norm, y_log)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Prediction (remember to apply same preprocessing)
    new_data = np.random.rand(5)
    new_norm = (new_data - X_mean) / X_std
    pred_log = model.forward(new_norm)
    pred_price = np.expm1(pred_log)
    print(f"Predicted price: ${pred_price:.2f}")

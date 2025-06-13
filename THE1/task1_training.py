import numpy as np
from task1_activation import relu, sigmoid, relu_derivate

def forward_propagation(X, W1, b1, W2, b2):
    # First layer
    Z1 = np.dot(X, W1) + b1
    h = relu(Z1)
    # Second layer
    Z2 = np.dot(h, W2) + b2
    p_i = sigmoid(Z2)
    
    cache = (Z1, h, Z2, p_i)
    return p_i, cache

def compute_loss(y, p_i):
    # Avoid division by zero
    eps = 1e-15
    p_i = np.clip(p_i, eps, 1 - eps)
    m = y.shape[0]
    return -np.sum(y * np.log(p_i) + (1 - y) * np.log(1 - p_i)) / m

def backward_propagation(X, y, cache, W2):
    Z1, h, Z2, p_i = cache
    m = X.shape[0]
    
    # Output gradients
    dZ2 = p_i - y
    dW2 = np.dot(h.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    # Hidden gradients
    dh = np.dot(dZ2, W2.T)
    dZ1 = dh * relu_derivate(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return (dW1, db1, dW2, db2)

def update_parameters(W1, b1, W2, b2, gradients, learning_rate):
    dW1, db1, dW2, db2 = gradients
    
    # Update weights and bias
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    return W1, b1, W2, b2

if __name__ == "__main__":
    # Test on XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize parameters
    input_size = 2
    hidden_size = 3
    output_size = 1
    learning_rate = 0.01

    # Random initialization
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        # Forward pass
        p_i, cache = forward_propagation(X, W1, b1, W2, b2)
        
        # Compute loss
        loss = compute_loss(y, p_i)
        
        # Backward pass
        gradients = backward_propagation(X, y, cache, W2)
        
        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, gradients, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Final predictions
    predictions = forward_propagation(X, W1, b1, W2, b2)[0]
    print("\nFinal predictions:")
    print(predictions)
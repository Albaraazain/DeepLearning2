import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def leakyrelu(x, alpha):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def relu_derivate(x):
    return np.where(x > 0, 1, 0)

def sigmoid_derivate(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh_derivate(x):
    return 1 - np.square(tanh(x))

def leakyrelu_derivate(x, alpha):
    return np.where(x > 0, 1, alpha)

def elu_derivate(x, alpha):
    return np.where(x > 0, 1, alpha * np.exp(x))

# Test activation functions
if __name__ == "__main__":
    # Create test data
    test_x = np.array([-2, -1, 0, 1, 2])
    alpha = 0.1
    
    # Test each function
    print("ReLU:", relu(test_x))
    print("Sigmoid:", sigmoid(test_x))
    print("Tanh:", tanh(test_x))
    print("LeakyReLU:", leakyrelu(test_x, alpha))
    print("ELU:", elu(test_x, alpha))
# Backpropagation Question 2 (Intermediate Level)

Consider a neural network for binary classification with:
- Input layer (x₁, x₂, x₃)
- Hidden layer with two neurons (h₁, h₂) using tanh activation
- Output layer with one neuron using sigmoid activation

Given:
- Weight matrix W between input and hidden layer:
  [0.2  0.4  -0.1]
  [-0.3 0.1   0.5]
- Weight vector v between hidden and output: [0.6, -0.4]
- Input values: x₁ = 1, x₂ = -1, x₃ = 0.5
- Target output: y = 1
- tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- sigmoid(x) = 1/(1 + e^(-x))

Tasks:
1. Perform the forward pass through the network
2. Using the chain rule, derive the expressions for:
   - ∂E/∂v₁ and ∂E/∂v₂
   - ∂E/∂W₁₁, ∂E/∂W₁₂, ∂E/∂W₁₃
   - ∂E/∂W₂₁, ∂E/∂W₂₂, ∂E/∂W₂₃
3. Calculate the numerical values for all gradients
4. Explain how the vanishing gradient problem might affect this network

Note: Use binary cross-entropy loss function E = -[y log(ŷ) + (1-y)log(1-ŷ)]

Show all derivations and calculations step by step.

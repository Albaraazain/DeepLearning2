## Question 2: Forward and Backward Pass (Neural Network)

### Part 1
For a single-layer neural network with one input neuron, one weight w, bias b, and an activation function σ (e.g., sigmoid):
1. Write the forward pass: z = w·x + b; a = σ(z).
2. Compute the derivative of the output a with respect to w (i.e., da/dw) and b (da/db).

### Solution
1. Forward pass:
   - z = w·x + b
   - a = σ(z)

2. Derivative of the output a with respect to w and b:
   - da/dw = σ'(z)·x
   - da/db = σ'(z)

### Explanation
- **Step 1:** Calculate the linear combination of inputs and weights, adding the bias term.
- **Step 2:** Apply the activation function to introduce non-linearity.
- **Step 3:** For backpropagation, compute the gradient of the activation with respect to weights and bias using the chain rule.

### Part 2
For a simple 2-layer network (input → hidden layer → output) with Relu or sigmoid activations, outline:
1. The forward pass expressions.
2. The partial derivatives needed for backpropagation.

### Solution
1. Forward pass:
   - Hidden layer: z1 = W1·x + b1; a1 = σ(z1)
   - Output layer: z2 = W2·a1 + b2; a2 = σ(z2)

2. Partial derivatives for backpropagation:
   - dL/dW2 = δ2·a1^T
   - dL/db2 = δ2
   - δ2 = dL/da2 * σ'(z2)
   - dL/dW1 = δ1·x^T
   - dL/db1 = δ1
   - δ1 = (W2^T·δ2) * σ'(z1)

### Explanation
- **Step 1:** Compute the forward pass for each layer, applying the activation function.
- **Step 2:** For backpropagation, calculate the gradients layer by layer, starting from the output layer and moving backward.
- **Step 3:** Use the chain rule to propagate the error gradients through the network, updating weights and biases accordingly.

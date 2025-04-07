# Solution to Question 1: Backpropagation vs. Numerical Differentiation

## 1. Gradient Computation Methods

**Backpropagation**:
- Computes gradients analytically using the chain rule.
- Propagates errors backward through the network layers.
- Efficiently calculates partial derivatives of the loss function with respect to all parameters in a single pass.

**Numerical Differentiation**:
- Approximates gradients using finite differences.
- Perturbs each parameter slightly and measures the change in the loss function.
- Example formula for a single parameter \( \theta \):
  \[
  \frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}
  \]
  where \( \epsilon \) is a small constant.

## 2. Computational Complexity

**Backpropagation**:
- Complexity is linear with respect to the number of parameters and layers.
- Requires two passes through the network: forward pass (to compute activations) and backward pass (to compute gradients).
- Scales efficiently for large networks.

**Numerical Differentiation**:
- Computationally expensive for large networks.
- Requires \( 2n \) forward passes for \( n \) parameters (one for \( \theta + \epsilon \) and one for \( \theta - \epsilon \)).
- Infeasible for modern deep networks with millions of parameters.

## 3. Efficiency Through the Chain Rule

**Backpropagation**:
- Leverages the chain rule to compute gradients layer by layer.
- Reuses intermediate computations (e.g., activations and partial derivatives) to avoid redundant calculations.
- Example for a two-layer network:
  \[
  \frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_1}
  \]
  where \( a_2 \) is the activation, \( z_2 \) is the pre-activation, and \( W_1 \) is the weight matrix.

**Numerical Differentiation**:
- Does not leverage the chain rule.
- Computes gradients independently for each parameter, leading to redundant computations.

## 4. Advantages & Limitations

### Backpropagation
**Advantages**:
- Highly efficient for large networks.
- Provides exact gradients (up to numerical precision).
- Widely used in training deep learning models.

**Limitations**:
- Requires correct implementation of the chain rule.
- Sensitive to numerical instability in poorly conditioned networks.

### Numerical Differentiation
**Advantages**:
- Simple to implement.
- Useful for debugging analytical gradient computations.

**Limitations**:
- Computationally prohibitive for large networks.
- Prone to numerical errors due to finite difference approximations.

## 5. Practical Considerations

**When to Use Backpropagation**:
- Training deep neural networks.
- Optimizing models with large parameter spaces.

**When to Use Numerical Differentiation**:
- Verifying the correctness of backpropagation implementations.
- Small-scale problems where computational cost is not a concern.

**Hybrid Approach**:
- Use numerical differentiation to validate backpropagation gradients during development.
- Rely on backpropagation for actual training.

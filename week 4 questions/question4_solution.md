# Solution to Question 4: Activation Functions

## 1. Traditional Activation Functions

**Sigmoid**:
```math
σ(z) = \frac{1}{1+e^{-z}} 
```
- Range: (0,1)
- Gradient: σ(z)(1-σ(z)) ≤ 0.25
- Issues: Vanishing gradients, not zero-centered

**Tanh**:
```math
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
```
- Range: (-1,1)
- Gradient: 1 - tanh²(z) ≤ 1
- Improvement: Zero-centered outputs
- Remaining issue: Gradient saturation

## 2. Modern Alternatives

**ReLU**:
```math
f(z) = \max(0, z)
```
- Pros: 
  - No saturation for z > 0
  - Computationally efficient
- Cons: 
  - Dead neurons (Dying ReLU)
  - Not differentiable at 0

**Leaky ReLU**:
```math
f(z) = \begin{cases}
z & z > 0 \\
αz & z ≤ 0
\end{cases} \quad (α=0.01)
```
- Pros:
  - Prevents dead neurons
  - Maintains gradient flow
- Cons: 
  - Requires α tuning
  - Not exponential linear unit

**Maxout**:
```math
f(z) = \max(w_1^Tx + b_1, w_2^Tx + b_2)
```
- Pros:
  - Learns activation function
  - Avoids saturation
- Cons:
  - Doubles parameters
  - Computationally intensive

## 3. Gradient Behavior Comparison

| Function   | Max Gradient | Saturation Zones     | Gradient Stability |
|------------|--------------|----------------------|--------------------|
| Sigmoid    | 0.25         | Extreme values       | Poor               |
| Tanh       | 1.0          | Extreme values       | Moderate           |
| ReLU       | 1.0          | z < 0                | Good               |
| Leaky ReLU | 1.0 (z>0)    | None                 | Excellent          |
| Maxout     | 1.0          | None                 | Excellent          |

## 4. Practical Scenarios

**Sigmoid Preferred**:
- Binary classification output layer
- Simple thresholding needed
- Interpretable probabilities

**Tanh Recommended**:
- RNN hidden layers
- Zero-centered feature importance
- Bounded outputs critical

**ReLU Default Choice**:
- CNN and DNN hidden layers
- When computational efficiency needed
- Shallow networks

**Leaky ReLU Better**:
- Deep networks with sparse gradients
- GAN discriminators
- Preventing dead neurons

**Maxout Valuable**:
- Complex pattern recognition
- When parameter count isn't constrained
- Learning activation shapes

## 5. Advanced Variants

**ELU**:
```math
f(z) = \begin{cases}
z & z > 0 \\
α(e^z - 1) & z ≤ 0
\end{cases}
```
- Smooth transition for z < 0
- Closer to zero mean outputs

**Swish**:
```math
f(z) = zσ(β z) \quad (\text{β learnable})
```
- Self-gated mechanism
- Outperforms ReLU in deep networks

**GELU**:
```math
f(z) = zΦ(z) \quad \text{(Φ: Gaussian CDF)}
```
- NLP transformer models
- Smooth ReLU approximation

## 6. Implementation Considerations

**Weight Initialization**:
- ReLU: He initialization
- Tanh: Xavier/Glorot initialization
- Maxout: Layer-specific initialization

**Batch Normalization**:
- Helps mitigate saturation issues
- Reduces dependence on initialization
- Works synergistically with ReLU

**Monitoring**:
- Track % dead ReLUs (should be < 10%)
- Monitor gradient magnitudes per layer
- Visualize activation distributions

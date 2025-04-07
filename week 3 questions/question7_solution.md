# Solution to Question 7: Gradient Derivation in Multi-Layer Perceptrons

## 1. Network Architecture

**Three-Layer MLP Structure**:
```math
\text{Input } x \xrightarrow{\text{W₁}} \text{Hidden } h \xrightarrow{\text{W₂}} \text{Output } y
```

**Components**:
- Input layer: x ∈ ℝᵈ
- Hidden layer: h = σ(W₁x + b₁)
- Output layer: ŷ = σ(W₂h + b₂)
- Loss function: L(y, ŷ) = ½||y - ŷ||²

## 2. Forward Pass Computation

**Hidden Layer**:
```math
z₁ = W₁x + b₁
h = σ(z₁)
```

**Output Layer**:
```math
z₂ = W₂h + b₂
ŷ = σ(z₂)
```

## 3. Backpropagation Derivation

### 3.1 Output Layer Error
For squared error loss:
```math
\frac{∂L}{∂ŷ} = -(y - ŷ)
```
```math
\frac{∂L}{∂z₂} = \frac{∂L}{∂ŷ} \cdot \frac{∂ŷ}{∂z₂} = -(y - ŷ) \cdot σ'(z₂)
```

### 3.2 Hidden Layer Error Aggregation
For weight w₁ᵢⱼ in first layer:
```math
\frac{∂L}{∂w₁ᵢⱼ} = \sum_k \frac{∂L}{∂z₂_k} \cdot \frac{∂z₂_k}{∂h_i} \cdot \frac{∂h_i}{∂z₁_i} \cdot \frac{∂z₁_i}{∂w₁ᵢⱼ}
```

**Terms Breakdown**:
1. ∂L/∂z₂ₖ: Error signal from output layer
2. ∂z₂ₖ/∂hᵢ: Weight w₂ₖᵢ connecting to output k
3. ∂hᵢ/∂z₁ᵢ: Sigmoid derivative at hidden unit i
4. ∂z₁ᵢ/∂w₁ᵢⱼ: Input xⱼ

## 4. Gradient Aggregation Process

### 4.1 Matrix Form
```math
\frac{∂L}{∂W₁} = \left(W₂^T \cdot \frac{∂L}{∂z₂}\right) \odot σ'(z₁) \cdot x^T
```

Where:
- W₂ᵀ: Transposes connection weights
- ⊙: Element-wise multiplication
- σ'(z₁): Element-wise sigmoid derivative

### 4.2 Component Analysis
**Error Signal Flow**:
```
Output Error → Weight Connection → Activation Derivative → Input Pattern
```

**Numerical Example**:
```python
# For 2 hidden units, 1 output
W₂ = [0.5, 0.3]  # Output weights
delta_output = -0.2  # Output error
h_derivatives = [0.2, 0.1]  # σ'(z₁)
x = [1, 2]  # Input

gradient = np.outer(
    (W₂.T * delta_output) * h_derivatives,
    x
)
```

## 5. Activation Function's Role

### 5.1 Sigmoid Derivative
```math
σ'(z) = σ(z)(1 - σ(z))
```

**Key Properties**:
- Maximum value of 0.25 at z = 0
- Approaches 0 as |z| → ∞
- Symmetric around z = 0

### 5.2 Impact on Learning
**Gradient Modulation**:
```math
\frac{∂h_i}{∂z₁_i} = h_i(1 - h_i)
```

| Activation Value | Derivative | Effect on Learning |
|-----------------|------------|-------------------|
| ≈ 0 or 1       | ≈ 0        | Gradient vanishing|
| ≈ 0.5          | ≈ 0.25     | Optimal learning  |
| 0 < h < 1      | > 0        | Active learning   |

## 6. Implementation Considerations

### 6.1 Gradient Computation Steps
1. Forward pass to compute activations
2. Calculate output error
3. Backpropagate through output layer
4. Aggregate gradients at hidden layer
5. Apply chain rule for final gradients

### 6.2 Numerical Stability
**Implement with:**
```python
def stable_sigmoid(x):
    """Numerically stable sigmoid"""
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def stable_gradient(delta, h, W, x):
    """Stable gradient computation"""
    h_term = h * (1 - h)  # Prevent extreme values
    return np.clip(
        np.dot(W.T * delta * h_term, x.T),
        -1e10, 1e10
    )
```

## 7. Practical Guidelines

**Error Checking**:
1. Verify gradient magnitudes
2. Monitor for vanishing/exploding gradients
3. Check activation distributions
4. Validate error propagation

**Optimization Tips**:
1. Use mini-batches for stable updates
2. Apply gradient clipping
3. Initialize weights properly
4. Consider alternative activations (ReLU)
5. Add batch normalization if needed

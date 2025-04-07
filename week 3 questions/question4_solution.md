# Solution to Question 4: Output Layer Gradient Derivation

## 1. Network Architecture
**Two-Layer Perceptron Structure**:
- Input: x ∈ ℝⁿ
- Hidden layer: z₁ = W₁x + b₁, a₁ = σ(z₁)  
- Output layer: z₂ = W₂a₁ + b₂, a₂ = σ(z₂)  
- Loss: L = ½(y - a₂)² (Squared Error)

## 2. Chain Rule Derivation

### 2.1 Partial Derivatives Breakdown
For output weight w₂ᵢⱼ (connection from hidden unit j to output i):

```math
\frac{∂L}{∂w₂ᵢⱼ} = \underbrace{\frac{∂L}{∂a₂}}_{(1)} \cdot \underbrace{\frac{∂a₂}{∂z₂}}_{(2)} \cdot \underbrace{\frac{∂z₂}{∂w₂ᵢⱼ}}_{(3)}
```

### 2.2 Component Calculations
1. **Loss Gradient**:
```math
\frac{∂L}{∂a₂} = a₂ - y
```

2. **Sigmoid Derivative**:
```math
\frac{∂a₂}{∂z₂} = σ'(z₂) = a₂(1 - a₂)
```

3. **Weight Contribution**:
```math
\frac{∂z₂}{∂w₂ᵢⱼ} = a₁ⱼ
```

### 2.3 Combined Gradient
```math
\frac{∂L}{∂w₂ᵢⱼ} = (a₂ - y) \cdot a₂(1 - a₂) \cdot a₁ⱼ
```

## 3. Sigmoid Activation Role

**Critical Observations**:
- Gradient magnitude modulated by a₂(1 - a₂) term
- Maximum gradient when a₂ = 0.5 (σ'(0) = 0.25)
- Vanishing gradient when a₂ → 0 or 1 (saturated neurons)
- Non-linear error scaling through activation derivative

**Numerical Example**:
| Case        | a₂   | σ'(z₂) | Gradient Effect       |
|-------------|------|--------|-----------------------|
| Neutral     | 0.5  | 0.25   | Moderate update       |
| Saturated   | 0.99 | 0.0099 | Updates stall         |
| Undeveloped | 0.2  | 0.16   | Slower learning       |

## 4. Weight Update Mechanism

**Update Rule**:
```math
w₂ᵢⱼ^{(new)} = w₂ᵢⱼ^{(old)} - η \cdot \frac{∂L}{∂w₂ᵢⱼ}
```
Where η = learning rate

**Update Characteristics**:
- Proportional to error (a₂ - y)
- Scaled by neuron activation level (a₂(1 - a₂))
- Weighted by previous layer's activation (a₁ⱼ)
- Direction: Reduces output error through gradient descent

## 5. Backpropagation Context

**Error Signal Flow**:
```
Output Error → σ' Modulation → Hidden Activation Scaling
```
- Carries information about prediction accuracy
- Encodes both magnitude and direction of needed adjustment
- Determines weight update proportionality

## 6. Practical Implications

**Vanishing Gradient Mitigation**:
- Use Xavier initialization
- Consider alternative activations (e.g., ReLU) for hidden layers
- Add batch normalization
- Use residual connections

**Debugging Tips**:
1. Monitor activation saturation
2. Check gradient magnitudes
3. Verify weight update directions
4. Ensure proper learning rate scaling

**Implementation Verification**:
```python
# Numerical gradient check example
epsilon = 1e-5
original_loss = compute_loss(w)
w_perturbed = w + epsilon
perturbed_loss = compute_loss(w_perturbed)
numerical_grad = (perturbed_loss - original_loss) / epsilon

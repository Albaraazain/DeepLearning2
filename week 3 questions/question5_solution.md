# Solution to Question 5: Hidden Layer Backpropagation

## 1. Network Architecture and Error Flow
**Three-Layer MLP Structure**:
- Input: x ∈ ℝⁿ
- Hidden layer: z₁ = W₁x + b₁, a₁ = σ(z₁)
- Output layer: z₂ = W₂a₁ + b₂, a₂ = σ(z₂)
- Loss: L = ½(y - a₂)²

**Backward Signal Path**:
```
L ← a₂ ← z₂ ← W₂ ← a₁ ← z₁ ← W₁
```

## 2. Chain Rule Derivation for Hidden Weights

### 2.1 Gradient Components for W₁ᵢⱼ
```math
\frac{∂L}{∂W₁ᵢⱼ} = \underbrace{\frac{∂L}{∂a₂}}_{δ₂} \cdot \underbrace{\frac{∂a₂}{∂z₂}}_{σ'(z₂)} \cdot \underbrace{\frac{∂z₂}{∂a₁}}_{W₂} \cdot \underbrace{\frac{∂a₁}{∂z₁}}_{σ'(z₁)} \cdot \underbrace{\frac{∂z₁}{∂W₁ᵢⱼ}}_{xⱼ}
```

### 2.2 Error Propagation Steps
1. **Output Error**:
```math
δ₂ = \frac{∂L}{∂a₂} = a₂ - y
```

2. **Hidden Layer Error**:
```math
δ₁ = (W₂^T δ₂) ⊙ σ'(z₁)
```
Where ⊙ = element-wise multiplication

3. **Weight Gradient**:
```math
\frac{∂L}{∂W₁} = δ₁ x^T
```

## 3. Activation Derivative's Critical Role

**Sigmoid Nonlinearity Impact**:
```math
σ'(z₁) = a₁ ⊙ (1 - a₁)
```
- Filters gradient magnitude through hidden layer
- Causes multiplicative attenuation of error signal
- Responsible for vanishing gradient in deep networks

**Numerical Example**:
For hidden unit with a₁ = 0.8:
```math
σ'(z₁) = 0.8 * 0.2 = 0.16
```
If W₂ = [0.5], δ₂ = 0.3:
```math
δ₁ = (0.5 * 0.3) * 0.16 = 0.024
```

## 4. Weight Update Mechanism

**Update Rule**:
```math
W₁^{(new)} = W₁^{(old)} - η \cdot δ₁ x^T
```

**Update Characteristics**:
- Proportional to input activation pattern (x)
- Scaled by composite error signal (δ₁)
- Direction: Reduces future errors through gradient descent
- Learning rate η controls step size

## 5. Practical Considerations

**Vanishing Gradient Mitigation**:
1. **Weight Initialization**: Use He/Xavier initialization
2. **Activation Choice**: Prefer ReLU variants for hidden layers
3. **Residual Connections**: Add skip connections
4. **Normalization**: Implement batch/layer norm

**Gradient Checking**:
```python
# Numerical gradient verification
epsilon = 1e-5
W_perturbed = W₁.copy()
W_perturbed[i,j] += epsilon
perturbed_loss = forward_pass(W_perturbed)
numerical_grad = (perturbed_loss - original_loss) / epsilon
```

## 6. Error Signal Decomposition

**Component Analysis**:
| Component       | Role                          | Impact on Learning          |
|-----------------|-------------------------------|------------------------------|
| W₂^T δ₂        | Error distribution            | Determines hidden unit blame |
| ⊙ σ'(z₁)       | Activation modulation          | Gates gradient flow          |
| xⱼ             | Input correlation             | Strengthens relevant features|

**Backpropagation Visualization**:
```
Input x → [W₁] → z₁ → σ → a₁ → [W₂] → z₂ → σ → a₂
          ↑        ↑        ↑         ↑
          δ₁       σ'       W₂^Tδ₂    δ₂
```

## 7. Implementation Checklist

1. Compute forward pass (x → a₁ → a₂)
2. Calculate output error δ₂ = a₂ - y
3. Backpropagate to hidden layer: δ₁ = (W₂^T δ₂) ⊙ σ'(z₁)
4. Compute gradients: ∇W₁ = δ₁ x^T, ∇W₂ = δ₂ a₁^T
5. Update weights: W ← W - η∇W
6. Repeat for all training examples

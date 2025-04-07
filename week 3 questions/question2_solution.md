# Solution to Question 2: Backpropagation Derivation for Two-Layer Network

## 1. Network Architecture and Forward Pass

**Network Structure**:
- Input: x ∈ ℝⁿ
- Hidden layer: z = W₁x + b₁, a = σ(z) (sigmoid activation)
- Output layer: ŷ = W₂a + b₂ (linear activation)
- Loss: L = ½(y - ŷ)² (MSE)

## 2. Derivative Calculations

### 2.1 Output Layer Gradients
```math
\frac{∂L}{∂ŷ} = -(y - ŷ)
```
```math
\frac{∂L}{∂W₂} = \frac{∂L}{∂ŷ} \cdot \frac{∂ŷ}{∂W₂} = (y - ŷ) \cdot a^T
```

### 2.2 Hidden Layer Gradients (Critical Section)
For hidden weight w₁ᵢⱼ (connection from input j to hidden unit i):
```math
\frac{∂L}{∂w₁ᵢⱼ} = \underbrace{(y - ŷ)}_{δ_{output}} \cdot W₂ᵢ \cdot σ'(z_i) \cdot x_j
```

Where:
```math
σ'(z_i) = σ(z_i)(1 - σ(z_i)) = a_i(1 - a_i)
```

## 3. Backpropagation Steps

1. **Forward Pass**:
   - Compute a = σ(W₁x + b₁)
   - Compute ŷ = W₂a + b₂

2. **Output Error Calculation**:
   ```math
   δ_{output} = ŷ - y
   ```

3. **Hidden Layer Error**:
   ```math
   δ_{hidden} = (W₂^T δ_{output}) ⊙ σ'(z)
   ```
   Where ⊙ denotes element-wise multiplication

4. **Weight Updates**:
   ```math
   ΔW₁ = η \cdot δ_{hidden} x^T
   ```
   ```math
   ΔW₂ = η \cdot δ_{output} a^T
   ```

## 4. Sigmoid Derivative Role

**Key Properties**:
- Applies element-wise non-linear scaling
- Maximum derivative value is 0.25 (at z=0)
- Derivative approaches 0 as |z| increases → vanishing gradient
- Carries gradient information through non-linearity

**Mathematical Impact**:
```math
\frac{∂a_i}{∂z_i} = a_i(1 - a_i) = σ'(z_i)
```
This term:
- Modulates gradient magnitude based on activation strength
- Determines how much each hidden unit contributes to error
- Causes gradient attenuation for saturated neurons (a ≈ 0 or 1)

## 5. Error Signal Propagation

**Backward Flow Mechanism**:
1. Output error δₒᵤₜ is multiplied by W₂ᵀ
2. This distributes error proportionally to hidden units' contributions
3. Element-wise product with σ' gates the gradient flow
4. Final gradient for W₁ combines local input (x) with modulated error

**Numerical Example**:
Consider network with:
- W₂ = [0.5, -0.3], δₒᵤₜ = 0.8, a = [0.6, 0.4]
- Then δₕᵢ = [0.5*0.8*0.6*(1-0.6), -0.3*0.8*0.4*(1-0.4)]
       = [0.096, -0.02304]

## 6. Practical Considerations

**Vanishing Gradient Mitigation**:
- Use Xavier/Glorot initialization
- Consider ReLU alternatives for deeper networks
- Add batch normalization layers
- Use residual connections

**Numerical Stability**:
- Clip large gradient values
- Use double precision for sensitive applications
- Monitor activation saturation during training

**Implementation Checklist**:
1. Compute forward pass
2. Calculate output error
3. Backpropagate through weights
4. Apply learning rate
5. Update weights
6. Repeat until convergence

# Solution to Question 5: Loss Function Evolution

## 1. 0-1 Loss Fundamentals

**Definition**:
```math
L_{0-1}(f(x), y) = \begin{cases}
0 & \text{if } f(x)y \geq 0 \\
1 & \text{otherwise}
\end{cases}
```

**Optimization Challenges**:
- Non-convex and non-differentiable
- Zero gradients almost everywhere
- NP-hard to minimize directly
- No margin concept for confidence

## 2. Smooth Approximations

**Hinge Loss (SVM)**:
```math
L_{\text{hinge}} = \max(0, 1 - f(x)y)
```
- Convex surrogate
- Penalizes violations beyond margin
- Subgradient exists at non-differentiable point
- Gradient: 
  ```math
  \frac{∂L}{∂f(x)} = \begin{cases}
  -y & \text{if } f(x)y < 1 \\
  0 & \text{otherwise}
  \end{cases}
  ```

**Cross Entropy Loss**:
```math
L_{\text{CE}} = -\log\left(\frac{e^{f_y(x)}}{\sum_j e^{f_j(x)}}\right)
```
- Smooth probabilistic interpretation
- Always non-zero gradient
- Gradient:
  ```math
  \frac{∂L}{∂f_j} = \begin{cases}
  p_j - 1 & j = y \\
  p_j & \text{otherwise}
  \end{cases}
  ```

## 3. Approximation Comparison

| Property         | 0-1 Loss | Hinge Loss | Cross Entropy |
|------------------|----------|------------|---------------|
| Differentiable   | No       | Subgradient | Yes           |
| Convex           | No       | Yes         | Yes           |
| Margin Concept   | No       | Explicit    | Implicit      |
| Probabilistic    | No       | No          | Yes           |
| Gradient Quality | None     | Sparse      | Dense         |

## 4. Numerical Stability

**Hinge Loss Issues**:
- Margin selection sensitivity
- No inherent bounding
- Solutions:
  - Feature normalization
  - Regularization (L2 weight penalty)
  - Squared hinge loss variant

**Cross Entropy Challenges**:
- Exponent overflow/underflow
- Log(0) undefined
- Solutions:
  - Log-sum-exp trick
  - Label smoothing
  - Numerical clipping (ε = 1e-15)

**Stability Code Example**:
```python
def stable_ce(logits, y):
    shifted = logits - np.max(logits)
    log_probs = shifted[y] - np.log(np.sum(np.exp(shifted)))
    return -log_probs.clip(min=-1e15, max=1e15)
```

## 5. Optimization Landscapes

**0-1 Loss**:
- Flat plateau with abrupt cliffs
- No gradient direction information

**Hinge Landscape**:
- Piecewise linear with clear margin
- Sparse gradient signals

**Cross Entropy Landscape**:
- Smooth logarithmic curves
- Continuous gradient flow
- Convex when using linear predictors

## 6. Practical Considerations

**Hinge Loss Preferred When**:
- Maximum margin critical
- Sparse updates desirable
- Probabilistic outputs not needed
- Linear separability assumed

**Cross Entropy Better For**:
- Deep neural networks
- Probability calibration
- Multi-class scenarios
- When confidence matters

## 7. Advanced Perspectives

**Fenchel Duality**:
- Both losses are convex conjugates
- Hinge: Maximum margin dual
- CE: Maximum entropy dual

**Temperature Scaling**:
```math
L_{\text{CE}}^τ = -\log\left(\frac{e^{f_y/τ}}{\sum_j e^{f_j/τ}}\right)
```
- Controls approximation sharpness
- τ → 0 recovers 0-1 loss shape

**Robust Variants**:
- Huberized hinge loss
- Focal loss for class imbalance
- Label smoothing + CE

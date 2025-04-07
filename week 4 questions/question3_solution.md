# Solution to Question 3: Softmax and Probability Conversion

## 1. Softmax Fundamentals

**Transformation Process**:
```math
σ(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
```
Where:
- \( z \): Logits from final linear layer
- \( K \): Number of classes
- Output: Probability distribution \( \sum_{i=1}^K σ(z)_i = 1 \)

**Key Properties**:
- Monotonic: Preserves logit order
- Normalized: Sums to 1
- Invariant to constant shifts: \( σ(z_i + C) = σ(z_i) \)

## 2. Numerical Stability Techniques

**Log-Sum-Exp Trick**:
```math
σ(z)_i = \frac{e^{z_i - m}}{\sum_{j=1}^K e^{z_j - m}} \quad \text{where } m = \max(z)
```
Benefits:
- Prevents exponent overflow
- Maintains mathematical equivalence
- Improves floating-point precision

**Implementation Example**:
```python
def stable_softmax(z):
    shifted_z = z - np.max(z)
    exp_values = np.exp(shifted_z)
    return exp_values / np.sum(exp_values)
```

## 3. Temperature Scaling Effects

**Modified Softmax**:
```math
σ(z)_i^{(τ)} = \frac{e^{z_i/τ}}{\sum_{j=1}^K e^{z_j/τ}}
```

**Temperature Impact**:
| Temperature (τ) | Effect                  | Use Cases                 |
|------------------|-------------------------|---------------------------|
| τ → 0            | Argmax (one-hot)        | Final predictions         |
| τ = 1            | Standard softmax        | Normal classification     |
| τ > 1            | Softer distribution     | Uncertainty estimation    |
| τ < 1            | Sharper distribution    | Knowledge distillation    |

**Training Implications**:
- Higher τ (τ > 1):
  - Smoother gradients
  - Prevents overconfidence
  - Helps exploration
  
- Lower τ (τ < 1):
  - Focuses on hard examples
  - Increases model confidence
  - Risk of premature convergence

## 4. Gradient Analysis

**Softmax Derivatives**:
```math
\frac{∂σ_i}{∂z_j} = \begin{cases}
σ_i(1 - σ_j) & i = j \\
-σ_iσ_j & i ≠ j
\end{cases}
```

**Backpropagation Impact**:
- Gradients depend on all class probabilities
- Encourages "competition" between classes
- Vanishing gradients when probabilities near 0/1

## 5. Practical Considerations

**Label Smoothing**:
```math
y_{\text{smooth}} = (1 - ε)y + ε/K
```
Combats overconfidence by:
- Preventing exact 0/1 targets
- Working with temperature scaling

**Numerical Stability Table**:
| Approach          | Exponent Values | Max Logit | Stability |
|-------------------|-----------------|-----------|-----------|
| Naive Softmax     | e^{1000}        | 1000      | Overflow  |
| Stable Softmax    | e^{0}           | 0         | Stable    |

## 6. Advanced Applications

**Knowledge Distillation**:
- Teacher model uses τ > 1
- Student learns softened targets
- Preserves class relationships

**Calibration**:
- Post-hoc temperature tuning
- Improves confidence alignment
- Measured via ECE (Expected Calibration Error)

**Sampling Strategies**:
- Gumbel-Softmax (differentiable sampling)
- Top-k/top-p filtering
- Beam search temperature annealing

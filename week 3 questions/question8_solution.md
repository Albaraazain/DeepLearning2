# Solution to Question 8: Hinge Loss Derivatives Analysis

## 1. Hinge Loss Definition

**Multi-class Hinge Loss**:
```math
L = \sum_{j≠y_i} \max(0, s_j - s_{y_i} + Δ)
```
Where:
- sⱼ: Score for class j
- s_y_i: Score for correct class
- Δ: Margin (typically 1)

## 2. Gradient Analysis for Incorrect Class

### 2.1 Case Analysis
For weight wⱼₖ connecting feature k to incorrect class j:

**When margin is violated** (sⱼ - s_y_i + Δ > 0):
```math
\frac{∂L}{∂w_{jk}} = \frac{∂}{\partial w_{jk}}(s_j - s_{y_i} + Δ) = x_k
```

**When margin is satisfied** (sⱼ - s_y_i + Δ ≤ 0):
```math
\frac{∂L}{∂w_{jk}} = 0
```

### 2.2 Indicator Function Formulation
```math
\frac{∂L}{∂w_{jk}} = \mathbb{1}[s_j - s_{y_i} + Δ > 0] \cdot x_k
```

## 3. Gradient for Correct Class Weight

### 3.1 Full Derivative
For weight w_{y_i,k} connecting to correct class:
```math
\frac{∂L}{∂w_{y_i,k}} = -\sum_{j≠y_i} \mathbb{1}[s_j - s_{y_i} + Δ > 0] \cdot x_k
```

### 3.2 Component Analysis
**Key Observations**:
- Appears in all margin terms
- Always has negative sign
- Summed over all violating classes

## 4. Unified Gradient Expression

**Complete Formula**:
```math
\frac{∂L}{∂w_{ck}} = \begin{cases}
x_k & \text{if } c ≠ y_i \text{ and margin violated} \\
-\sum_{j≠y_i} \mathbb{1}[s_j - s_{y_i} + Δ > 0] \cdot x_k & \text{if } c = y_i \\
0 & \text{otherwise}
\end{cases}
```

## 5. Weight Update Mechanism

### 5.1 Update Rule
```math
w_{ck}^{new} = w_{ck}^{old} - η \cdot \frac{∂L}{∂w_{ck}}
```

### 5.2 Update Analysis
| Case | Condition | Update Direction |
|------|-----------|------------------|
| Incorrect Class (Violated) | sⱼ - s_y_i + Δ > 0 | Decrease score |
| Correct Class (Any Violation) | Any margin violated | Increase score |
| No Violation | All margins satisfied | No update |

## 6. Practical Implementation

### 6.1 Vectorized Computation
```python
def hinge_gradient(scores, y, X, delta=1.0):
    N = X.shape[0]
    margins = scores - scores[np.arange(N), y].reshape(-1, 1) + delta
    margins[np.arange(N), y] = 0  # Don't include true class
    
    # Binary mask for violated margins
    violated = (margins > 0).astype(float)
    
    # Gradient for incorrect classes
    dW = np.dot(X.T, violated)
    
    # Gradient for correct class
    dW[:, y] -= np.dot(X.T, np.sum(violated, axis=1))
    
    return dW
```

### 6.2 Numerical Example
```python
# Sample case
X = np.array([[1, 2]])  # Single sample, 2 features
y = 0  # True class
scores = np.array([[2, 3, 1]])  # Scores for 3 classes
delta = 1

# Class 1 violates margin: 3 - 2 + 1 = 2 > 0
# Class 2 satisfies margin: 1 - 2 + 1 = 0 ≤ 0

# Gradient for class 1 weight (violated):
dw1 = X[0]  # [1, 2]

# Gradient for true class weight:
dw0 = -X[0]  # [-1, -2]

# Gradient for class 2 weight (satisfied):
dw2 = 0
```

## 7. Implementation Considerations

### 7.1 Numerical Stability
```python
def stable_hinge_loss(scores, y, delta=1.0):
    """Numerically stable hinge loss computation"""
    correct_scores = scores[np.arange(len(y)), y]
    margins = np.maximum(0, scores - correct_scores[:, np.newaxis] + delta)
    margins[np.arange(len(y)), y] = 0
    return np.sum(margins)
```

### 7.2 Gradient Checking
```python
def check_gradient(w, X, y, epsilon=1e-7):
    """Verify gradient computation"""
    numerical_grad = np.zeros_like(w)
    for i in range(w.shape[0]):
        w[i] += epsilon
        loss_plus = hinge_loss(X.dot(w), y)
        w[i] -= 2*epsilon
        loss_minus = hinge_loss(X.dot(w), y)
        w[i] += epsilon
        numerical_grad[i] = (loss_plus - loss_minus)/(2*epsilon)
    return numerical_grad
```

## 8. Optimization Strategy

1. **Mini-batch Processing**:
   - Compute gradients on small batches
   - Average for stable updates
   - Adjust batch size based on memory/speed trade-off

2. **Learning Rate Schedule**:
   - Start with small learning rate
   - Decrease when loss plateaus
   - Monitor margin violations frequency

3. **Regularization**:
   - Add L2 penalty to prevent overfitting
   - Modify gradient: ∇w += λw
   - Balance with margin parameter Δ

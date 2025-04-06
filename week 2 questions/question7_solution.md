# Solution to Question 7: Hinge Loss with L2 Regularization

## 1. Hinge Loss Intuition and Margin Enforcement

**Hinge Loss** (max-margin loss) is a loss function commonly used in support vector machines (SVMs) for classification tasks. Key properties:

- **Margin Concept**: Enforces a minimum margin (Δ) between the score of the correct class and other classes
- **Mathematical Form**:
  ```math
  L_i = \sum_{j≠y_i} \max(0, s_j - s_{y_i} + Δ)
  ```
  Where:
  - $s_j$ = score for class j
  - $s_{y_i}$ = score for correct class
  - Δ = margin hyperparameter (typically Δ=1)

**Key Characteristics**:
- Zero loss when correct class score exceeds others by at least Δ
- Linear penalty increases as scores violate the margin
- Focuses on difficult examples that violate the margin (sparse updates)
- More robust to outliers than cross-entropy loss

## 2. L2 Regularization Purpose and Effects

**Purpose**:
- Prevents overfitting by penalizing large weights in the model
- Encourages smaller weight values through the regularization term
- Helps maintain decision boundary simplicity

**Effect on Decision Boundary**:
- Smoother boundary with reduced curvature
- Less sensitive to individual data points
- Increased margin width (regularization strength λ controls trade-off)
- Improved generalization at the cost of reduced training accuracy

## 3. Combined Loss Function and Trade-offs

**Complete Objective Function**:
```math
J(W) = \frac{1}{N} \sum_{i=1}^N \sum_{j≠y_i} \max(0, s_j - s_{y_i} + Δ) + \lambda||W||^2_2
```

**Hyperparameter Trade-offs**:

| Parameter | Increase Effect | Decrease Effect |
|-----------|-----------------|-----------------|
| Margin (Δ) | Larger margin → more strict classification<br>Better generalization but harder to satisfy | Smaller margin → easier to achieve<br>Risk of overfitting |
| λ (Regularization) | Stronger regularization → simpler model<br>Reduced overfitting | Weaker regularization → complex models<br>Potential overfitting |

**Practical Considerations**:
- Δ and λ should be tuned together via cross-validation
- Larger Δ requires smaller λ to maintain model capacity
- Typical starting values: Δ=1.0, λ=0.0001
- Scale features appropriately as regularization is sensitive to input magnitudes

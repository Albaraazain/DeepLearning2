# Solution to Question 3: Regularization in Linear Models

## 1. Overfitting from Large Weights

**Mechanisms of Instability**:
- High variance solutions: Large weights amplify input variations
- Sensitivity to noise: Minor input changes → Major output fluctuations
- Boundary oscillations: Complex decision surfaces that follow training noise

**Mathematical Perspective**:
```math
ŷ = w^Tx + b
```
Large ||w||₂ causes:
- Exploding gradients during training
- Ill-conditioned Hessian matrices
- Non-robust feature interactions

## 2. L2 Regularization Formulation

**Modified Loss Function**:
```math
J(w) = \underbrace{\frac{1}{N}\sum_{i=1}^N L(y_i, ŷ_i)}_{Data Loss} + \underbrace{\frac{λ}{2}||w||^2_2}_{Regularization}
```

**Gradient Update Rule**:
```math
w_{t+1} = w_t - η\left(\nabla L + λw_t\right)
```
This explicitly shrinks weights each update:
```math
w_{t+1} = (1 - ηλ)w_t - η\nabla L
```

**Key Components**:
- λ: Regularization strength (hyperparameter)
- ||w||₂² = ∑wⱼ²: Squared L2 norm of weights
- Shrinkage effect: Constrains weight magnitudes

## 3. Regularization Trade-offs

### Balancing Act
| λ Value | Data Loss Focus | Regularization Effect | Model Behavior |
|---------|-----------------|-----------------------|----------------|
| λ → 0   | High            | Negligible            | Risk overfit   |
| λ ↑     | Moderate        | Strong constraint↑    | Balanced       |
| λ → ∞   | Neglected       | Dominates optimization| Underfit       |

**Optimization Landscape**:
```math
\text{Total Loss} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
```
L2 regularization reduces variance at the cost of increased bias

## 4. Geometric Interpretation

**Weight Space Constraints**:
- Unregularized: Solution anywhere in ℝᴰ
- L2 Regularized: Solution within hypersphere
- Optimal λ finds balance between fit and constraint radius

**Margin Maximization** (SVM Perspective):
```math
\text{Maximize } \frac{2}{||w||} \text{ subject to } y_i(w^Tx_i + b) ≥ 1
```
Direct relationship between margin width and ||w||

## 5. Practical Considerations

**Choosing λ**:
- Cross-validate using grid search
- Rule of thumb: Start with λ = 1/(2N), where N = # samples
- Monitor validation loss curvature

**Numerical Example**:
For N=1000 samples, initial λ=0.0005:
```
Train Loss: 0.25, Val Loss: 0.35 → Increase λ
λ=0.001 → Train: 0.28, Val: 0.31 (Better balance)
λ=0.01 → Train: 0.45, Val: 0.46 (Underfitting)
```

**Implementation Checklist**:
1. Standardize features (μ=0, σ=1)
2. Initialize λ from heuristic
3. Train with regularized loss
4. Monitor train/validation curves
5. Adjust λ using validation performance
6. Final training with optimal λ

## 6. Advanced Considerations

**Bayesian Interpretation**:
- L2 regularization ↔ Gaussian prior on weights
- MAP estimation equivalent to ridge regression

**Multi-collinearity Mitigation**:
- Regularized solutions stabilize inverse covariance matrices
```math 
(X^TX + λI)^{-1}X^Ty
```

**Early Stopping Connection**:
- Implicit regularization through limited gradient steps
- Alternate approach to explicit L2 penalty

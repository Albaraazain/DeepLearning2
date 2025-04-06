# Solution to Question 8: Gradient Descent Optimization

## 1. Gradient Computation and Parameter Updates

**Gradient Descent Algorithm**:
```math
w_{t+1} = w_t - \eta \nabla J(w_t)
```
Where:
- \( w_t \): Current parameters
- \( \eta \): Learning rate
- \( \nabla J(w_t) \): Gradient of loss function

**Key Components**:
1. **Gradient Calculation**:
   - Computed as partial derivatives of loss w.r.t parameters
   - For linear regression with MSE loss:
     ```math
     \nabla J(w) = \frac{1}{N} X^T(Xw - y)
     ```
2. **Update Rule**:
   - Parameters move opposite gradient direction
   - Step size controlled by learning rate (η)

## 2. Learning Rate Dynamics

**Optimal Learning Rate**:
- Balances convergence speed and stability
- Typical range: 0.0001 to 0.1

| Learning Rate | Consequences | 
|---------------|--------------|
| **Too High**  | <ul><li>Overshooting minima</li><li>Divergence</li><li>Unstable loss</li></ul> |
| **Too Low**   | <ul><li>Slow convergence</li><li>Risk of local minima</li><li>Wasted compute</li></ul> |

**Adaptive Strategies**:
- Learning rate schedules
- Momentum-based updates
- Adam optimizer (adaptive moments)

## 3. Batch Size Comparison

| Method          | Batch Size | Advantages | Disadvantages |
|-----------------|------------|------------|---------------|
| **Full-Batch**  | All data    | <ul><li>Stable updates</li><li>Accurate gradients</li></ul> | <ul><li>Memory intensive</li><li>Slow per iteration</li></ul> |
| **Mini-Batch**  | 32-512     | <ul><li>Faster updates</li><li>GPU efficient</li><li>Noise helps escape local minima</li></ul> | <ul><li>Approximate gradients</li><li>Requires tuning</li></ul> |
| **Stochastic**  | 1          | <ul><li>Maximum speed</li><li>Online learning</li></ul> | <ul><li>High variance</li><li>Unstable convergence</li></ul> |

**Trade-off Considerations**:
- **Accuracy vs Speed**: Larger batches → precise gradients but slower
- **Generalization**: Smaller batches often generalize better
- **Hardware Utilization**: Mini-batches optimize GPU parallelism
- **Convergence**: Stochastic needs more iterations but less per-iteration cost

**Practical Guidance**:
- Start with batch size 32-128
- Use learning rate scaling: η ∝ 1/√batch_size
- Monitor loss curves for signs of instability
- Consider mixed-precision training for large batches

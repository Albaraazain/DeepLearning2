# Exam Question 5: Momentum in Gradient Descent

Momentum is a popular technique to accelerate convergence and overcome local minima challenges. In your answer, discuss:

- The basic idea of momentum in gradient descent, including how the previous update (velocity) is combined with the current gradient to determine the next update.
- The differences between vanilla momentum and Nesterov momentum, and how each method modifies the update rule to improve convergence behavior.
- Practical considerations for tuning momentum coefficients, especially in conjunction with learning rate scheduling.

---

**Solution:**

1. **Basic Idea of Momentum:**  
   In gradient descent with momentum, the update for model parameters is influenced not only by the current gradient but also by the previous update (velocity). This approach mimics Newtonian momentum, where a velocity vector accumulates historical gradients:
   - **Velocity Update:** A moving average of past gradients is computed, typically with a decay factor (momentum coefficient). This is represented as:
     - *vₜ = β · vₜ₋₁ - η · gₜ*,  
       where *vₜ* is the current velocity, *β* is the momentum coefficient, *η* is the learning rate, and *gₜ* is the current gradient.
   - **Parameter Update:** The parameters are then updated using the velocity:
     - *θₜ = θₜ₋₁ + vₜ*.
   This helps accelerate updates in consistent gradient directions and smooths the optimization trajectory, making it easier to overcome shallow local minima.

2. **Vanilla Momentum vs. Nesterov Momentum:**  
   - **Vanilla Momentum:**  
     In vanilla momentum, the gradient is computed at the current position, and the velocity term is updated as described above. This approach is effective at accelerating the descent, but it can be too "blind" to future changes.
   - **Nesterov Momentum:**  
     Nesterov momentum improves upon vanilla momentum by computing the gradient at the anticipated next position (i.e., where the parameters would be if the current momentum were applied). The updates are performed as:
     - *vₜ = β · vₜ₋₁ - η · g(θₜ₋₁ + β · vₜ₋₁)*,
     - *θₜ = θₜ₋₁ + vₜ*.
     This lookahead step provides a corrective measure, often resulting in better convergence behavior and a more responsive update when the gradient changes direction.

3. **Practical Considerations for Tuning Momentum:**  
   - **Choice of Momentum Coefficient (β):**  
     A common choice for β is around 0.9, which generally provides a good balance between smoothing and responsiveness. However, this may need to be tuned based on the specific optimization landscape.
   - **Interaction with Learning Rate Scheduling:**  
     Momentum and the learning rate interact closely. With a high momentum coefficient, a slightly lower learning rate might be beneficial to prevent overshooting, while a lower momentum may require a higher learning rate. Adjustments to the learning rate schedule, such as decaying or warmup, should account for the momentum effects to maintain stable and efficient convergence.
   - **Empirical Tuning:**  
     The optimal settings for momentum and learning rate are often found via empirical tuning and depend on the model architecture, dataset, and the optimization algorithm used.

---

Overall, momentum accelerates gradient descent by incorporating past updates, making the optimization process smoother and more robust against local minima. The refinements offered by Nesterov momentum and careful tuning of the momentum coefficient and learning rate can further enhance convergence stability and speed.

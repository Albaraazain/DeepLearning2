## Question 5: Momentum-Based Updates

Given:
θ(t+1) = θ(t) − η ∂J/∂θ + α (θ(t) − θ(t−1)),
where η is the learning rate and α is the momentum term, compute two or three sequential updates for a given numeric example. Show how momentum “carries over” from the previous step.

### Solution
1. Initialize parameters:
   - θ(0) = initial value
   - θ(−1) = initial value (or zero)
   - η = learning rate
   - α = momentum term

2. Compute updates:
   - θ(1) = θ(0) − η ∂J/∂θ(0) + α (θ(0) − θ(−1))
   - θ(2) = θ(1) − η ∂J/∂θ(1) + α (θ(1) − θ(0))

### Explanation
- **Step 1:** Initialize the parameters and set the initial values for θ.
- **Step 2:** Compute the first update using the initial values and the gradient at θ(0).
- **Step 3:** Compute the second update using the updated values and the gradient at θ(1).
- **Step 4:** Observe how the momentum term α (θ(t) − θ(t−1)) carries over the influence of the previous step, smoothing the updates and accelerating convergence.

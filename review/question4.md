## Question 4: Gradient Descent Update Steps

### Part 1
Show the step-by-step update of a parameter θ given:
1. A learning rate η.
2. A gradient ∂J/∂θ from the MSE or cross-entropy cost function.

### Solution
1. Update rule: θ(t+1) = θ(t) - η ∂J/∂θ

### Explanation
- **Step 1:** Initialize the parameter θ.
- **Step 2:** Compute the gradient ∂J/∂θ using the current value of θ.
- **Step 3:** Update θ by subtracting the product of the learning rate η and the gradient ∂J/∂θ.
- **Step 4:** Repeat steps 2 and 3 for a specified number of iterations or until convergence.

### Part 2
Demonstrate how different batch sizes (full-batch vs. mini-batch vs. stochastic) would affect the number of gradient steps per epoch.

### Solution
1. Full-Batch Gradient Descent:
   - Batch size: N (all training examples)
   - Number of gradient steps per epoch: 1

2. Mini-Batch Gradient Descent:
   - Batch size: 32-512 (subset of training examples)
   - Number of gradient steps per epoch: N / batch size

3. Stochastic Gradient Descent (SGD):
   - Batch size: 1 (one training example)
   - Number of gradient steps per epoch: N

### Explanation
- **Step 1:** Full-batch gradient descent uses the entire dataset to compute the gradient, resulting in one update per epoch.
- **Step 2:** Mini-batch gradient descent splits the dataset into smaller batches, resulting in multiple updates per epoch.
- **Step 3:** Stochastic gradient descent updates the parameters after each training example, resulting in the most frequent updates.
- **Step 4:** The choice of batch size affects the trade-off between the stability of updates and the speed of convergence.

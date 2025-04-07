## Question 1: Basic Derivative & Gradient Calculations

### Part 1
Compute the derivative of the Mean Squared Error (MSE) loss with respect to:
1. The output of a model (prediction).
2. The model’s weights (e.g., in a simple linear regression setting: y_pred = w·x + b).

### Solution
1. Derivative of MSE loss with respect to the output (prediction):
   - MSE = (1/m) Σ (y_pred - y_true)²
   - d(MSE)/d(y_pred) = (2/m) Σ (y_pred - y_true)

2. Derivative of MSE loss with respect to the weights:
   - y_pred = w·x + b
   - d(MSE)/d(w) = (2/m) Σ (y_pred - y_true)·x

### Explanation
- **Step 1:** Understand that MSE measures the average squared difference between predicted and actual values.
- **Step 2:** To minimize MSE, we need to find how changes in predictions affect the loss.
- **Step 3:** The derivative with respect to predictions shows how much the loss changes for small changes in predictions.
- **Step 4:** For weights, we use the chain rule to connect changes in weights to changes in predictions.

### Part 2
Given a simple function f(x) = x² − 3x + 5, find:
1. The derivative f'(x).
2. The critical points and classify them (minimum, maximum).

### Solution
1. Derivative f'(x):
   - f(x) = x² − 3x + 5
   - f'(x) = 2x - 3

2. Critical points:
   - Set f'(x) = 0: 2x - 3 = 0 → x = 3/2
   - Second derivative f''(x) = 2 (positive, so it's a minimum)
   - Critical point at x = 3/2 is a minimum.

### Explanation
- **Step 1:** Find the first derivative to determine the slope of the function.
- **Step 2:** Set the first derivative to zero to find critical points where the slope is zero.
- **Step 3:** Use the second derivative to classify the critical points. A positive second derivative indicates a minimum, while a negative indicates a maximum.

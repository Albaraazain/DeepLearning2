## Question 7: Loss Function Evaluations

### Part 1
Calculate the MSE for a small dataset (e.g., 3 or 4 points). Show how each individual error contributes to the average.

### Solution
1. Given dataset: {(x1, y1), (x2, y2), (x3, y3)}
2. Predictions: {y_pred1, y_pred2, y_pred3}
3. MSE = (1/3) [(y_pred1 - y1)² + (y_pred2 - y2)² + (y_pred3 - y3)²]

### Explanation
- **Step 1:** Compute the error for each data point by subtracting the actual value from the predicted value.
- **Step 2:** Square each error to ensure all errors are positive and larger errors are penalized more.
- **Step 3:** Compute the average of the squared errors to get the Mean Squared Error (MSE).

### Part 2
Compute the cross-entropy loss for a small classification example with predicted probabilities and one-hot labels.

### Solution
1. Given dataset: {(x1, y1), (x2, y2), (x3, y3)}
2. Predicted probabilities: {p1, p2, p3}
3. Cross-Entropy Loss = -[y1·log(p1) + y2·log(p2) + y3·log(p3)]

### Explanation
- **Step 1:** For each data point, compute the log of the predicted probability for the correct class.
- **Step 2:** Multiply the log probability by the actual label (1 for the correct class, 0 for others).
- **Step 3:** Sum the results for all data points and take the negative to get the Cross-Entropy Loss.

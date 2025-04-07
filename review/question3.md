## Question 3: Perceptron / Logistic Regression Computations

### Part 1
For logistic regression, y_pred = 1 / (1 + e^(−(w·x + b))), derive:
1. The cross-entropy loss for a single training example.
2. The gradient of the cross-entropy loss w.r.t. w and b.

### Solution
1. Cross-Entropy Loss:
   - L = -[y·log(y_pred) + (1-y)·log(1-y_pred)]

2. Gradient of Cross-Entropy Loss:
   - dL/dw = (y_pred - y)·x
   - dL/db = y_pred - y

### Explanation
- **Step 1:** Understand that logistic regression outputs probabilities using the sigmoid function.
- **Step 2:** The cross-entropy loss measures the difference between the predicted probabilities and the actual labels.
- **Step 3:** To minimize the loss, compute the gradients with respect to the weights and bias.
- **Step 4:** Use these gradients to update the weights and bias during training.

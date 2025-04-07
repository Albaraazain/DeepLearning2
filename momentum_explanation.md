## Momentum in Optimization

Momentum in optimization is like giving your learning process some "inertia." Instead of just following the current gradient (which tells you the steepest direction to move), momentum lets you combine it with the previous updates. This helps smooth the path to the minimum, prevents you from getting stuck in small local bumps, and speeds up convergence.

### Why Use Momentum?
Momentum is used in optimization to:
1. **Speed Up Convergence:** It helps the model converge faster by combining the current gradient with the previous updates, allowing it to build up speed in the right direction.
2. **Reduce Oscillations:** It smooths out the updates, reducing the zigzagging effect, especially in regions with steep gradients.
3. **Escape Local Minima:** It helps the optimizer to escape small local minima by maintaining a consistent direction, thus avoiding getting stuck.

In summary, momentum improves the efficiency and stability of the training process.

## Different Loss Functions

Loss functions are measures of how far off a prediction is from the actual outcome. Here are some common loss functions:
- **Mean Squared Error (MSE):** Squares the errors so that larger mistakes are penalized more heavily.
- **Mean Absolute Error (MAE):** Takes the absolute value of errors, treating all deviations equally.
- **Cross-Entropy Loss:** Evaluates the difference between predicted probabilities and actual outcomes, making it ideal for classification tasks.

Each loss function influences how the model learns by affecting the magnitude and direction of updates during training.

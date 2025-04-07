# Solution to Question 6: Softmax and Temperature Scaling in Multi-Class Classification

## 1. Mathematical Formulation of Softmax

The softmax function converts raw logits (model outputs) into a probability distribution over \( C \) classes. For a given logit vector \( z \) of length \( C \), the softmax function is defined as:
\[
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
\]
where \( \sigma(z_i) \) is the probability of class \( i \).

## 2. Numerical Stability in Softmax Computation

Direct computation of the softmax function can lead to numerical instability, especially when the logits have large values. To improve numerical stability, we can subtract the maximum logit from each logit before applying the softmax function:
\[
\sigma(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{C} e^{z_j - \max(z)}}
\]
This transformation does not change the output probabilities but prevents overflow and underflow issues.

## 3. Temperature Scaling

Temperature scaling modifies the softmax function by introducing a temperature parameter \( T \):
\[
\sigma(z_i, T) = \frac{e^{z_i / T}}{\sum_{j=1}^{C} e^{z_j / T}}
\]
- **High Temperature (T > 1)**: Produces a softer probability distribution, making the model less confident in its predictions.
- **Low Temperature (T < 1)**: Produces a sharper probability distribution, making the model more confident in its predictions.

### Practical Implications

- **Training**: Temperature scaling can be used to control the confidence of the model during training, potentially improving generalization.
- **Inference**: Adjusting the temperature can help in scenarios where calibrated probabilities are needed, such as in uncertainty estimation or when combining predictions from multiple models.

## 4. Advantages & Limitations

### Softmax Function
**Advantages**:
- Converts logits into a probability distribution.
- Ensures that the sum of probabilities is 1.

**Limitations**:
- Sensitive to numerical instability without proper scaling.
- Can produce overconfident predictions.

### Temperature Scaling
**Advantages**:
- Provides control over the confidence of predictions.
- Useful for model calibration and uncertainty estimation.

**Limitations**:
- Requires careful tuning of the temperature parameter.
- May not always improve model performance.

## 5. Practical Considerations

**When to Use Softmax**:
- Standard choice for multi-class classification tasks.
- When converting logits to probabilities is required.

**When to Use Temperature Scaling**:
- When model calibration is important.
- To adjust the confidence of predictions during training or inference.

**Hybrid Approach**:
- Use softmax for initial training.
- Apply temperature scaling during post-processing to fine-tune model confidence.

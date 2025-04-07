# Solution to Question 10: Cross Entropy Loss and Softmax

## 1. Softmax Activation Function

**Definition**:
```math
p_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}
```
Where:
- z_i: Logit for class i
- C: Total number of classes
- p_i: Class probability

**Properties**:
- Outputs probability distribution (∑p_i = 1)
- Magnifies differences between logits
- Preserves ordinal relationships

## 2. Cross Entropy Loss Derivation

**General Form**:
```math
L = -\sum_{c=1}^C y_c \log(p_c)
```

**One-Hot Encoding Simplification**:
For true class k:
```math
y_c = \begin{cases}
1 & c = k \\
0 & \text{otherwise}
\end{cases}
```
Simplified loss:
```math
L = -\log(p_k)
```

**Numerical Example**:
```python
logits = [2.0, 1.0, 0.1]
probs = [0.659, 0.242, 0.099]  # After softmax
true_class = 0
loss = -np.log(probs[0])  # ≈ 0.417
```

## 3. Combined Softmax + Cross Entropy

**Numerical Stability Trick**:
```math
L = -\left(z_k - \log\sum_{j=1}^C e^{z_j}\right)
```

**Implementation Advantage**:
- Avoids intermediate exponentiation of large values
- Prevents underflow/overflow
- More precise gradient calculations

## 4. Stability Analysis

**Problem Cases**:
| Scenario | Raw Softmax Issue | Combined Solution |
|----------|-------------------|-------------------|
| Large z_i | Exponent overflow | Subtract max(z) |
| Small z_i | Exponent underflow | Stable log-sum-exp |

**Stable Implementation**:
```python
def cross_entropy(logits, true_class):
    shifted = logits - np.max(logits)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    return -shifted[true_class] + log_sum_exp
```

## 5. Gradient Computation

**Derivative**:
```math
\frac{∂L}{∂z_i} = \begin{cases}
p_i - 1 & i = k \\
p_i & i ≠ k
\end{cases}
```

**Backprop Advantage**:
- Simple, efficient gradient calculation
- Direct relationship between probabilities and errors
- No unstable intermediate steps

## 6. Practical Considerations

**Implementation Checklist**:
1. Always use combined softmax + cross entropy
2. Apply log-sum-exp trick
3. Verify gradient magnitudes
4. Monitor numerical stability
5. Use double precision if needed

**Framework Usage**:
```python
# TensorFlow/Keras
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# PyTorch
loss = nn.CrossEntropyLoss()  # Logits input expected
```

## 7. Error Analysis Table

| Case | Probability p_k | Loss Value | Gradient Magnitude |
|------|-----------------|------------|--------------------|
| Confident Correct | 0.99 | 0.01 | Small |
| Uncertain | 0.5 | 0.69 | Moderate |
| Wrong Prediction | 0.01 | 4.6 | Large |

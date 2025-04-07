# Solution to Question 11: Cross Entropy, KL Divergence, and MLE

## 1. Cross Entropy and Maximum Likelihood

### 1.1 Mathematical Relationship
**Likelihood Function**:
```math
L(θ) = \prod_{i=1}^N p(y_i|x_i; θ)
```

**Negative Log-Likelihood**:
```math
-\log L(θ) = -\sum_{i=1}^N \log p(y_i|x_i; θ)
```

**Cross Entropy**:
```math
H(p,q) = -\sum_c p(c) \log q(c)
```

**Equivalence**:
```math
\text{argmin}_θ H(p_{data}, q_θ) = \text{argmax}_θ L(θ)
```

## 2. KL Divergence Analysis

### 2.1 Definition and Asymmetry
**KL Divergence**:
```math
D_{KL}(p \| q) = \sum_c p(c) \log\frac{p(c)}{q(c)} = H(p,q) - H(p)
```

**Asymmetry Example**:
```python
p = [0.8, 0.2]
q = [0.9, 0.1]

D_KL_pq = 0.8*np.log(0.8/0.9) + 0.2*np.log(0.2/0.1) ≈ 0.023
D_KL_qp = 0.9*np.log(0.9/0.8) + 0.1*np.log(0.1/0.2) ≈ 0.041
```

### 2.2 Divergence Comparison
| Property       | KL Divergence | JS Divergence | Wasserstein |
|----------------|---------------|---------------|-------------|
| Symmetric      | No            | Yes           | Yes         |
| Triangle Ineq  | No            | No            | Yes         |
| Handling Zero  | Undefined     | Stable        | Stable      |

## 3. Training Implications

### 3.1 Distribution Mismatch Effects
1. **Overconfidence**:
   - q(c) → 1 for wrong classes
   - Large gradients → unstable training

2. **Calibration Issues**:
   - Confident wrong predictions
   - Poor uncertainty estimation

3. **Adversarial Vulnerability**:
   - Small input changes → large output changes

### 3.2 Mitigation Strategies
1. **Label Smoothing**:
   ```math
   p_{smooth}(c) = (1-ε)p(c) + ε/C
   ```
2. **Temperature Scaling**:
   ```math
   q_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
   ```
3. **Regularization**:
   - Add KL term to loss: 
   ```math
   L_{total} = H(p,q) + λD_{KL}(q\|p_{prior})
   ```

## 4. Practical Implementation

### 4.1 KL Computation
```python
def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

# Example usage
p = np.array([0.8, 0.2])
q = np.array([0.7, 0.3])
print(kl_divergence(p, q))  # ≈ 0.023
```

### 4.2 Label Smoothing
```python
def smooth_labels(y, epsilon=0.1):
    K = y.shape[1]
    return y * (1 - epsilon) + epsilon/K

# Usage in Keras
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1))
```

## 5. Theoretical Insights

### 5.1 Information Geometry
**Manifold Perspective**:
- KL divergence measures "distance" on probability simplex
- Cross entropy = Entropy + KL divergence

**Fisher Information**:
```math
I(θ) = \mathbb{E}[(\frac{∂}{∂θ} \log q_θ)^2]
```
KL divergence approximates Fisher-Rao distance for small perturbations

### 5.2 Bayesian Interpretation
**Maximum a Posteriori**:
```math
θ_{MAP} = \text{argmax}_θ \log p(θ|D) = \text{argmax}_θ [\log p(D|θ) + \log p(θ)]
```
KL divergence naturally appears in variational inference

## 6. Diagnostic Tools

**Calibration Curve**:
```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
```

**Expected Calibration Error**:
```python
def ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_boundaries)
    return np.sum([np.abs(np.mean(y_true[bin]) - np.mean(y_prob[bin]))
                   for bin in bin_indices])

# Solution to Question 12: Regularization Techniques in Deep Learning

## 1. L1 vs L2 Regularization

**Mathematical Formulation**:
```math
L1: \Omega(w) = λ\sum|w_i|
L2: \Omega(w) = \frac{λ}{2}\sum w_i^2
```

**Effects Comparison**:
| Property        | L1 Regularization | L2 Regularization |
|-----------------|-------------------|-------------------|
| Sparsity        | Induces           | No sparsity       |
| Solution uniqueness | Multiple solutions | Unique solution |
| Gradient        | Discontinuous     | Smooth            |
| Feature selection| Yes               | No                |

**Optimization Example**:
```python
# L1 implementation
def l1_loss(params, lambda_):
    return lambda_ * np.sum(np.abs(params))

# L2 implementation  
def l2_loss(params, lambda_):
    return 0.5 * lambda_ * np.sum(params**2)
```

## 2. Dropout Regularization

**Mechanism**:
- Randomly deactivate neurons during training (p=0.5 common)
- Scale activations during inference

**Math Representation**:
```math
h_{drop} = h \odot m, \quad m_i \sim \text{Bernoulli}(p)
```

**Implementation**:
```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape) > self.p
            return x * mask / (1 - self.p)
        return x
```

## 3. Batch Normalization

**Regularization Effects**:
1. Reduces internal covariate shift
2. Adds noise through mini-batch statistics
3. Allows higher learning rates by stabilizing gradients

**Training vs Inference**:
```math
\hat{x} = \frac{x - μ_{batch}}{σ_{batch}} \quad \text{(Training)}
```
```math
\hat{x} = \frac{x - μ_{running}}{σ_{running}} \quad \text{(Inference)}
```

**Implementation Benefits**:
```python
# PyTorch example
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

## 4. Comparative Analysis

**Effectiveness**:
| Technique       | Training Speed | Generalization | Compute Cost |
|-----------------|----------------|----------------|--------------|
| L1              | Moderate       | Feature select | Low          |
| L2              | Fast           | Good           | Low          |
| Dropout         | Slower         | Excellent      | Moderate     |
| Batch Norm       | Faster         | Good           | High         |

## 5. Practical Guidelines

**When to Use**:
- **L1**: Sparse models, feature selection
- **L2**: Default choice, prevents overfitting
- **Dropout**: Large networks, computer vision
- **Batch Norm**: Deep networks, unstable training

**Combination Strategies**:
1. L2 + Dropout (common in CNNs)
2. Batch Norm + L2 (common in ResNets)
3. Layer Norm + Dropout (common in Transformers)

## 6. Advanced Techniques

**Spatial Dropout**:
- Drops entire feature maps in CNNs
```python
nn.Dropout2d(p=0.2)
```

**Stochastic Depth**:
- Randomly drops entire layers during training
```python
def forward(self, x):
    if self.training and torch.rand(1) < 0.5:
        return x  # Skip layer
    return self.layer(x)
```

**Gradient Noise**:
```python
def train_step(x, y):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    
    # Add Gaussian noise to gradients
    for param in model.parameters():
        param.grad += torch.randn_like(param.grad) * 0.01
        
    optimizer.step()

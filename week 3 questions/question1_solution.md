# Solution to Question 1: Discriminative vs. Generative Models

## 1. Fundamental Differences

**Discriminative Models**:
- Learn decision boundaries between classes
- Model conditional probability P(Y|X)
- Focus on separating classes through direct mapping
- Examples: Logistic Regression, SVM, Neural Networks

**Generative Models**:
- Learn joint probability P(X,Y)
- Model data generation process for each class
- Capture underlying data distributions
- Examples: Naive Bayes, Gaussian Mixture Models, Hidden Markov Models

## 2. Classification Mechanism Comparison

| Aspect                | Discriminative                  | Generative                     |
|-----------------------|---------------------------------|--------------------------------|
| Probability Model     | P(Y|X) - Conditional            | P(X,Y) - Joint                |
| Decision Boundary     | Directly learned               | Derived from distributions    |
| Training Objective    | Maximize class separation       | Model data generation         |
| New Sample Handling   | Classify based on boundaries    | Compare likelihood ratios     |

## 3. Out-of-Distribution Performance

**Discriminative Models**:
- Vulnerable to novel patterns
- No inherent mechanism for uncertainty estimation
- May confidently misclassify OOD samples
- Example: Image classifier trained on cats/dogs failing on bird images

**Generative Models**:
- Can detect anomalies through likelihood thresholds
- Estimate P(X) to identify low-probability samples
- Better uncertainty quantification
- Example: Fraud detection systems flagging unusual transaction patterns

## 4. Mathematical Formulations

**Discriminative Approach** (Logistic Regression):
```math
P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta X)}}
```

**Generative Approach** (Naive Bayes):
```math
P(Y=k|X) \propto P(X|Y=k)P(Y=k) = \left(\prod_{i=1}^n P(x_i|Y=k)\right)P(Y=k)
```

## 5. Advantages & Limitations

### Discriminative Models
**Advantages**:
- Higher accuracy when training data is sufficient
- Fewer computational requirements
- Directly optimizes classification objective

**Limitations**:
- Require retraining for new classes
- Poor calibration of uncertainty estimates
- Limited understanding of data structure

### Generative Models
**Advantages**:
- Handle missing data naturally
- Can generate new samples
- Better OOD detection capabilities
- Incorporate domain knowledge through priors

**Limitations**:
- Sensitive to distributional assumptions
- Computationally intensive for complex data
- Potential model mismatch with real data

## 6. Practical Considerations

**When to Choose Discriminative**:
- Primary goal is classification accuracy
- Limited computational resources
- Clear separation between classes exists
- Large labeled datasets available

**When to Choose Generative**:
- Need to understand data generation process
- Require uncertainty quantification
- Dealing with incomplete datasets
- Semi-supervised learning scenarios

**Hybrid Approaches**:
- Generative Discriminative Tradeoff framework
- Deep generative models (VAEs, GANs) with discriminative components
- Bayesian neural networks combining both paradigms

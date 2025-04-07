# Solution to Question 6: Bias-Variance Trade-off and Model Selection

## 1. Bias and Variance Definitions

**Bias**:
- Systematic error in model predictions
- Difference between expected predictions and true values
- Indicates model's ability to capture underlying patterns
```math
\text{Bias}[f̂(x)] = \mathbb{E}[f̂(x)] - f(x)
```

**Variance**:
- Model's sensitivity to fluctuations in training data
- Spread of predictions across different training sets
- Measures prediction consistency
```math
\text{Var}[f̂(x)] = \mathbb{E}[(f̂(x) - \mathbb{E}[f̂(x)])^2]
```

## 2. Bias-Variance Decomposition

**Total Error Breakdown**:
```math
\mathbb{E}[(y - f̂(x))^2] = \underbrace{(\mathbb{E}[f̂(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(f̂(x) - \mathbb{E}[f̂(x)])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}
```

**Model Complexity Relationship**:
| Complexity | Bias    | Variance | Total Error |
|------------|---------|----------|-------------|
| Low        | High ↑  | Low ↓    | High        |
| Medium     | Medium  | Medium   | Optimal     |
| High       | Low ↓   | High ↑   | High        |

## 3. Overfitting vs. Underfitting

### 3.1 Underfitting (High Bias)
**Characteristics**:
- Model too simple to capture patterns
- Poor performance on both training and test data
- High training error
- Insufficient model capacity

**Example**:
- Linear model for non-linear data
```python
# Linear model trying to fit quadratic data
y = wx + b  # Underfits when true relationship is y = ax² + bx + c
```

### 3.2 Overfitting (High Variance)
**Characteristics**:
- Model memorizes training data
- Excellent training performance
- Poor generalization
- Captures noise in training data

**Example**:
- High-degree polynomial fitting noise
```python
# Overly complex polynomial
degree = 15  # Too high for simple underlying pattern
model = np.polyfit(x, y, degree)
```

## 4. Model Selection Techniques

### 4.1 Data Splitting
**Standard Split**:
- Training (60-80%): Model learning
- Validation (10-20%): Hyperparameter tuning
- Test (10-20%): Final evaluation

**Implementation**:
```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)

# Second split: create validation set
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
```

### 4.2 Cross-Validation
**K-Fold Process**:
1. Split data into K folds
2. Train on K-1 folds
3. Validate on remaining fold
4. Rotate validation fold
5. Average performance

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
scores = []

for train_idx, val_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[val_idx], y[val_idx])
    scores.append(score)

avg_score = np.mean(scores)
```

## 5. Hyperparameter Tuning

### 5.1 Grid Search
**Systematic Approach**:
```python
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 5.2 Random Search
**Efficient Alternative**:
```python
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=100)
```

## 6. Practical Guidelines

**Model Selection Process**:
1. Start simple (establish baseline)
2. Gradually increase complexity
3. Monitor validation performance
4. Use early stopping when applicable
5. Ensemble methods for robustness

**Diagnostic Tools**:
1. Learning curves (train vs. validation error)
2. Validation curves (performance vs. hyperparameter)
3. Cross-validation scores distribution
4. Residual analysis

**Warning Signs**:
- Large gap between train/validation performance → Overfitting
- Both errors high → Underfitting
- Unstable cross-validation scores → High variance
- Consistent poor performance → High bias

## 7. Advanced Considerations

**Nested Cross-Validation**:
- Outer loop: Performance estimation
- Inner loop: Model selection
- Unbiased performance estimate

**Bayesian Optimization**:
- Probabilistic model of objective
- Efficient hyperparameter search
- Particularly useful for expensive models

**Ensemble Methods**:
- Reduce variance through averaging
- Bagging: Random subsets
- Boosting: Sequential learning
- Stacking: Meta-models

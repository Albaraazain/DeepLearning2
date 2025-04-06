## Derivation of $\frac{\partial E}{\partial w_2}$

### Introduction

In backpropagation, understanding how each weight affects the loss function $E$ is crucial. This document explains the derivation of $\frac{\partial E}{\partial w_2}$ in a simple neural network.

### Network Architecture

- Input layer: $x_1$, $x_2$
- Hidden layer with sigmoid activation
- Output layer with linear activation

### Forward Pass

1. Hidden layer input: $z = w_1x_1 + w_2x_2$
2. Hidden layer output: $h = \sigma(z) = \frac{1}{1 + e^{-z}}$
3. Output: $\hat{y} = vh$

### Loss Function

$E = \frac{1}{2}(\hat{y} - y)^2$

### Chain Rule for Backpropagation

$\frac{\partial E}{\partial w_2} = \frac{\partial E}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial h} \times \frac{\partial h}{\partial z} \times \frac{\partial z}{\partial w_2}$

### Step-by-Step Derivation

1. $\frac{\partial E}{\partial \hat{y}} = \hat{y} - y$
2. $\frac{\partial \hat{y}}{\partial h} = v$
3. $\frac{\partial h}{\partial z} = h(1-h)$ (sigmoid derivative)
4. $\frac{\partial z}{\partial w_2} = x_2$

## Combining the Terms

$\frac{\partial E}{\partial w_2} = (\hat{y} - y) \times v \times h(1-h) \times x_2$

## Example Calculation

Given:
- $\hat{y} \approx 0.534$
- $y = 0.7$
- $v = 0.8$
- $h \approx 0.668$
- $x_2 = 1$

$\frac{\partial E}{\partial w_2} = (0.534 - 0.7) \times 0.8 \times 0.668 \times (1 - 0.668) \times 1$

$\frac{\partial E}{\partial w_2} \approx -0.029$

## Key Points

1. The derivative $\frac{\partial E}{\partial w_2}$ depends on the error, the weights, and the inputs.
2. The sigmoid derivative $h(1-h)$ plays a critical role.
3. This process helps in updating $w_2$ during training.

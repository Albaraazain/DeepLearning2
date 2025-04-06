# Backpropagation Question2 Solution

## Step1: Understand the Problem

Consider a neural network for binary classification with:
- Input layer ($x_1, x_2, x_3$)
- Hidden layer with two neurons ($h_1, h_2$) using tanh activation
- Output layer with one neuron using sigmoid activation

Given:
- Weight matrix $W$ between input and hidden layer:
\[
W =
\begin{bmatrix}
0.2 &0.4 & -0.1 \\
-0.3 &0.1 &0.5
\end{bmatrix}
\]
- Weight vector $v$ between hidden and output: $v = [0.6, -0.4]$
- Input values: $x_1 =1$, $x_2 = -1$, $x_3 =0.5$
- Target output: $y =1$
- $\tanh(x) = \frac{e^x - e^(-x)}{e^x + e^(-x)}$
- $\text{sigmoid}(x) = \frac{1}{1 + e^(-x)}$

### Step2: Forward Pass

#### Hidden Layer
Given:
- $x_1 =1$, $x_2 = -1$, $x_3 =0.5$

##### Hidden Layer Inputs
\[
z_{h1} =0.2*1 +0.4*(-1) + (-0.1)*0.5 =0.2 -0.4 -0.05 = -0.25
\]
\[
z_{h2} = -0.3*1 +0.1*(-1) +0.5*0.5 = -0.3 -0.1 +0.25 = -0.15
\]

##### Hidden Layer Outputs
\[
h_1 = \tanh(-0.25) \approx -0.2447
\]
\[
h_2 = \tanh(-0.15) \approx -0.1483
\]

#### Output Layer
##### Output Layer Input
\[
z_o =0.6*(-0.2447) -0.4*(-0.1483) \approx -0.1468 +0.0593 = -0.0875
\]

##### Output Layer Output
\[
\hat{y} = \text{sigmoid}(-0.0875) \approx0.4833
\]

### Step3: Error Calculation

#### Binary Cross-Entropy Loss
\[
E = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
\]
Given $y =1$:
\[
E = -[1 \log(0.4833) +0 \log(1-0.4833)] \approx -0.7343
\]

### Step4: Backward Pass

#### Gradients

1. $\frac{\partial E}{\partial v_1}$ and $\frac{\partial E}{\partial v_2}$

\[
\frac{\partial E}{\partial v_1} = \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_o} \cdot \frac{\partial z_o}{\partial v_1}
\]

\[
\frac{\partial E}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} (-\log(\hat{y})) = -\frac{1}{\hat{y}}
\]

\[
\frac{\partial \hat{y}}{\partial z_o} = \hat{y}(1-\hat{y})
\]

\[
\frac{\partial z_o}{\partial v_1} = h_1
\]

2. Similarly, derive $\frac{\partial E}{\partial W_{ij}}$

### Step5: Numerical Values

### Step6: Vanishing Gradient Problem

The vanishing gradient problem occurs when gradients become very small, causing the weights to update minimally. This is common in deep networks with sigmoid or tanh activations.

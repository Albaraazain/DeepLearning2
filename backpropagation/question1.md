# Backpropagation Question1 (Basic Level)

Consider a simple neural network with two layers:
- Input layer ($x_1$, $x_2$)
- Hidden layer with one neuron using sigmoid activation
- Output layer with one neuron using linear activation

Given:
- Weights: $w_1 =0.5$, $w_2 = -0.3$ (input to hidden)
- Weight: $v =0.8$ (hidden to output)
- Input values: $x_1 =2$, $x_2 =1$
- Target output: $y =0.7$
- Sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$

Tasks:
1. Calculate the forward pass through the network
2. Calculate the gradients using backpropagation:
 - $\frac{\partial E}{\partial v}$
 - $\frac{\partial E}{\partial w_1}$
 - $\frac{\partial E}{\partial w_2}$
3. If learning rate $\eta =0.1$, calculate the updated weights

Note: Use mean squared error loss function $E = \frac{1}{2}(\hat{y} - y)^2$

Show all your steps clearly, including intermediate calculations.

## Step-by-Step Solution with Explanations

###1. Network Structure Visualization
Input Layer $\longrightarrow$ Hidden Layer $\longrightarrow$ Output Layer $x_1 =2$ ----$\longrightarrow$ $w_1 =0.5$ $\longrightarrow$ $h$ --$v =0.8$--> $\hat{y}$ $w_2 = -0.3$ $\longrightarrow$ $x_2 =1$ ----$\longrightarrow$

###2. Forward Pass (Let's break it down!)

#### A. Calculate Hidden Layer Input ($z$)
First, we need to calculate the input to the hidden layer, denoted as $z$. The formula for $z$ is:
$z = w_1x_1 + w_2x_2$

Given that $w_1 =0.5$, $w_2 = -0.3$, $x_1 =2$, and $x_2 =1$, we substitute these values into the equation:
$z = (0.5 \times2) + (-0.3 \times1)$
$z =1.0 -0.3$
$z =0.7$

#### B. Apply Sigmoid Activation to Hidden Layer
Next, we apply the sigmoid activation function to $z$ to get the hidden layer output $h$. The sigmoid function is given by:
$\sigma(x) = \frac{1}{1 + e^{-x}}$

So,
$h = \sigma(z) = \frac{1}{1 + e^{-0.7}}$
$h \approx0.668$ (rounded to3 decimals)

Key Point: The sigmoid function squishes any input into a value between0 and1.

#### C. Calculate Output
Now, we calculate the output $\hat{y}$ using the weight $v$ and the hidden layer output $h$:
$\hat{y} = vh$
$\hat{y} =0.8 \times0.668$
$\hat{y} \approx0.534$

###3. Backward Pass (Chain Rule is our friend!)

#### A. Calculate Error
We start by calculating the error $E$ using the mean squared error formula:
$E = \frac{1}{2}(\hat{y} - y)^2$
$E = \frac{1}{2}(0.534 -0.7)^2$
$E \approx0.0137$

#### B. Calculate Gradients

1. Output Layer ($\frac{\partial E}{\partial v}$):
$\frac{\partial E}{\partial v} = (\hat{y} - y) \times h$

## Calculating $\frac{\partial E}{\partial v}$

To calculate $\frac{\partial E}{\partial v}$, we start with the mean squared error (MSE) formula:
$E = \frac{1}{2}(\hat{y} - y)^2$

Given that $\hat{y} = vh$, we substitute $\hat{y}$:
$E = \frac{1}{2}(vh - y)^2$

Now, we'll find the partial derivative of $E$ with respect to $v$ using the chain rule:
$\frac{\partial E}{\partial v} = \frac{\partial}{\partial v} \left( \frac{1}{2}(vh - y)^2 \right)$

Applying the chain rule:
$\frac{\partial E}{\partial v} = (vh - y) \times \frac{\partial}{\partial v}(vh)$

Since $\frac{\partial}{\partial v}(vh) = h$:
$\frac{\partial E}{\partial v} = (vh - y) \times h$

Substituting $\hat{y} = vh$:
$\frac{\partial E}{\partial v} = (\hat{y} - y) \times h$

Given $\hat{y} \approx0.534$, $y =0.7$, and $h \approx0.668$:
$\frac{\partial E}{\partial v} = (0.534 -0.7) \times0.668$
$\frac{\partial E}{\partial v} = -0.166 \times0.668$
$\frac{\partial E}{\partial v} \approx -0.111$

## Explanation:

1. We started with the MSE formula and substituted $\hat{y} = vh$.
2. Applied the chain rule to differentiate $E$ with respect to $v$.
3. Simplified to $\frac{\partial E}{\partial v} = (\hat{y} - y) \times h$.
4. Calculated the numerical value using given values.

## Calculating $\frac{\partial E}{\partial w_1}$

To find $\frac{\partial E}{\partial w_1}$, we use the chain rule:
$\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial h} \times \frac{\partial h}{\partial z} \times \frac{\partial z}{\partial w_1}$

1. $\frac{\partial E}{\partial \hat{y}} = \hat{y} - y =0.534 -0.7 = -0.166$
2. $\frac{\partial \hat{y}}{\partial h} = v =0.8$
3. $\frac{\partial h}{\partial z} = h(1-h) =0.668(1-0.668) =0.668 \times0.332 =0.222$
4. $\frac{\partial z}{\partial w_1} = x_1 =2$

## Putting it all together:
$\frac{\partial E}{\partial w_1} = -0.166 \times0.8 \times0.222 \times2 = -0.059$

## Calculating $\frac{\partial E}{\partial w_2}$

Similarly, for $\frac{\partial E}{\partial w_2}$:
$\frac{\partial E}{\partial w_2} = \frac{\partial E}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial h} \times \frac{\partial h}{\partial z} \times \frac{\partial z}{\partial w_2}$

1. $\frac{\partial E}{\partial \hat{y}} = \hat{y} - y =0.534 -0.7 = -0.166$
2. $\frac{\partial \hat{y}}{\partial h} = v =0.8$
3. $\frac{\partial h}{\partial z} = h(1-h) =0.668(1-0.668) =0.668 \times0.332 =0.222$
4. $\frac{\partial z}{\partial w_2} = x_2 =1$

## Putting it all together:
$\frac{\partial E}{\partial w_2} = -0.166 \times0.8 \times0.222 \times1 = -0.029$

###4. Update Weights ($\eta =0.1$)

New weights = Old weights - $\eta \times$ gradient

$v_{new} =0.8 - (0.1 \times -0.111) =0.811$
$w_{1_{new}} =0.5 - (0.1 \times -0.059) =0.506$
$w_{2_{new}} = -0.3 - (0.1 \times -0.029) = -0.297$

## Key Points to Remember:
1. Forward pass: Just multiply and apply activations.
2. Backward pass: Use chain rule, working backwards.
3. The sigmoid derivative h(1-h) is a key formula.
4. Negative gradients mean we increase weights.
5. Positive gradients mean we decrease weights.

## Quick Check:
- If output is too low ($\hat{y} < y$): Gradients will be negative, so weights increase.
- If output is too high ($\hat{y} > y$): Gradients will be positive, so weights decrease.

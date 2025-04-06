Exam Question 3: Comparing Backpropagation and Numerical Differentiation
Discuss the differences between backpropagation and numerical differentiation for computing gradients in deep neural networks. Your answer should cover:

An explanation of how numerical differentiation estimates gradients by perturbing each parameter and why this method becomes computationally prohibitive as the number of parameters increases (e.g., quadratic complexity with respect to the number of parameters).

A description of how backpropagation leverages the chain rule to compute gradients in a single backward pass, resulting in linear complexity with respect to the number of parameters.

The practical implications of these differences in the context of training deep networks.
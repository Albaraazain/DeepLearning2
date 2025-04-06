# Neural Network Implementation Exercises

## Basic Implementation
1. Experiment with weights initialization:
   - What happens if weights are initialized to zero?
   - Test different initialization methods
   - Analyze the impact on model performance

## Architecture Design
1. Experiment with hidden layers:
   - Add different numbers of hidden layers
   - Modify learning rates accordingly
   - Determine which setting works best

2. Activation Functions:
   - Try different activation functions
   - Compute the derivative of the pReLU activation function
   - Prove that tanh(x) + 1 = 2sigmoid(2x)
   - Show that an MLP using only ReLU/pReLU constructs a continuous piecewise linear function

## Dropout and Regularization
1. Experiment with dropout probabilities:
   - Change dropout probabilities for first and second layers
   - Switch probabilities between layers
   - Design quantitative experiments
   - Document qualitative takeaways

2. Bias and Gradients:
   - Add bias to hidden layers
   - Consider regularization terms
   - Analyze dimensionality of gradients for n × m matrix inputs

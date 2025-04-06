# Backpropagation Question 3 (Advanced Level)

Consider a recurrent neural network (RNN) processing a sequence of 3 timesteps:
- Input dimensions: 2 features per timestep
- Hidden state dimension: 3
- Output dimension: 1 per timestep
- Using tanh activation for hidden states
- Using linear activation for outputs

Given:
- Input sequence X = [(1.0, -0.5), (0.2, 0.8), (-0.3, 0.4)]
- Initial hidden state h₀ = [0, 0, 0]
- Target sequence y = [0.5, -0.2, 0.7]

Parameters:
- Input-to-hidden weight matrix U:
  [0.1  0.2]
  [-0.1 0.3]
  [0.2  0.1]
- Hidden-to-hidden weight matrix W:
  [0.2  -0.1  0.3]
  [0.1   0.3  -0.2]
  [-0.2  0.1  0.4]
- Hidden-to-output weight vector v:
  [0.5, -0.3, 0.2]

Tasks:
1. Write out the equations for forward propagation through time
2. Derive the expressions for backpropagation through time (BPTT):
   - ∂E/∂v
   - ∂E/∂W
   - ∂E/∂U
3. Calculate gradients for the last timestep
4. Explain how the following affect training:
   - Vanishing gradients
   - Exploding gradients
   - Long-term dependencies

Note: Use mean squared error loss for each timestep: E = ½Σₜ(yₜ - ŷₜ)²

Provide detailed mathematical derivations and explain how the gradients flow backward through time.

# Deep Learning Exam: Mathematical Calculation Questions

Below are additional exam-style questions that incorporate mathematical calculations and derivations. These questions are designed to test your understanding of the mathematical foundations underlying deep learning algorithms.

---

## Question 1: Derivation of Backpropagation Gradient for a 2-Layer Network

**Problem:**  
Consider a simple neural network with one hidden layer. The hidden layer uses the ReLU activation function, and the network employs Mean Squared Error (MSE) as its loss function.  
- **Task:** Derive the gradient update for the weights connecting the hidden layer to the output layer.  
- **Instructions:**  
  1. Clearly express the forward pass equations.  
  2. Use the chain rule to derive the gradient of the loss with respect to these weights.  
  3. Show all intermediate steps and justify each transition.

---

## Question 2: Exponential Learning Rate Decay Calculation

**Problem:**  
An optimizer uses exponential decay for its learning rate according to the formula:
\[
\eta_t = \eta_0 \cdot e^{-kt}
\]
where:
- \(\eta_0\) is the initial learning rate,
- \(k\) is the decay constant,
- \(t\) represents the epoch number.

**Task:**  
Given that \(\eta_0 = 0.1\), \(k = 0.05\), and \(t = 20\) epochs, calculate the learning rate \(\eta_{20}\) after 20 epochs.  
- **Instructions:**  
  1. Substitute the provided values into the decay formula.
  2. Show step-by-step calculations leading to the final numerical value.
  3. Interpret the result briefly.

---

## Question 3: Gradient Derivation for Softmax Cross-Entropy Loss

**Problem:**  
For a softmax classifier, the cross-entropy loss is commonly used to measure the discrepancy between the predicted probability distribution and the true distribution.  
- **Task:** Derive the gradient of the cross-entropy loss with respect to the input logits of the softmax function.
- **Instructions:**  
  1. Begin with the expressions for the softmax function and cross-entropy loss.
  2. Apply the chain rule to compute the gradient.
  3. Provide the final gradient expression and explain the meaning of each term briefly.

---

Feel free to study these questions and use them as practice for your deep learning exam. They are intended to stimulate critical thinking about the mathematical concepts behind learning algorithms.

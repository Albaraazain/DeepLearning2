## Question 6: Simple Matrix Operations

### Part 1
Multiply a 2×3 matrix by a 3×1 matrix and interpret the result (e.g., how it might represent a layer’s computations).

### Solution
1. Matrix multiplication:
   - A = [[a11, a12, a13], [a21, a22, a23]]
   - B = [[b11], [b21], [b31]]
   - C = A·B = [[a11·b11 + a12·b21 + a13·b31], [a21·b11 + a22·b21 + a23·b31]]

### Explanation
- **Step 1:** Understand that matrix multiplication involves taking the dot product of rows of the first matrix with columns of the second matrix.
- **Step 2:** Compute the resulting matrix by performing the dot products.
- **Step 3:** Interpret the result as the transformation of input vectors by the weight matrix in a neural network layer.

### Part 2
Compute partial derivatives when the weight matrix is used to transform input vectors.

### Solution
1. Given a weight matrix W and input vector x, the output y = W·x.
2. Compute the partial derivatives of the output y with respect to the elements of W.

### Explanation
- **Step 1:** Understand that the output y is a linear transformation of the input vector x by the weight matrix W.
- **Step 2:** Compute the partial derivatives of each element of the output y with respect to each element of the weight matrix W.
- **Step 3:** Use these partial derivatives to update the weights during backpropagation in a neural network.

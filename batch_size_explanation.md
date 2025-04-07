## Batch Size in Training

### Basics
When training a machine learning model, you update the model's parameters (weights) to reduce the error between its predictions and the actual outcomes. This process is called optimization, and it involves calculating gradients (slopes) to adjust the weights in the right direction.

### Types of Batch Sizes
1. **Full-Batch Gradient Descent:**
   - **Batch Size:** All training examples (N)
   - **Variance:** Low (more stable updates)
   - **Compute Efficiency:** Memory intensive (requires a lot of memory to process all data at once)
   - **Explanation:** You use the entire dataset to compute the gradient and update the weights. This is very accurate but can be slow and requires a lot of memory.

2. **Mini-Batch Gradient Descent:**
   - **Batch Size:** A subset of training examples (e.g., 32-512)
   - **Variance:** Moderate (balance between stability and speed)
   - **Compute Efficiency:** GPU optimized (efficient for parallel processing)
   - **Explanation:** You split the dataset into smaller batches and update the weights after processing each batch. This is a good balance between speed and accuracy and is commonly used in practice.

3. **Stochastic Gradient Descent (SGD):**
   - **Batch Size:** One training example (1)
   - **Variance:** High (more noisy updates)
   - **Compute Efficiency:** CPU efficient (less memory required)
   - **Explanation:** You update the weights after processing each individual training example. This is very fast and requires less memory but can be noisy and less stable.

### Why Different Kinds?
- **Memory Constraints:** Full-batch requires a lot of memory, which may not be feasible for large datasets.
- **Speed:** Mini-batch and SGD can be faster because they update weights more frequently.
- **Stability:** Full-batch is more stable but slower, while SGD is faster but noisier. Mini-batch offers a good compromise.

### Summary
- **Full-Batch:** Accurate but slow and memory-intensive.
- **Mini-Batch:** Balanced approach, commonly used.
- **SGD:** Fast and memory-efficient but noisy.

Choosing the right batch size depends on your specific needs and constraints, such as available memory and desired training speed.

## Batch vs. Epoch

### Batch
- A batch is a subset of the training data.
- The model's weights are updated after processing each batch.
- Example: If you have 1000 training examples and a batch size of 100, you will have 10 batches.

### Epoch
- An epoch is one complete pass through the entire training dataset.
- During one epoch, the model sees every training example once.
- Example: If you have 1000 training examples and a batch size of 100, one epoch consists of 10 batches.

### Summary
- **Batch:** A small group of training examples used to update the model's weights.
- **Epoch:** One full pass through the entire training dataset, consisting of multiple batches.

Understanding the difference helps in configuring the training process for better performance and efficiency.

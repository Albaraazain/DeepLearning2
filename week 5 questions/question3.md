# Exam Question 3: Learning Rate Scheduling and Warmup Strategies

Learning rate is a fundamental hyperparameter in optimization. Address the following points:

- **Learning Rate Scheduling:**  
  Different strategies for adjusting the learning rate throughout training, including:
  - **Step Decay:** Reduces the learning rate by a fixed factor after a set number of epochs.
  - **Exponential Decay:** Continuously decreases the learning rate exponentially over time.
  - **Cosine Schedules:** Varies the learning rate following a cosine function, often allowing for rapid drops and occasional restarts.

- **Warmup Phase:**  
  The warmup phase involves starting with a lower learning rate for a few initial iterations or epochs. This helps:
  - Stabilize training when the network weights are near their random initialization.
  - Prevent large weight updates that can destabilize early learning.
  - Provide a smoother transition to the target learning rate, reducing abrupt changes.

- **Effect of Batch Size on Learning Rate:**  
  When the batch size is altered, the variance of the gradient estimation changes. Larger batch sizes typically produce more accurate gradients, which might allow for using a higher learning rate. Conversely, smaller batch sizes with noisier gradients often require a reduced learning rate to maintain stable convergence. Rescaling the learning rate accordingly ensures consistent training behavior regardless of the batch size.

---

**Solution:**

1. **Learning Rate Scheduling Strategies:**  
   Scheduling the learning rate over training epochs can improve convergence and overall performance.  
   - **Step Decay:** The learning rate is reduced at fixed intervals. For instance, reducing the rate by half every 10 epochs allows for larger updates initially and finer adjustments later.  
   - **Exponential Decay:** The learning rate decreases continuously, following an exponential function. This smooth reduction helps in gradually refining the weights.  
   - **Cosine Schedules:** The learning rate follows a cosine curve, often with periodic restarts (cosine annealing with restarts), which can help escape local minima and promote better exploration during training.  

2. **Warmup Phase:**  
   Initiating training with a small learning rate helps mitigate the instability associated with random weight initialization. Neural networks are particularly sensitive in the initial phase of training, when drastic weight changes could lead to divergence. By gradually increasing the learning rate (warmup), the optimizer is given time to adapt and stabilize the learning process.

3. **Batch Size and Learning Rate Rescaling:**  
   The batch size has a direct impact on gradient estimation:
   - **Larger Batch Size:** Produces more reliable gradient estimates. In this case, a higher learning rate might be feasible as the updates are more stable.
   - **Smaller Batch Size:** Introduces extra noise into the gradient, which may require a lower learning rate to prevent unstable updates.
   
   In practice, if the batch size changes from one experiment to another, the learning rate is often rescaled proportionally to maintain similar convergence behavior (e.g., scaling linearly with the batch size under certain conditions).

---

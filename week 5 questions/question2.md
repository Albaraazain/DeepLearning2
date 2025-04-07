# Exam Question 2: Batch Size, Gradient Noise, and Local Minima

Discuss how the choice of batch size influences the optimization process in deep learning. Your response should cover:
- The definitions of an iteration and an epoch in the context of mini-batch gradient descent, and how the batch size affects the number of iterations per epoch.
- How larger batch sizes yield more reliable (less noisy) gradient estimates, while smaller batch sizes introduce noise that might help the optimizer escape poor local minima.
- The trade-offs involved in selecting batch size with respect to convergence speed and generalization performance.

---

**Solution:**

1. **Iterations, Epochs, and Batch Size:**  
   In mini-batch gradient descent, an **iteration** refers to one update of the model parameters using a subset (batch) of the training data. An **epoch** is a full pass through the entire training dataset. Thus, the number of iterations per epoch is equal to the total number of training examples divided by the batch size. A smaller batch size increases the number of iterations per epoch while a larger batch size decreases it.

2. **Gradient Noise and Reliability:**  
   - **Larger Batch Sizes:** Provide gradient estimates that are closer to the true gradient computed over the entire dataset. This results in more stable and reliable updates during training. However, it might also cause the optimizer to converge to sharp minima that may not generalize well.
   - **Smaller Batch Sizes:** Introduce more stochasticity (noise) into the gradient estimates because only a small subset of data is used for each update. This noise can help the optimizer jump out of poor local minima and possibly find flatter minima, which often generalize better. However, too much noise might hinder the convergence speed.

3. **Trade-offs in Batch Size Selection:**  
   - **Convergence Speed:** Larger batches can speed up convergence per epoch since the gradient estimates are more accurate, although each iteration may be computationally heavier. On the other hand, smaller batches may require more iterations to converge.
   - **Generalization Performance:** The additional noise from smaller batches can act as a form of regularization, potentially leading to better generalization on unseen data. Conversely, while larger batches may converge faster, they might overfit if the model settles into sharp minima.
   - **Computational Considerations:** The batch size also affects memory usage and parallelism. Very large batches may stress available hardware resources, whereas very small batches may not fully utilize the hardware throughput.

---

Through appropriate selection of batch size, one seeks a balance between stable gradient estimates and beneficial stochastic behavior, ultimately aiming to improve convergence speed and generalization performance.

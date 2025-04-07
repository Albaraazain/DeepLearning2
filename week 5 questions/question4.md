# Exam Question 4: Adaptive Learning Rate Methods

Many modern optimizers adapt the learning rate on a per-parameter basis. Describe adaptive learning rate methods such as AdaGrad, RMSProp, and Adam by explaining:

- How these methods accumulate (or exponentially average) gradient information to adjust individual parameter updates.
- The motivation behind dividing the global learning rate by the square root of accumulated squared gradients, and the challenges that arise if early gradients overly influence later updates.
- How momentum terms are incorporated into these methods to further improve convergence stability and speed.

---

**Solution:**

1. **Accumulation of Gradient Information:**  
   - **AdaGrad:** Collects the sum of squared gradients for each parameter. The learning rate for each parameter is adjusted by dividing the global learning rate by the square root of this accumulated sum (plus a small constant to prevent division by zero). This means that parameters that have received large updates in the past will have their effective learning rate reduced.  
   - **RMSProp:** Instead of a cumulative sum, RMSProp uses an exponential moving average of the squared gradients. This allows it to “forget” old gradients and adapt more quickly to recent changes in the gradient, making it more suited for non-stationary objectives.  
   - **Adam:** Combines ideas from RMSProp and momentum. Adam maintains both an exponential moving average of past gradients (first moment) and squared gradients (second moment). The first moment acts similarly to a momentum term, while the second moment scales each parameter’s learning rate, allowing adaptive adjustments.

2. **Dividing by the Square Root of Accumulated Squared Gradients:**  
   - The division by the square root of the accumulated (or averaged) squared gradients serves to normalize the update step for each parameter. This prevents parameters that have consistently large gradients from experiencing overly large updates and helps stabilize training.  
   - However, one challenge is that if early gradients are very large, they can dominate the accumulation and result in a very small effective learning rate later on, potentially slowing down convergence. Methods like RMSProp and Adam counter this by using exponential averaging, which diminishes the influence of outdated gradients.

3. **Incorporating Momentum:**  
   - **Momentum Terms:** In optimizers like Adam, the momentum (first moment) term averages past gradients, which smooths the updates and accelerates convergence by dampening oscillations.  
   - This momentum helps guide the parameters in consistent directions by accumulating information over time, which can speed up the convergence process and contribute to stability, especially in regions with noisy gradients.

---

Overall, these adaptive methods adjust the learning rate on a per-parameter basis by dynamically scaling updates according to historical gradient information, while also incorporating momentum to foster stable and efficient convergence.

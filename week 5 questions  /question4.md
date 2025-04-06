Exam Question 4: Adaptive Learning Rate Methods
Many modern optimizers adapt the learning rate on a per-parameter basis. Describe adaptive learning rate methods such as AdaGrad, RMSProp, and Adam by explaining:

How these methods accumulate (or exponentially average) gradient information to adjust individual parameter updates.

The motivation behind dividing the global learning rate by the square root of accumulated squared gradients, and the challenges that arise if early gradients overly influence later updates.

How momentum terms are incorporated into these methods to further improve convergence stability and speed.
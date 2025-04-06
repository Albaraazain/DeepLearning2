# Model Training Exercises

## Training Process
1. Batch Size and Loss Function:
   - Explain why changing l = loss(output, y) to l = loss(output, y).mean() 
     requires changing trainer.step(batch_size) to trainer.step(1)
   - Review documentation for loss functions in gluon.loss
   - Implement Huber's loss

## Performance and Optimization
1. Data Iterator Performance:
   - Analyze impact of reducing batch_size to 1 on reading performance
   - Evaluate current implementation speed
   - Explore options for improvement
   - Review other available datasets in the framework's API

2. Training Dynamics:
   - Adjust hyperparameters:
     * Batch size
     * Number of epochs
     * Learning rate
   - Analyze impact on results
   - Investigate why test accuracy might decrease after many epochs
   - Propose solutions for accuracy degradation

## Model Selection and Regularization
1. Polynomial Regression:
   - Implement exact solution using linear algebra
   - Plot training loss vs. model complexity
   - Determine minimum polynomial degree for zero training loss
   - Use validation set to find optimal λ value
   - Analyze if the found λ is truly optimal

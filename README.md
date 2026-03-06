# LogisticRegression_SoftmaxRegression_NeuralNets_MachineLearning 
## Handwritten Digit Recognition System ​
The project focuses on building a comprehensive handwritten digit recognition system using the MNIST dataset, progressing from simple linear models to advanced deep neural networks. ​

The assignment is divided into four main parts:
## Part A: Linear Classification Models ​
1. Data Preparation:  
- Downloading and preprocessing the MNIST dataset.   ​
- Normalizing pixel values to the [0,1] range. ​
- Flattening images for linear models (28×28 → 784 features). ​
- Creating stratified splits for training, validation, and testing (60%, 20%, 20%). ​
- Implementing PyTorch DataLoaders for efficient data handling. ​

2. Logistic Regression:
- Binary classification to distinguish between two digits (e.g., 0 vs 1). ​
- Implementing logistic regression from scratch using PyTorch tensors. ​
- Using sigmoid activation, binary cross-entropy loss, and gradient descent optimization. ​

3. Softmax Regression:
- Multi-class classification for all 10 digits. ​
- Implementing softmax regression from scratch using PyTorch. ​
- Comparing results with PyTorch's built-in implementations. ​

## Part B: Neural Network Implementation ​
1. Custom Neural Network Architecture:
- Designing a feedforward neural network with at least two hidden layers. ​
- Using ReLU activation and proper weight initialization (Xavier/He initialization). ​

2. Training Infrastructure:
- Implementing a custom training loop with batch processing, gradient computation, and backpropagation. ​
- Using Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and batch size of 64. ​

3. Performance Visualization:
- Generating plots for training/validation loss and accuracy over epochs. ​
- Analyzing convergence and learning curves. ​

## Part C: Comprehensive Analysis ​
Hyperparameter Analysis:
- Experimenting with different learning rates, batch sizes, and network architectures.
- Analyzing convergence speed, stability, and performance. ​

Model Comparison:
- Comparing logistic regression, softmax regression, and the best neural network model. ​
- Evaluating computational complexity, training time, and performance. ​
- Providing a detailed performance summary table and analyze misclassified examples. ​

​

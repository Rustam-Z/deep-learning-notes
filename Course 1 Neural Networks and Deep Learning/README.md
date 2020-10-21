# Neural Networks and Deep Learning
- Understand the major technology trends driving Deep Learning
- Be able to build, train and apply fully connected deep neural networks 
- Know how to implement efficient (vectorized) neural networks 
- Understand the key parameters in a neural network's architecture 

## Introduction to Deep Learning
> Be able to explain the major trends driving the rise of deep learning, and understand where and how it is applied today.

### Machine Learning vs Deep Learning
- <br><img src="media/MLvsDL.png" width=300>

### What is neural network? 
- [Lecture notes](01_What_is_Neural_Network.pdf) 
- NN is a powerful learning algorithm inspired by how the brain works. 
- <img src="media/what-is-nn.png" width=300>

### Supervised Learning for Neural Networks
- [Lecture notes](02_Supervised_Learning_for_Neural_Network.pdf)
- Supervised - regression and classification problems. Regression problem - predict results within a continuous output, trying to map input variables to some continuous function. Classification problem, we are instead trying to predict results in a discrete output, we are trying to map input variables into discrete categories.  
- Application of supervised learning <br><img src="media/supervised-learning.png" width=300>
- Types of neural networks: **Convolution Neural Network (CNN)** used often for image application and **Recurrent Neural Network (RNN)** used for one-dimensional sequence data such as translating English to Chinses. As for the autonomous driving, it is a hybrid neural network architecture.
- Structured vs unstructured data <br><img src="media/structured-and-unstructured-data.png" width=300>

### Why is Deep Learning taking off?
- [Lecture notes](03_Why_is_Deep_Learning_Taking_Off.pdf) 
- Large amount of data! We see that traditional algorithms reach to a threshold on performance. However, NN always works better with more data. So you can get better performance as long as you collecting more and more data, without changing the algorithm itself.
- <img src="media/dl-taking-off.jpeg" width=300>


## Neural Networks Basics
- LogReg for NN, making prediction, derivative computation, and gradient descent
- Compute LogReg using back prop
- Python, NumPy, implement vectorization

### Binary Classification
- [Lecture notes](04_Binary_Classification.pdf) 

### Logistic Regression
- Predict whether 0 or 1
- [Lecture notes](05_Logistic_Regression.pdf), [YouTube Video Part 1](https://www.youtube.com/watch?v=L_xBe7MbPwk) & [Part 2](https://www.youtube.com/watch?v=uFfsSgQgerw)

### Logistic Regression Cost Function
- [Lecture notes](), [YouTube Video](https://www.youtube.com/watch?v=MztgenIfGgM)
- Loss function `L(y',y) = - (y*log(y') + (1-y)*log(1-y'))`
### Gradient Descent

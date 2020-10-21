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
> LogReg for NN, making prediction, derivative computation, and gradient descent | Compute LogReg using back prop | Python, NumPy, implement vectorization

### Binary Classification
- [Lecture notes](04_Binary_Classification.pdf) 
- Here are some notations:
  - `M is the number of training vectors`
  - `Nx is the size of the input vector`
  - `Ny is the size of the output vector`
  - `X(1) is the first input vector`
  - `Y(1) is the first output vector`
  - `X = [x(1) x(2).. x(M)]`
  - `Y = (y(1) y(2).. y(M))`
### Logistic Regression
- [Lecture notes](05_Logistic_Regression.pdf), [YouTube Video Part 1](https://www.youtube.com/watch?v=L_xBe7MbPwk) & [Part 2](https://www.youtube.com/watch?v=uFfsSgQgerw)
- Predict whether `0 or 1`, classification algorithm of 2 classes
- `y` to be in between `0 and 1` (probability): `y = sigmoid(w^T*x + b)`
- <img src="media/sigmoid.png" width=200>

### Logistic Regression Cost Function
- [Lecture notes](), [YouTube Video](https://www.youtube.com/watch?v=MztgenIfGgM)
- The cost function measures the accuracy of our hypothesis function. Quantifies the error between predicted values and expected values. 
- Now we are able to concretely measure the accuracy of our predictor function (hypothesis) against the correct results we have so that we can predict new results we don't have.
- First loss function would be the square root error:` L(y',y) = 1/2 (y' - y)^2`
  - But we won't use this notation because it leads us to optimization problem which is non convex, means it contains local optimum points.
- This is the function that we will use: `L(y',y) = - (y*log(y') + (1-y)*log(1-y'))`
- To explain the last function lets see:
  - if `y = 1` ==> `L(y',1) = -log(y')` ==> we want y' to be the largest ==> y' biggest value is 1
  - if `y = 0` ==> `L(y',0) = -log(1-y')` ==> we want 1-y' to be the largest ==> y' to be smaller as possible because it can only has 1 value.
- Then the Cost function will be: `J(w,b) = (1/m) * Sum(L(y'[i],y[i]))`
- The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.

### Gradient Descent
- *Gradient Descent* - so we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.
- [YouTube video: What is the Gradient Descent?](https://www.youtube.com/watch?v=gzrQvzYEvYc)
- Gradient Descent basically just does what we were doing by hand — change the theta values, or parameters, bit by bit, until we hopefully arrived at a minimum. I am comparing it with a cost function right now. 
- We want to predict `w` and `b` that minimize the cost function.
- First we initialize `w` and `b` to 0, 0 or initialize them to a random value in the convex function and then try to improve the values the reach minimum value.
- The derivative give us the direction to improve our parameters.
- The actual equations we will implement:
  - `w = w - alpha * d(J(w,b) / dw)` (how much the function slopes in the w direction)
  - `b = b - alpha * d(J(w,b) / db)` (how much the function slopes in the d direction)
  - <img src="media/gradient1.png" width=300> <img src="media/gradient2.png" width=300> 

### Derivatives
- Derivative of a linear line is its slope.
  - ex. `f(a) = 3a` `d(f(a))/d(a) = 3`
  - if `a = 2` then `f(a) = 6`
  - if we move a a little bit `a = 2.001` then `f(a) = 6.003` means that we multiplied the derivative (Slope) to the moved area and added it to the last result.
- To conclude, Derivative is the slope, and this slope is different in different points in the function thats why the derivative is a function.

### Derivatives with a Computation Graph
- Calculus chain rule says: If `x -> y -> z` (x effect y and y effects z) Then `d(z)/d(x) = d(z)/d(y) * d(y)/d(x)`
- Back prop for computing the derivatives
- <img src="media/03.png" width=300>

### Logistic Regression Gradient Descent
- In the video were discussed the derivatives of gradient decent for a **single training example** with two features `x1` and `x2`.
- <img src="media/04.png" width=300>

### Gradient Descent on *m* Examples
- Lets say we have these variables:
  ```
  	X1                  Feature
  	X2                  Feature
  	W1                  Weight of the first feature.
  	W2                  Weight of the second feature.
  	B                   Logistic Regression parameter.
  	M                   Number of training examples
  	Y(i)                Expected output of i
  ```
- So we have:
  <img src="media/09.png">

- Then from right to left we will calculate derivations compared to the result:

  ```
  	d(a)  = d(l)/d(a) = -(y/a) + ((1-y)/(1-a))
  	d(z)  = d(l)/d(z) = a - y
  	d(W1) = X1 * d(z)
  	d(W2) = X2 * d(z)
  	d(B)  = d(z)
  ```

- From the above we can conclude the logistic regression pseudo code:

  ```
  	J = 0; dw1 = 0; dw2 =0; db = 0;         # Devs.
  	w1 = 0; w2 = 0; b=0;					# Weights
  	for i = 1 to m
  		# Forward pass
  		z(i) = W1*x1(i) + W2*x2(i) + b
  		a(i) = Sigmoid(z(i))
  		J += (Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i)))

  		# Backward pass
  		dz(i) = a(i) - Y(i)
  		dw1 += dz(i) * x1(i)
  		dw2 += dz(i) * x2(i)
  		db  += dz(i)
  	J /= m
  	dw1/= m
  	dw2/= m
  	db/= m

  	# Gradient descent
  	w1 = w1 - alpha * dw1
  	w2 = w2 - alpha * dw2
  	b = b - alpha * db
  ```

- The above code should run for some iterations to minimize error.

- So there will be two inner loops to implement the logistic regression.

- Vectorization is so important on deep learning to reduce loops. In the last code we can make the whole loop in one step using vectorization!

### Vectorization
- `for-loops` are slow. Thats why we need vectorization to get rid of some of our for loops.
- `a=random.rand(1000000) b=random.rand(1000000)` - NumPy library `numpy.dot(a, b)` function is using vectorization by default.
- The vectorization can be done on CPU or GPU thought the SIMD operation. But its faster on GPU.
- Whenever possible, avoid for-loops.
- Most of the NumPy library methods are vectorized version.

### Vectorizing Logistic Regression
- As an input we have a matrix `X` and its `[Nx, m]` and a matrix `Y` and its `[Ny, m]`.
- We will then compute at instance `[z1,z2...zm] = W' * X + [b,b,...b]`. This can be written in python as:
```python
  Z = np.dot(W.T,X) + b    # Vectorization, then broadcasting, Z shape is (1, m)
  A = 1 / 1 + np.exp(-Z)   # Vectorization, A shape is (1, m)
``` 
- Vectorizing Logistic Regression's Gradient Output
``` python
  dz = A - Y                  # Vectorization, dz shape is (1, m)
  dw = np.dot(X, dz.T) / m    # Vectorization, dw shape is (Nx, 1)
  db = dz.sum() / m           # Vectorization, dz shape is (1, 1)
```
- <img src="media/06.png" width=300> <img src="media/05.png" width=300>


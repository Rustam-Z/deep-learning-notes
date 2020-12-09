# Deep Learning Projects🚀

By Rustam_Z🚀 | December 2, 2020

Here you can find all projects I have made using Deep Learing & Neural Netowrks & CNN & TensorFlow

### [Logistic Regression as a Neural Network](Logistic-Regression-as-a-Neural-Network)
- Cat vs non-cat classification using logistic regression. You can see the architecture of model below:

- <img src="img/LogReg_kiank.png" width=300>

### [Planar data classification with one hidden layer](Planar-data-classification-with-one-hidden-layer)
- Data classification, NN with a hidden layer vs logistic regression
- Practiced with the size of a hidden layer
- <img src="img/classification_kiank.png" width=300>

### [Building Deep Neural Network](Building-Deep-Neural-Network)
- A deep NN with many layers using ReLU activation fuction
- Deep Learning methodology to build the model:
    - Initialize parameters/Define hyperparameters
    - Loop for num_iterations:
        1. Forward propagation
        2. Compute cost function
        3. Backward propagation
        4. Update parameters (using parameters, and grads from backprop) 
  - Use trained parameters to predict labels
- <img src="img/final outline.png" width=300>

### [Image Classification with NN](Image-Classification-with-NN)
- Cat vs non-cat classification using 2-layer & L-layer NN

- Used the helper funtions implementations from the "Building Deep Neural Network"

- <img src="img/imvectorkiank.png" width=350><img src="img/2layerNN_kiank.png" width=300><img src="img/LlayerNN_kiank.png" width=320>

### [Optimization methods from scratch](Optimization_Methods) 
- Stochastic Gradient Descent, Mini-batch GD with momentum, Adam optimization

- **Adam** is one of the most effective optimization algorithms for training neural networks. It combines ideas from RMSProp and Momentum

- <img src="img/adam.png" width=500>

- <img src="img/opt1.gif" width=300><img src="img/opt2.gif" width=300>

### [Sign Language Detector](TensorFlow)
- First project using TensorFlow

- <img src="img/hands.png" width=400>

### [Convolutional Neural Networks: Application](CNNs-App)
- Sign language using the TensorFlow CNN model

- Info: training set - 95% accuracy, test set - 80% accuracy

- Used Adam optimization technique (mini batches) for minimizing the cost

- <img src="img/conv_kiank.gif" width=400><img src="img/conv1.png" width=400>

### [Emotion Tracking](Emotion-Tracking)
- `#keras`

- <img src="img/face_images.png" width=400>

- **Key Points to remember**
  - Keras is a tool for rapid prototyping. It allows you to quickly try out different model architectures.

  - The four steps in Keras:
    - **Create a model** `happyModel = HappyModel(X_train.shape[1:])`

    - **Compile a model** `happyModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])`

    - **Fit/Train** `happyModel.fit(x = X_train, y = Y_train, epochs = 40, batch_size = 60)`

    - **Evaluate/Test** `preds = happyModel.evaluate(x = X_test, y = Y_test)`

## ResNets
- Convolutional block <br><img src="img/convblock_kiank.png" width=500>

- Identity block 3 <br><img src="img/idblock3_kiank.png" width=500>

- ResNet <br><img src="img/resnet_kiank.png" width=600>
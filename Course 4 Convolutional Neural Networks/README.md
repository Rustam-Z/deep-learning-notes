# [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks)

Rustam_Z🚀 | 29 November 2020

- Foundations of Convolutional Neural Networks (padding, striding, polling, FC, CNN example)

- Deep convolutional models (LeNet-5, AlexNet, VGG, ResNet, Inception)

- Object detection

- Special applications: Face recognition & Neural style transfer

- About CNN briefly: https://towardsdatascience.com/convolutional-neural-network-17fb77e76c05

- https://cs231n.github.io/convolutional-networks/

- Types of Convolutions: https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

- ResNet: https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff

## Contents
- [WEEK 1 - Foundations of Convolutional Neural Networks](#WEEK-1:-Convolutional-Neural-Networks)

    - [Computer vision](#Computer-Vision)

    - [Edge detection example](#Edge-Detection-Example)

    - [More edge detection](#More-Edge-Detection) vertical and horizontal edges detection

    - [Padding](#Padding)

    - [Strided convolution](#Strided-Convolutions)

    - [Convolutions over volumes](#Convolutions-Over-Volume)

    - [One layer of a convolutional network](#One-Layer-of-a-Convolutional-Network)

    - [Simple convolution network example](Simple-Convolutional-Network-Example)

    - [Pooling layers](#Pooling-Layers)

    - [CNN Example](#CNN-Example)

    - [Why Convolutions?](#Why-Convolutions?)

- [WEEK 2 - Deep convolutional models: case studies](#Week-2)
    - Case studies
        - [Why look at case studies?](#Why-look-at-case-studies?)

        - [Classic Networks](#Classic-Networks)

        - [Residual Networks (ResNets)](#ResNets)

        - [Network in Network and 1 X 1 convolutions](#Networks-in-Networks-and-1x1-Convolutions)

        - [Inception Network](#Inception-Network)

    - Practical advices for using ConvNets
        - [Transfer learning](#Transfer-learning)

        - [Data augmentation](#Data-augmentation)

- [WEEK 3 - Object detection](#WEEK-3:-Object-detection) | Detection algorithms
    - [Object localization](#Object-Localization)

    - [Landmark detection](#Landmark-Detection)

    - [Object detection](#Object-Detection)

    - [Convolutional Implementation of Sliding Windows](#Convolutional-Implementation-of-Sliding-Windows)

    - [Bounding Box Predictions](#Bounding-Box-Predictions)

    - [Non-max Suppression](#Non-max-Suppression)

    - [Anchor Boxes](#Anchor-Boxes)

    - [YOLO Algorithm](#YOLO-Algorithm)

- [WEEK 4 - Face Recognition](#WEEK-4:-Face-Recognition)
    - [Face Recognition](#Face-Recognition)
        - [Triplet Loss](#Triplet-Loss)

    - [Neural Style Transfer](#Neural-Style-Transfer)

## WEEK 1: Convolutional Neural Networks
> Learn to implement the foundational layers of CNNs (pooling, convolutions) and to stack them properly in a deep network to solve multi-class image classification problems.

> Learning Objectives:
> - Explain the convolution operation
> - Apply two different types of pooling operations
> - Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
> - Build and train a ConvNet in TensorFlow for a classification problem

### Computer Vision
<img src="img/01.PNG" width=500> <img src="img/02.PNG" width=500>

### Edge Detection Example
<img src="img/03.PNG" width=500> <img src="img/04.PNG" width=500> <img src="img/05.PNG" width=500>

### More Edge Detection
<img src="img/06.PNG" width=500> <img src="img/07.PNG" width=500> <img src="img/08.PNG" width=500>

### Padding
<img src="img/09.PNG" width=500> <img src="img/10.PNG" width=500>

### Strided Convolutions
<img src="img/11.PNG" width=500> <img src="img/12.PNG" width=500> <img src="img/13.PNG" width=500>

### Convolutions Over Volume
<img src="img/14.PNG" width=500> <img src="img/15.PNG" width=500> <img src="img/16.PNG" width=500>

### One Layer of a Convolutional Network
<img src="img/17.png" width=500> <img src="img/18.png" width=500> <img src="img/19.png" width=500>

### Simple Convolutional Network Example
<img src="img/20.PNG" width=500> <img src="img/21.PNG" width=500>

### Pooling Layers
<img src="img/22.jpg" width=500> <img src="img/23.PNG" width=500> <img src="img/24.PNG" width=500> <img src="img/25.PNG" width=500>

### CNN Example
<img src="img/26.PNG" width=500> <img src="img/27.PNG" width=500>

### Why Convolutions?
<img src="img/28.PNG" width=500> <img src="img/29.PNG" width=500> <img src="img/30.PNG" width=500>


## WEEK 2
> Learn about the practical tricks and methods used in deep CNNs straight from the research papers.

> Learning Objectives:
> - Discuss multiple foundational papers written about convolutional neural networks
> - Analyze the dimensionality reduction of a volume in a very deep network
> - Implement the basic building blocks of ResNets in a deep neural network using Keras
> - Train a state-of-the-art neural network for image classification
> - Implement a skip connection in your network
> - Clone a repository from github and use transfer learning

### Case studies
### Why look at case studies?
- Some neural networks architecture that works well in some tasks can also work well in other tasks.

- Here are some classical CNN networks:
    - **LeNet-5**
    - **AlexNet**
    - **VGG**

- **ResNet** with 152 layers

- **Inception** architecture by Google is a good example to apply

### Classic Networks
<img src="img/31.PNG" width=500><img src="img/32.PNG" width=500><img src="img/33.PNG" width=500>

### ResNets
<img src="img/34.PNG" width=500><img src="img/35.PNG" width=500>
- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#why-resnets-work

### Networks in Networks and 1x1 Convolutions
<img src="img/36.PNG" width=500><img src="img/37.PNG" width=500>
- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#network-in-network-and-1-x-1-convolutions

### Inception Network
- ***Inception network morivation***
    <br><img src="img/38.PNG" width=500><img src="img/39.PNG" width=500><img src="img/40.PNG" width=500>

    - We could face the problem of computational cost. To overcome it we must use the 1*1 covolution.

    - https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#inception-network-motivation

- ***Inception Network***
<br><img src="img/41.PNG" width=500><img src="img/42.PNG" width=500>

- GoogleNet - https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#inception-network-googlenet

### Practical advices for using ConvNets
### Transfer Learning
- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#transfer-learning

### Data Augmentation
- Mirroring the picture, random cropping, rotation, shearing
<br><img src="img/43.PNG" width=500>

- <img src="img/44.PNG" width=500>

- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#data-augmentation

- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#state-of-computer-vision

### General notes Quiz
- Quiz analysis: https://www.programmersought.com/article/43173972595


##  WEEK 3: Object detection 
> Learn how to apply your knowledge of CNNs to one of the toughest but hottest field of computer vision: Object detection.

> Learning Objectives:
> - Describe the challenges of Object Localization, Object Detection and Landmark Finding
> - Implement non-max suppression to increase accuracy
> - Implement intersection over union
> - Label a dataset for an object detection application
> - Identify the components used for object detection (landmark, anchor, bounding box, grid, ...) and their purpose

### Object Localization
- Classification with localization problem (one object). Detection is finding multiply objects.

- <img src="img/45.PNG" width=500>

- We will output `pc=1` if there is an object, the placement (`bx, by, bh, bw`), and the object in the picture (`1. 0, 0`)

- If `pc=0` then other numbers are *don't-cares*

- The lost function:
    ```
    L(y',y) = {
                (y1'-y1)^2 + (y2'-y2)^2 + ...           if y1 = 1
                (y1'-y1)^2						        if y1 = 0
            }
    ```
- <img src="img/46.PNG" width=500><img src="img/47.PNG" width=500>

### Landmark Detection
- Outputting some points from the piture is the **landmark detection**

- `lx, ly` corners of eye, mouth, nose or finding the edge of the face

- <img src="img/48.PNG" width=500>

### Object Detection
- Cut up the picture and detect the object, we will use 'Sliding windows algorithm'.

- **Sliding windows detection** - choose a region, stride and detect, then make a bigger region. For each region feed the Conv net and decide if it's a car or not.

- SWD is very slow, high computational cost.

- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#object-detection-1

### Convolutional Implementation of Sliding Windows
- We can turn Fully Connected layers into Convolutional layers

- <img src="img/49.PNG" width=500>

- Convolution implementation of sliding windows: <br><img src="img/50.PNG" width=500>

- <img src="img/51.PNG" width=500>

- The weakness of the algorithm is that the position of the rectangle wont be so accurate. Maybe none of the rectangles is exactly on the object you want to recognize. <br><img src="img/53.png" width=500>

### Bounding Box Predictions
- YOLO - You Only Look Once, developed in 2015

- Take a picture, divide into grid cells, implement classification with localization for each grid cell

- <img src="img/52.png" width=500><img src="img/54.png" width=500>

- https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/

- How to Label images for YOLO https://cloudxlab.com/blog/label-custom-images-for-yolo/

### Intersection Over Union
- Intersection Over Union is a function used to evaluate the object detection algorithm.

- The higher the IOU the better is the accuracy. Usually `>0.5` or `>0.6`

- <img src="img/55.PNG" width=500>

### Non-max Suppression
- Non-max suppression helps to choose the best bounding box for the particular object.

- <img src="img/57.PNG" width=500><img src="img/56.PNG" width=500>

### Anchor Boxes
- For detecting multiple objects in one grid. You must define the number of boxes by hand.

- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#anchor-boxes

### YOLO Algorithm
- https://towardsdatascience.com/object-detection-part1-4dbe5147ad0a

- https://datascience.stackexchange.com/questions/26403/how-does-yolo-algorithm-detect-objects-if-the-grid-size-is-way-smaller-than-the

- Receptive field in CNNs: https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807

- The algorithm runs a convolutional network on the entire image at the same time, and the way you've drawn it the network will output a tensor of shape 3x3xk where k is the number of features that it needs to represent the selected number of bounding boxes. Each one of these output neurons has a receptive field much larger than the box itself, meaning that the output neurons in the lower center can see the entire woman and the entire car. The output neurons in the lower right can see most or all of the car too, but it sees that the center of the car isn't in it's box so it knows it doesn't have to spit out a bounding box.

- <img src="img/58.PNG" width=500><img src="img/59.PNG" width=500><img src="img/60.PNG" width=500>

### Quiz on Detection algorithms
5. When training one of the object detection systems described in lecture, you need a training set that contains many pictures of the object(s) you wish to detect. However, bounding boxes do not need to be provided in the training set, since the algorithm can learn to detect the objects by itself.
    > False

7. .In the YOLO algorithm, at training time, only one cell —the one containing the center/midpoint of an object— is responsible for detecting this object.(**Note:** Even thought the object might be detected in more than one bounding box the `winner` will always be the one with its midpoint since all the other boxes would lead to a smaller IoU and be cut out in the process)
    > True

## WEEK 4: Face Recognition

### Face Recognition
- Face recognition (identifies face) and liveness detection (supervised learning)

- Face verification vs. face recognition:
    - Verification:
        - Input: image, name/ID. (1 : 1)
        - Output: whether the input image is that of the claimed person.
        - "is this the claimed person?"
    - Recognition:
        - Has a database of K persons
        - Get an input image
        - Output ID if the image is any of the K persons (or not recognized)
        - "who is this person?"

- We can use a face verification system to make a face recognition system.

### One Shot Learning
<img src="img/61.png" width=500><img src="img/62.png" width=500>

### Siamese Network
- The function of `d` is to input two faces and tell how similar or how different they are. A good way to do this is to use a **Siamese network**.

- <img src="img/63.png" width=500><img src="img/64.png" width=500>

- Use back propagation to tune the parameters

-  But how do you actually define an objective function to make a neural network learn to do what we just discussed here? (triplet loss function)

### Triplet Loss
<img src="img/65.png" width=500><img src="img/66.png" width=500><img src="img/67.png" width=500><img src="img/68.png" width=500>

### Face Verification and Binary Classification
<img src="img/69.png" width=500><img src="img/70.png" width=500>

## Neural Style Transfer
- Generating artwork

- <img src="img/71.png" width=500>

### What are deep ConvNets learning?
<img src="img/72.png" width=500><img src="img/73.png" width=500>

- Deeper network detects more complex objects than shallow network

### Cost Function
<img src="img/74.png" width=500><img src="img/75.png" width=500>

### Content Cost Function
<img src="img/76.png" width=500>

### Style Cost Function
<img src="img/77.PNG" width=500><img src="img/78.PNG" width=500><img src="img/79.PNG" width=500><img src="img/80.PNG" width=500>
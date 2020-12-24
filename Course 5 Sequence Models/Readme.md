# [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models) (NLP)

Rustam_Z🚀 | 20 December 2020

    - RNN, LSTM, BRNN, GRU

## References:
- [RNN Notes](https://github.com/mbadry1/DeepLearning.ai-Summary)

- https://towardsdatascience.com/learn-how-recurrent-neural-networks-work-84e975feaaf7

- LSTM http://colah.github.io/posts/2015-08-Understanding-LSTMs/

- Difference between RNN, LSTM, GRU: https://stats.stackexchange.com/questions/222584/difference-between-feedback-rnn-and-lstm-gru

## Contents:
- [WEEK 1](#WEEK-1)
    - Recurrent Neural Networks (forward prop, BP)

    - Gated Recurrent Unit (GRU) 

    - Long Short Term Memory (LSTM)

    - Bidirectional RNN (remebers before-after)

    - Deep RNNs

    - [Lab 1: Building RNN](#Lab:-Building-RNN) (RNN, LSTM implementations step-by-step)

- [WEEK 2](#WEEK-2)

- [WEEK 3](#WEEK-3)

## WEEK 1
- Recurrent Neural Networks 
- LSTMs, GRUs, Bidirectional RNNs
Applications: speech recognition, musics generation, machine traslation, name entity recognition...

### Notation
- The text of words is represented as the vector of words
- <img src="img/01.PNG" width=50    0><img src="img/02.PNG" width=500>

### Recurrent Neural Network Model
- <img src="img/03.PNG" width=500><img src="img/04.PNG" width=500>
- <img src="img/05.PNG" width=500><img src="img/06.PNG" width=500>

### Backpropagation through time
- <img src="img/07.png" width=500>

### Different types of RNNs
- <img src="img/08.PNG" width=500><img src="img/09.PNG" width=500>
- <img src="img/10.PNG" width=500><img src="img/11.PNG" width=500>

### Language model and sequence generation
<img src="img/12.PNG" width=500> 

### Gated Recurrent Unit (GRU)
<img src="img/13.PNG" width=500><img src="img/14.PNG" width=500><img src="img/15.PNG" width=500>

### Long Short Term Memory (LSTM)
<img src="img/17.PNG" width=500><img src="img/16.png" width=500><img src="img/18.PNG" width=500>

### Lab: Building RNN
- https://www.coursera.org/learn/nlp-sequence-models/ungradedLab/Jbfx1/lab

- **Recurrent Neural Networks (RNN)** are very effective for Natural Language Processing and other sequence tasks because they have *"memory"*.

- They can read inputs x<sup>⟨t⟩</sup> (such as words) one at a time, and remember some information/context through the hidden layer activations that get passed from one time-step to the next.

-  This allows a unidirectional RNN to take information from the past to process later inputs.

- **A bidirectional RNN** can take context from **both** the past and the future.

- Notation:
    - Superscript [l] denotes an object associated with the  l<sup>th</sup>  layer.

    - Superscript (i) denotes an object associated with the  i<sup>th</sup> example.

    - Superscript ⟨t⟩ denotes an object at the t<sup>th</sup> time-step.

    - Subscript i denotes the i<sup>th</sup> entry of a vector.

    - a<sub>5</sub><sup>(2)[3]<4></sup> denotes the activation of the 2nd training example (2), 3rd layer [3], 4th time step , and 5th entry in the vector.

- Forward propagation for the **basic Recurrent Neural Network**: <br><img src="img/RNN.png" width=500>

- **Basic RNN cell**. Takes as input x<sup>⟨t⟩</sup> (current time-step's input data) and a<sup>⟨t−1⟩</sup> (previous hidden state containing information from the past), and outputs a<sup>⟨t⟩</sup> which is given to the next RNN cell and also used to predict ŷ<sup>⟨t⟩</sup>: <br><img src="img/rnn_step_forward.png" width=500>

- A recurrent neural network (RNN) is a repetition of the RNN cell described above. The input sequence `x=(x⟨1⟩,x⟨2⟩,...,x⟨Tx⟩)` is carried over T<sub>x</sub> time steps. The network outputs `y=(y⟨1⟩,y⟨2⟩,...,y⟨Tx⟩)`:<br><img src="img/rnn_forward_sequence.png" width=700>

- **Long Short-Term Memory (LSTM) network**. LSTM cell - tracks and updates a "cell state" or memory variable c<sup>⟨t⟩</sup> at every time-step, which can be different from a<sup>⟨t⟩</sup>.
Note, the *softmax** includes a dense layer and softmax:
<br><img src="img/LSTM.png" width=500>

- Forward pass for LSTM: iterate this over this using a for-loop to process a sequence of T<sub>x</sub> inputs: <br><img src="img/LSTM_rnn.png" width=700>

## WEEK 2

## WEEK 3
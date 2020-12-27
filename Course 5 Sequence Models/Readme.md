# [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models)

Rustam_Z🚀 | 20 December 2020

    - RNN, LSTM, BRNN, GRU
    - Natural Language Processing & Word Embeddings (Word2vec & GloVe)
    - Sequence models & Attention mechanism (Speech recognition)

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
    - [Lab 2: Character level language model (Dinosaurus Island)](Lab:-Character-level-language-model-(Dinosaurus-Island))
- [WEEK 2](#WEEK-2) (NLP & Word Embeddings)
    - Embedding words (vs one-shot vector)
    - Word2vec & GloVe
    - Negative sampling
    - [Lab: Operations on word vectors](#Lab:-Operations-on-word-vectors)
    - [Lab: Emojify!](#Lab:-Emojify!)
- [WEEK 3](#WEEK-3) (Sequence models & Attention mechanism)
    - Various sequence to sequence architectures
    - Speech recognition - Audio data

## WEEK 1
> Recurrent Neural Networks 
> LSTMs, GRUs, Bidirectional RNNs
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

### Lab: Character level language model (Dinosaurus Island)
- https://www.coursera.org/learn/nlp-sequence-models/ungradedLab/19hZX/lab
- Skills:
    - How to store text data for processing using an RNN
    - How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
    - How to build a character-level text generation recurrent neural network
    - Why clipping the gradients is important
- Model:
    - Initialize parameters
    - Run the optimization loop
    - *Forward propagation* to compute the loss function
    - *Backward propagation* to compute the gradients with respect to the loss function
    - Clip the gradients to avoid exploding gradients
    - Using the gradients, update your parameters with the gradient descent update rule.
    - Return the learned parameters
- Loop:
    - forward pass
    - cost computation
    - backward pass
    - parameter update

### Lab: Music generation
- Jazz generation using LSTM

## WEEK 2
> Natural Language Processing & Word Embeddings

### Introduction to Word Embeddings
### Word Representation
<img src="img/19.PNG" width=500><img src="img/20.PNG" width=500><img src="img/21.PNG" width=500>

### Using word embeddings
<img src="img/evolving-word-embeddings-fig-1.jpeg" width=700>

### Properties of word embeddings
<img src="img/22.PNG" width=500><img src="img/23.PNG" width=500><img src="img/24.PNG" width=500>

### Embedding matrix
- <img src="img/25.PNG" width=500>
- In practice multiplying to zeros is not used. And here Word2vec & GloVe are coming.

### Learning Word Embeddings: Word2vec & GloVe
### Learning word embeddings
https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/5-%20Sequence%20Models#learning-word-embeddings

### Word2Vec
- Simple and comfortably more efficient way to learn types of embeddings

- Randomly choose a target word

- <img src="img/26.PNG" width=500><img src="img/27.PNG" width=500><img src="img/28.PNG" width=500>

### Negative Sampling
<img src="img/29.PNG" width=500><img src="img/30.PNG" width=500><img src="img/31.PNG" width=500>

### GloVe word vectors
<img src="img/32.PNG" width=500><img src="img/33.PNG" width=500><img src="img/34.PNG" width=500>

### Applications using Word Embeddings
### Sentiment Classification

### Debiasing word embeddings
- Gender, ethics specific
- Doctor --> Man & Woman

### Lab: Operations on word vectors
- https://www.coursera.org/learn/nlp-sequence-models/ungradedLab/SRDFh/lab
- **Goal**:
    - Load pre-trained word vectors, and measure similarity using cosine similarity
    - Use word embeddings to solve word analogy problems such as Man is to Woman as King is to __.
    - Modify word embeddings to reduce their gender bias

- **Embedding vectors versus one-hot vectors**
    - One-hot vectors do not do a good job of capturing the level of similarity between words (every one-hot vector has the same Euclidean distance from any other one-hot vector).
    - Embedding vectors such as GloVe vectors provide much more useful information about the meaning of individual words.
    - Implemented `Cosine similarity`

- **Summary**:
    - Cosine similarity is a good way to compare the similarity between pairs of word vectors.
    - Note that L2 (Euclidean) distance also works.
    - For NLP applications, using a pre-trained set of word vectors is often a good way to get started.

- **Debiasing word vectors**

### Lab: Emojify!
- https://www.coursera.org/learn/nlp-sequence-models/ungradedLab/HxLGO/lab
- LSTM, word embeddings

## WEEK 3
> Sequence models & Attention mechanism

### Various sequence to sequence architectures
#### Basic Models
- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/5-%20Sequence%20Models#basic-models

#### Picking the most likely sentence
- Language model vs Machine translation model

- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/5-%20Sequence%20Models#picking-the-most-likely-sentence

#### Beam Search
<img src="img/35.PNG" width=500><img src="img/36.PNG" width=500><img src="img/37.PNG" width=500>

#### Refinements to Beam Search
https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/5-%20Sequence%20Models#refinements-to-beam-search

#### Bleu Score
- Bilingual evaluation undestudy
- How good is the machine translation?

### Speech recognition - Audio data

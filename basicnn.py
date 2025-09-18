import math
import numpy as np
import pandas as pd
import random
from enum import Enum


# Useful functions
def element_wise_multiply(vector1: list[int], vector2: list[int]) -> list[int]:
    return [a*b for a, b in zip(vector1, vector2)]

def element_wise_multiply(vector1: list[int], application) -> list[int]:
    return [application(a) for a in vector1]

def show_weights(weights: list[int]):
    for i in weights: print([[round(float(j), 4) for j in x] for x in i])

# Activation functions
def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lrelu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def tanh(x):
    return math.tanh(x)

class activs(Enum):
    RELU = relu
    TANH = tanh
    SIGM = sigmoid
    LRLU = lrelu
    SFMX = softmax



# Classification functions
def binary(nums): # For when each output is 1 or 0
    return [1/(1+(math.e**-x)) for x in nums]

def multi_class(nums): # For when only one output is 1 (one-hot)
    total = sum([math.e**x for x in nums])
    return [(math.e**x)/total for x in nums]
    
class classifications(Enum):
    LIN = 1
    BIN = binary
    MLT = multi_class


# Weight distribution functions
def he_init(n_in, n_out): # for RELU-type activations
    return np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)

def xavier_init(n_in, n_out): # for sigmoid and tanh
    return np.random.randn(n_out, n_in) * np.sqrt(1.0 / (n_in + n_out))


# Loss functions
def regress(prediction: list, expected: list): # For predicting numbers (Regression)
    return (1/len(prediction))*sum([(a-b)**2 for a, b in zip(prediction, expected)])

def binary_class(prediction: list, expected: list): # For when each output is either 1 or 0
    return (-1/len(prediction))*sum([(b*np.log(a))+((1-b)*(1-np.log(a))) for a, b in zip(prediction, expected)])

def multi_class_loss(prediction: list, expected: list): # For when only one output is 1
    return -1*sum([b*np.log(a) for a, b in zip(prediction, expected)])

class losses(Enum):
    REG = regress
    BIN = binary_class
    MLT = multi_class_loss


class Neuron():
    def __init__(self):
        self.val = None
        self.name = "X"

    def reset(self):
        self.val = None

class InputNeuron(Neuron):
    pass

class HiddenNeuron(Neuron):
    def __init__(self, original_bias):
        super().__init__()
        self.bias = original_bias

class OutputNeuron(Neuron):
    def __init__(self, original_bias):
        super().__init__()
        self.bias = original_bias


class Network():
    def __init__(self, lays: list[int], activation: activs = activs.RELU, original_bias: float = 0.0, classification: classifications = classifications.LIN, loss: losses = losses.REG):
        if len(lays) == 1:
            raise ValueError("Invalid number of layers")
        
        self.activation = activation
        self.classif = classification
        self.loss = loss

        self.layers = self.create_layers(lays, original_bias)
        print(f"{lays[0]} inputs")
        print()
        print("Layers created:")
        for i in self.layers: print([x.name for x in i])
        print()

        self.weights = self.create_weights(lays, activation)
        print("Weights created:")
        show_weights(self.weights)
        print()

        print(f"List of all biases: (Should be all {original_bias})")
        biases = self.count_biases()
        for i in biases:
            print(i)
        print()

    def create_layers(self, lays: list[int], original_bias: float) -> list:
        layers = []
        layers.append([InputNeuron() for x in range(lays[0])])
        for i, v in enumerate(lays[1:-1]):
            layers.append([HiddenNeuron(original_bias) for x in range(v)])
        layers.append([OutputNeuron(original_bias) for x in range(lays[-1])])
        return layers
    
    def create_weights(self, lays: list[int], activ: activs) -> list:
        weights = []
        for i, v in enumerate(lays[:-1]):
            weights.append(he_init(v, lays[i+1])) if activ == activs.RELU or activ == activs.LRLU else weights.append(xavier_init(v, lays[i+1]))
        return weights
    
    def count_biases(self) -> list[list]:
        return [[x.bias for x in i] for i in self.layers[1:]]
    
    def reset_neurons(self):
        for i in self.layers:
            for x in i:
                x.reset()

    # Must call self.reset_neurons() before running self.forward_pass() a second time!
    def forward_pass(self, inputs: list[int]) -> list[int]:
        for i, v in enumerate(self.layers[0]):
            v.val = inputs[i]
        for i, v in enumerate(self.layers[1:]):
            for j, v2 in enumerate(v):
                value = sum([self.weights[i][j][x] * self.activation(self.layers[i][x].val) for x in range(len(self.layers[i]))]) + v2.bias
                v2.val = value
        output = [float(x.val) for x in self.layers[-1]]
        if not self.classif == classifications.LIN:
            output = self.classif(output)
        return output
    
    def back_prop(self, expected: list[int]): # INCOMPLETE, need to calculate gradient for each layer
        predicted = [float(x.val) for x in self.layers[-1]]
        l = self.loss(predicted, expected)
        print()
        print("Current weights:")
        show_weights(self.weights)
        print()
        return l

    def train(self, trainx: list[list], trainy: list[list], epochs: int = 20, batch_size: int = 16): # INCOMPLETE
        for i in range(epochs):
            predicted = self.forward_pass(trainx)
            l = self.back_prop(trainy)






default_bias = 0.01



net = Network([2, 3, 2], activation=activs.RELU, original_bias=default_bias, classification=classifications.LIN, loss=losses.REG)

output = net.forward_pass([5, 1])
expected = [2, 3]
loss = net.back_prop(expected)

print(f"Network output: {output}")
print(f"Expected output: {expected}")

print()
print(f"Network loss: {loss}")

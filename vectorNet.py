import math
import numpy as np
import pandas as pd
import random

def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

def xavier_init(n_in, n_out): # for sigmoid and tanh
    return np.random.randn(n_out, n_in) * np.sqrt(1.0 / (n_in + n_out))

def binary_class(prediction: list, expected: list): # For when each output is either 1 or 0
    return (-1/len(prediction))*sum([(b*np.log(a))+((1-b)*(1-np.log(a))) for a, b in zip(prediction, expected)])

def multi_class_loss(prediction: list, expected: list): # For when only one output is 1
    return -1*sum([b*np.log(a) for a, b in zip(prediction, expected)])


class Net():
    def __init__(self, layers, default_bias):
        self.layers = layers
        self.weights = []
        self.biases = [[default_bias for x in range(z)] for z in self.layers[1:]]

        for i, v in enumerate(self.layers[:-1]):
            self.weights.append(xavier_init(v, self.layers[i+1]).tolist())
        
        for i in self.weights:
            print(i)
            print()

        print()
        for i in self.biases:
            print(i)
            print()

    def infer(self, ins):
        if len(ins) == self.layers[0]:
            
            values = [sigmoid(x) for x in ins]
            all_vals = [values]
            for i in range(len(self.layers)-1):
                values = self.calc(i, values)
                all_vals.append(values)
            return all_vals
        else:
            raise ValueError("Incorrect amount of inputs")
    
    def calc(self, weights_layer, inputs):
        outputs = []
        # print(f"Weights Layer: {weights_layer}")
        # print(f"Weights: {self.weights[weights_layer]}")
        # print(f"Inputs: {inputs}")
        # print(f"Biases: {self.biases[weights_layer]}")
        # print()
        # print()

        for i, v in enumerate(self.weights[weights_layer]):
            outputs.append(sigmoid(sum([v2*inputs[i2] for i2, v2 in enumerate(v)]) + self.biases[weights_layer][i]))
        return outputs
        




net = Net([1, 3, 2, 1], 0.01)

print(net.infer([0.5]))
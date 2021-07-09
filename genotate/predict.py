import os
import sys
import numpy as np

from weights import *


def relu(A):
    return np.maximum(A, 0)

def sigmoid(A, derivative=False):
    if derivative:
        return (np.exp(-A))/((np.exp(-A)+1)**2)
    return 1/(1 + np.exp(-A))

def softmax_logits(output_array):
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis = 1, keepdims = True)

def softmax(A):
	expA = np.exp(A-np.max(A))
	return expA / expA.sum()


data = np.array([[0.44764,0.12821,0.15385,0.53846,0.17949,0.28205,0.35897,0.20513,0.15385,0.10256,0.07692,0.15385,0.66667,0.0,0.0,0.0,0.23077,0.0,0.10256,0.05128,0.02564,0.12821,0.0,0.0,0.07692,0.10256,0.0,0.02564,0.02564,0.0,0.05128,0.07692,0.02564,0.02564,0.02564,0.02564]])

layer0 = np.dot(data, layer0_weights)
layer0 = layer0 + layer0_bias
layer1 = np.dot(layer0, layer1_weights)
layer1 = relu(layer1 + layer1_bias)
layer2 = np.dot(layer1, layer2_weights)
layer2 = relu(layer2 + layer2_bias)
scores = np.dot(layer2, layer3_weights) + layer3_bias
probs = softmax(scores)
print(probs)

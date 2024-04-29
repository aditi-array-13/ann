#!/usr/bin/env python
# coding: utf-8

# # 11. Write a python program in python program for creating a Back Propagation Feed-forward neural network

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)


# In[3]:


class NeuralNetwork:
    def __init__(self,inputs,hidden,outputs):
        self.weights_ih = np.random.rand(inputs,hidden)
        self.weights_ho = np.random.rand(hidden,outputs)
        
    def feedforward(self,inputs):
        hidden_inputs = np.dot(inputs,self.weights_ih)
        hidden_outputs = sigmoid(hidden_inputs)
        output_inputs = np.dot(hidden_outputs,self.weights_ho)
        output_outputs = sigmoid(output_inputs)
        
        return output_outputs
    
    def train(self,inputs,targets,learning_rate):
        hidden_inputs = np.dot(inputs,self.weights_ih)
        hidden_outputs = sigmoid(hidden_inputs)
        output_inputs = np.dot(hidden_outputs,self.weights_ho)
        output_outputs = sigmoid(output_inputs)
        
        output_error = targets - output_outputs
        
        output_dervative = sigmoid_derivative(output_outputs)
        
        hidden_error = np.dot(output_error,self.weights_ho.T)
        
        hidden_derivative = sigmoid_derivative(hidden_outputs)
        
        self.weights_ho += learning_rate * np.dot(hidden_outputs.T,output_error * output_dervative)
        
        self.weights_ih += learning_rate * np.dot(inputs.T,hidden_error * hidden_derivative)
        
neural_network = NeuralNetwork(2,4,1)

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([[0],[0],[0],[1]])

for i in range(10000):
    neural_network.train(inputs,targets,0.1)


# In[4]:


print(neural_network.feedforward(np.array([0,0])).round())
print(neural_network.feedforward(np.array([0,1])).round())
print(neural_network.feedforward(np.array([1,0])).round())
print(neural_network.feedforward(np.array([1,1])).round())


# In[ ]:





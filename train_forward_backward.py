#!/usr/bin/env python
# coding: utf-8

# # 7. Implement Artificial Neural Network training process in Python by using Forward Propagation, Bacward Propagation.

# In[7]:


import numpy as np


# In[8]:


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def forward(inputs,hidden_wts,hidden_bias,output_wts,output_bias):
    hidden_output = sigmoid(np.dot (inputs,hidden_wts)+hidden_bias)
    predicted_output = sigmoid(np.dot(hidden_output,output_wts)+output_bias)
    return hidden_output,predicted_output

def backward(learn,inputs,target,hidden_output,predicted_output,output_wts,output_bias,hidden_wts,hidden_bias):
    error = target - learn
    delta_output = error*sigmoid_derivative(predicted_output)
    
    error_hidden = delta_output.dot(output_wts.T)
    delta_hidden = error*sigmoid_derivative(hidden_output)
    
    #Update weights and biases
    output_wts += hidden_output.T.dot(delta_output)*learn
    output_bias += np.sum(delta_output,axis=0)*learn
    hidden_wts += inputs.T.dot(delta_hidden)*learn
    hidden_bias += np.sum(delta_hidden,axis=0)*learn
    
    return output_wts,output_bias,hidden_wts,hidden_bias


# In[9]:


inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([[0],[0],[1],[0]]) # ANDNOT Gate

def train(inputs,target):
    input_neurons,hidden_neurons,output_neurons = 2,2,1
    hidden_wts = np.random.uniform(size=(input_neurons,hidden_neurons))
    hidden_bias = np.random.uniform(size=(1,hidden_neurons))
    output_wts = np.random.uniform(size=(hidden_neurons,output_neurons))
    output_bias = np.random.uniform(size=(1,output_neurons))
    
    epochs = 1000
    learn = 0.3
    
    for epoch in range(epochs):
        hidden_output,predicted_output = forward(inputs,hidden_wts,hidden_bias,output_wts,output_bias)
        output_wts,output_bias,hidden_wts,hidden_bias = backward(learn,inputs,target,hidden_output,predicted_output,output_wts,output_bias,hidden_wts,hidden_bias)
        if epoch == 999:
            loss = np.mean(np.square(target - predicted_output))
            print(f"Epoch {epoch}: Loss = {loss}")
            
    return output_wts,output_bias,hidden_wts,hidden_bias


# In[10]:


test = np.array([[0,0],[0,1],[1,0],[1,1]])
output_wts,output_bias,hidden_wts,hidden_bias = train(inputs,target)
hidden,predictions = forward(test,hidden_wts,hidden_bias,output_wts,output_bias)
print('Predictions: ',*predictions)


# In[11]:


difference = target - predictions
difference


# In[12]:


accuracy = 0
for i in range(len(difference)):
    accuracy += difference[i][0]

accuracy = (1 + accuracy/len(difference))*100
print("Average Accuracy of predictions: ",accuracy)


# In[ ]:





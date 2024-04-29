#!/usr/bin/env python
# coding: utf-8

# #  1. Write a Python program to plot a few activation functions that are being used in neural networks.

# In[1]:


import matplotlib.pyplot as plt 
import numpy as np


# In[7]:


def RELU(x): 
# It returns zero if the input is less than zero otherwise it returns the give x1=[] 
    x1=[]
    for i in x: 
        if i<0: 
            x1.append(0) 
        else: 
            x1.append(i) 
    return x1 

x = np.linspace(-10, 10) 
plt.grid(True) 
plt.plot(x, RELU(x)) 
plt.axis('tight') 
plt.title('Activation Function :RELU') 
plt.show() 


# In[8]:


def binary(x):
    x1=[]
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(1)
            
    return x1

x = np.linspace(-5, 20)
plt.grid(True)
plt.plot(x, binary(x))
plt.axis('tight')
plt.title('Activation Function : Binary')
plt.show()


# In[9]:


def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.linspace(-5, 10)
plt.grid(True)
plt.plot(x, sigmoid(x))
plt.axis('tight')
plt.title('Acitvation Function : Sigmoid')
plt.show()


# In[10]:


def tanh(x):
    return (2/(1+np.exp(-2*x)))-1

x=np.linspace(-1, 1)
plt.grid(True)
plt.plot(x, tanh(x))
plt.axis('tight')
plt.title('Acitvation Function : TanH')
plt.show()


# In[12]:


def linear(x):
    return x;

x=np.linspace(0, 10)
plt.grid(True)
plt.plot(x, linear(x))
plt.axis('tight')
plt.title('Acitvation Function : linear')
plt.show()


# In[13]:


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

x = np.linspace(-5, 1)
plt.grid(True)
plt.plot(x, softmax(x))
plt.axis('tight')
plt.title('Activation Function : softmax')
plt.show()


# In[ ]:





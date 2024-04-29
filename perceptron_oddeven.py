#!/usr/bin/env python
# coding: utf-8

# # 3. Write a Python Program using Perceptron Neural Network to recognise even and odd numbers. Given numbers are in ASCII form 0 to 9
# 

# In[3]:


import numpy as np

# Training data: ASCII representation of 0 to 9
training_data = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0],  # ASCII for 0
    [0, 1, 1, 1, 0, 0, 0, 1],  # ASCII for 1
    [0, 1, 1, 1, 0, 1, 0, 0],  # ASCII for 2
    [0, 1, 1, 1, 1, 0, 0, 1],  # ASCII for 3
    [0, 1, 1, 1, 1, 0, 1, 0],  # ASCII for 4
    [0, 1, 1, 1, 1, 1, 0, 1],  # ASCII for 5
    [0, 1, 1, 1, 1, 1, 1, 0],  # ASCII for 6
    [0, 1, 1, 1, 1, 1, 1, 1],  # ASCII for 7
    [1, 1, 0, 0, 0, 0, 0, 0],  # ASCII for 8
    [1, 1, 0, 0, 0, 0, 0, 1]   # ASCII for 9
])

# Labels: 0 for even, 1 for odd
labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Perceptron training
np.random.seed(42)  # Set a seed for reproducibility
weights = np.random.rand(len(training_data[1]))
print("Initial Weights:", weights)
bias = 0
learning_rate = 0.1
epochs = 10

for epoch in range(epochs):
    for inputs, label in zip(training_data, labels):
        prediction = np.dot(inputs, weights) + bias
        prediction = 1 if prediction >= 0 else 0
        weights += learning_rate * (label - prediction) * inputs
        bias += learning_rate * (label - prediction)

# Test a number (replace ASCII representation)
test_number = []
for i in range (0, 8):
    x = int(input("Enter the binary numbers : "))
    test_number.append(x)

# Perceptron prediction
prediction = np.dot(test_number, weights) + bias
print("prediction", prediction)

prediction = 1 if prediction >= 0 else 0
print("\n")
print("Predicted:", prediction)
print("\n")
# Output result
if prediction == 0:
    print("The number is even.")
else:
    print("The number is odd.")


# In[ ]:





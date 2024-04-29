#!/usr/bin/env python
# coding: utf-8

# # 5. Write a python Program for Bidirectional Associative Memory with two pairs of vectors

# In[1]:


import numpy as np

pattern1 = np.array([[1, 1, 1, -1, -1]]) #Pattern A
pattern2 = np.array([[-1, -1, -1, 1, 1]]) #Pattern B

pattern1_transpose = pattern1.T
pattern2_transpose = pattern2.T

weigths = np.dot(pattern1_transpose, pattern2)

input_pattern1 = np.array([[1, 1, 1, -1, -1]])
input_pattern2 = np.array([[-1, -1, -1, -1, 1]])
output_pattern1 = np.sign(np.dot(input_pattern1, weigths))
print(f"Recalled Pattern 1: {output_pattern1}")

output_pattern2 = np.sign(np.dot(input_pattern2, weigths.T))
print(f"Recalled Pattern 2: {output_pattern2}")


# In[ ]:





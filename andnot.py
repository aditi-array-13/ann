#!/usr/bin/env python
# coding: utf-8

# #  2. Generate ANDNOT function using McCulloch-Pitts neural net by a python program.

# ## ANDNOT

# In[6]:


def ANDNOT(x1, x2):
    w1  = 1
    w2 = -1
    y = []
    theta = 0

    for i in range(0, 4):
        sum = x1[i]*w1+x2[i]*w2
        if (sum > theta):
            y.append(1)
            print('Appended 1')
        else:
            y.append(0)
            print('Appended 0')
    print("X1 : ",x1)
    print("X2 : ",x2)
    print("Ans : ",y)
    
    expected_output = [0, 0, 1, 0]

    is_correct = y == expected_output

    print("Output:", y)
    if is_correct:
        print("ANDNOT function is correct for all inputs!")
    else:
        print("ANDNOT function is not correct!")


x1 = []
x2 = []
for i in range(0, 4):
    ele1 = int(input("Enter no. for x1 : "))
    x1.append(ele1)

for i in range(0, 4):
    ele2 = int(input("Enter no. for x2 : "))
    x2.append(ele2)

# x1 = [0, 0, 1, 1]
# x2 = [0, 1, 0, 1]
ANDNOT(x1, x2)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

'''
Estimation using closed form expression W=(((X^T) X)^-1)(X^T)Y
'''


#Training data 
fsize=np.array([1600,1260,1800,600,850,920,1090,890,1340,1650]).reshape(10,1) #x2 = flat sizes
fnumbed=np.array([3,2,4,1,2,2,2,2,3,2]).reshape(10,1)                         #x3 = number of bedrooms
X=np.ones(10,dtype=int).reshape(10,1)                                         #x1 = 1 dummy

X=np.hstack((X,fsize,fnumbed))
Y=np.array([8.2,6.6,10.3,1.7,3.6,4.4,5.4,4.8,10.5,7.4]).reshape(10,1)
t=np.linalg.inv(np.dot(X.T,X))
w=t.dot(X.T).dot(Y)                                                           # Obtained w for Y=Xw
print('Estimated w:',w)

#Test data
X1=np.array([1,950,2])                                                        # 950 sq ft, 2 bedrooms
Y1=np.dot(X1,w)                                                               # Y=Xw
print('Estimated price : lower range (950 sq ft, 2 bedrooms)=',Y1)

X2=np.array([1,1050,3])                                                       # 1050 sq ft, 3 bedrooms
Y2=np.dot(X2,w)
print('Estimated price : higher range (1050 sq ft, 3 bedrooms)=',Y2)


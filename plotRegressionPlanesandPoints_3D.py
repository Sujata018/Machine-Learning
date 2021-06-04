#!/usr/bin/env python
# coding: utf-8


import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


#Training data 
fsize=np.array([1600,1260,1800,600,850,920,1090,890,1340,1650]).reshape(10,1)               # x2 = flat sizes
fnumbed=np.array([3,2,4,1,2,2,2,2,3,2]).reshape(10,1)                                       # x3 = number of bedrooms
X=np.ones(10,dtype=int).reshape(10,1)                                                       # x1 = 1 dummy

X=np.hstack((X,fsize,fnumbed))
Y=np.array([8.2,6.6,10.3,1.7,3.6,4.4,5.4,4.8,10.5,7.4]).reshape(10,1)                       # y = flat prices

fig = plt.figure(figsize = (10, 7))                                                         # create 3D figure
ax = plt.axes(projection ="3d")


x1=X[:,1]                                                                                   # x axis
x2=X[:,2]                                                                                   # y axis
x1,x2=np.meshgrid(x1,x2)                                                                    # create mesh grid for x-y plane
y1=-1.9252501 + 0.00362953*x1 + 1.67818099*x2                                               # z axis Estimation : closed form solution
y2= 0.99924116 + 0.00269827*x1 + 0.9981126*x2                                               # z axis Estimation : gradient descent
 
g1=ax.plot_surface(x1, x2, y1, color="blue",alpha=0.1,label="closed form estimation")       # Plot closed form estimation in yellow transparent surface
g1._facecolors2d = g1._facecolor3d
g1._edgecolors2d = g1._edgecolor3d
g2=ax.plot_surface(x1, x2, y2, color="orange",alpha=0.1,label="gradient descent estimation")# Plot gradient descent estimation in orrange transparent surface
g2._facecolors2d = g2._facecolor3d
g2._edgecolors2d = g2._edgecolor3d


ax.scatter3D(X[:,1], X[:,2], Y, color = "red")                                              # scatter plot of the 10 training points
plt.title("flat price data")

ax.set_xlabel('flat size (sq ft)', fontweight ='bold')
ax.set_ylabel('# of bedrooms', fontweight ='bold')
ax.set_zlabel('Price (millions)', fontweight ='bold')
ax.legend()

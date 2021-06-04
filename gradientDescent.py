#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import matplotlib.pyplot as plt

'''
 Performs gradient descent given
 Inputs : X : feature matrix (with x0=1s),
          Y : dependent variabe vector,
          alpha: learning rate,
          epoch: the maximum cost difference between actual and estimates costs acceptible
          max_iter : number of maximum iteration after which the gradient descent proces will stop
 Output: w: the parameter vestor obtained by minimising cost with respect to the parameters
         plotdata: matrix with values of iteration sequence, values of cost and feature parameters for each iteration in gradient descent, for plotting graphs
  
'''

def gradient_descent(alpha,X,Y,epoch=0.0001,max_iter=500): # alpha : step, epoch : range for convergence, max_iter : maximum number of iterations
    converged = False         # converged will be set to True, when error difference between two iterations <= epoch
    iter = 0                  # number of iterations
    m=X.shape[0]              # number of samples
    w=np.ones(X.shape[1],dtype=int).reshape(X.shape[1],1) # Initialise w to 1s
    J=Y-np.dot(X,w)
    J= np.sum(J**2)           # J=(Y-Xw)^2
    print ('Iteration ',iter, ' cost = ',J)
    plotdata=np.array([iter,J,w[0][0],w[1][0],w[2][0]])
    
    
    while (converged==False and iter < max_iter): # iterative gradient descent until convergence or maximum iterations performed
            grad=(np.dot(X,w)-Y)*X
            grad=1/m*np.sum(grad,axis=0)
            grad=grad.reshape(X.shape[1],1)       # grad = 1/m sum (Xw-y)x
            temp=w-alpha*grad                     # update w = w - alpha * grad
            w=temp
            Jnew=Y-np.dot(X,w)
            Jnew= np.sum(Jnew**2)                 # new J=(Y-Xw)^2
            iter += 1
            print ('Iteration ',iter, ' cost = ',Jnew)
            plotdata=np.vstack([plotdata,np.array([iter,J,w[0][0],w[1][0],w[2][0]])])
            
            if abs(Jnew-J)<= epoch:
                converged = True                  # If cost difference between last step and this step is below epoch range, then end iterations
                print ('Converged!')
            J=Jnew

    return w,plotdata
    


# In[97]:


if __name__=='__main__':
    #Training data 
    fsize=np.array([1600,1260,1800,600,850,920,1090,890,1340,1650]).reshape(10,1) #x2 = flat sizes in square feet
    fnumbed=np.array([3,2,4,1,2,2,2,2,3,2]).reshape(10,1)                         #x3 = number of bedrooms
    X=np.ones(10,dtype=int).reshape(10,1)                                         #x1 = 1 dummy

    X=np.hstack((X,fsize,fnumbed))
    Y=np.array([8.2,6.6,10.3,1.7,3.6,4.4,5.4,4.8,10.5,7.4]).reshape(10,1)         # y in price / million

    alpha = 0.0000006                                                             # learning rate
    epoch = 0.000001                                                              # convergence criteria
    w,plotdata = gradient_descent(alpha,X,Y,epoch)
    print ('w=',w)
    
    plt.plot(plotdata[:,0],plotdata[:,1])
    plt.title("Cost vs iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
        
    plt.show()
 
    plt.plot(plotdata[:,0],plotdata[:,2],label='w0')
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()
    
    plt.plot(plotdata[:,0],plotdata[:,3],label='w1')
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()

    plt.plot(plotdata[:,0],plotdata[:,4],label='w2')
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()


from random import random, randrange
from math import exp, pi, log
from pylab import plot, ylabel, show
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import time
import random

################################################################################
#--------------------------SET OF FUNCTIONS------------------------------------#
################################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def dec_bound(x,weight):
    return -((weight[0]+weight[1]*x)/weight[2])

################################################################################
#------------------------------- DATASET --------------------------------------#
################################################################################

# XOR Truth' table
DataSet = np.array([[0,0,0],
                    [0,1,1],
                    [1,0,1],
                    [1,1,0]])

bias    = np.array([ 1 for i in range(len(DataSet[:,0])) ]).reshape(-1,1)

#-----------------------------INITIALIZATION-----------------------------------#

x         = np.append(bias, DataSet[:,0:2], axis=1) # INPUT + BIAS
t_outs    = np.array(DataSet[:,2]).reshape(-1,1)    # TARGET OUTPUT

n_input   = len(x[0])
n_neurons = 2           
weights1  = np.random.random((n_input, n_neurons))  # W1 x -> z
weights2  = np.random.random((n_neurons+1, 1))      # W2 z -> y

eta    = 0.5     # Learning rate
Loss   = []      # Initializing loss function
N_iter = 1000   # Number of iterations


################################################################################
#----------------------------------MAIN----------------------------------------#
################################################################################

start=time.time() 
for i in range(N_iter):
    z    = sigmoid(np.dot(x,weights1))
    z    = np.append(bias, z, axis=1)
    y    = sigmoid(np.dot(z,weights2))
    loss = 1/4*np.sum((y-t_outs)**2)
    gradient_weights2 = 2*np.dot(z.T,(y-t_outs)*y*(1-y))  # Back propagation 
    gradient_weights1 = 2*np.dot(x.T,np.dot((y-t_outs)*y*(1-y),weights2.T)*z*(1-z))
    weights2 += - eta*gradient_weights2                   # Weights corrections
    weights1 += - eta*gradient_weights1[:,1:n_neurons+1] 

    if i%100==0:
        x1f = [0,1]
        x2f = [0,1]
        x1t = [0,1]
        x2t = [1,0]
        plt.plot(x1f, x2f, 'ko', label = 'False')
        plt.plot(x1t, x2t, 'ro', label = 'True')
        x1 = np.array([i for i in np.arange(-0.5,5.5, 0.1)])

        for i in range(n_neurons):
            globals()['weights_'+str(i)] = []
            for j in range(len(weights1)):
                globals()['weights_'+str(i)].append(weights1[j,i])
            globals()['x2_'+str(i)] = dec_bound(x1, globals()['weights_'+str(i)])
            plt.plot( x1,globals()['x2_'+str(i)],label = 'Decision boundary {}'.format(str(i)))

        plt.grid(True)
        plt.title("")
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        plt.xlim([-1,2])
        plt.ylim([-1,2])
        plt.legend()
        plt.pause(0.01)
        plt.clf()






    Loss.append(loss)
end=time.time()
print("Wall-time used  : {} sec".format(str((np.round(end-start,2)))))

# final plot
plt.plot(x1f, x2f, 'ko', label = 'False')
plt.plot(x1t, x2t, 'ro', label = 'True')
for i in range(n_neurons):
    globals()['weights_'+str(i)] = []
    for j in range(len(weights1)):
        globals()['weights_'+str(i)].append(weights1[j,i])
    globals()['x2_'+str(i)] = dec_bound(x1, globals()['weights_'+str(i)])
    plt.plot( x1,globals()['x2_'+str(i)],label = 'Decision boundary {}'.format(str(i)))
plt.grid(True)
plt.title("")
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.xlim([-1,2])
plt.ylim([-1,2])
plt.legend()
plt.show()



################################################################################
# -------------------------------RESULTS---------------------------------------#
################################################################################

if 1:
    print("FINAL random weights : ")
    for j in range(len(weights1[0])):
         for i in range(len(weights1)):
            if i==0:
                print("Bias{}        : {}".format(str(j),str(np.round(weights1[i,j],3))))
            else:
                print("Omega1_z{}_x{} : {}".format(str(j),str(i),str(np.round(weights1[i,j],3))))
    for i in range(len(weights2)):
        print("Omega2_y_z{}  : {}".format(str(i),str(np.round(weights2[i,0],3))))
print("Output (estimated)  : {}".format(str(np.round(y.T[0,:],2))))
print("Output (target)     : {}".format(str(t_outs.T[0])))

#-----------------------------PLOT of RESULTS----------------------------------#

if 1:# EVOLUTION OF ERROR FUNCTION DURING LEARNING PROCESS
    plt.plot(Loss)
    plt.grid(True)
    plt.ylabel('Error Function')
    plt.xlabel('Iteration')
    plt.show()

if 1:# PLOT OF THE DATA and DECISION BOUNDARY
    x1f = [0,1]
    x2f = [0,1]
    x1t = [0,1]
    x2t = [1,0]
    plt.plot(x1f, x2f, 'ko', label = 'False')
    plt.plot(x1t, x2t, 'ro', label = 'True')
    x1 = np.array([i for i in np.arange(-0.5,5.5, 0.1)])

    for i in range(n_neurons):
        globals()['weights_'+str(i)] = []
        for j in range(len(weights1)):
            globals()['weights_'+str(i)].append(weights1[j,i])
        globals()['x2_'+str(i)] = dec_bound(x1, globals()['weights_'+str(i)])
        plt.plot( x1,globals()['x2_'+str(i)],label = 'Decision boundary {}'.format(str(i)))

    plt.grid(True)
    plt.title("")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.xlim([-1,2])
    plt.ylim([-1,2])
    plt.legend()
    plt.show()


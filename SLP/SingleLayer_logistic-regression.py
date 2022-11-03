from random import random, randrange
from math import exp, pi, log
from pylab import plot, ylabel, show
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import random

"""
Here we provide an example for the simplest possible network. A Single-Layer Network for a linearly separable dataset, with 2 classes where we use logistic discrimination and gradient descendent  to evaluate the decision boundary."""

################################################################################
#--------------------------SET OF FUNCTIONS------------------------------------#
################################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def dec_bound(x,weight):
    return -((weight[0]+weight[1]*x)/weight[2])

#------------------------------------------------------------------------------#
#                             GENERATING DATA SET                              #
#------------------------------------------------------------------------------#

"""The first row stand for the BIAS of the decision boundary and its entries are always activated."""

training_dataset = np.array([[1,1,1],
                             [1,1,0],
                             [1,0.75,1.25],
                             [1,0.5,0.5],
                             [1,0,1],
                             [1,0,0],
                             [1,1.25,1.25]])

training_outputs = np.array([[1,0,1,0,0,0,1]]).T

weights = np.random.random((3,1))

print("INITIAL random weights : ")
for i in range(len(weights)):
    if i==0:
         print("Bias   : {}".format(str(np.round(weights[i],3))))
    else:
        print("Omega{} : {}".format(str(i),str(np.round(weights[i],3))))
print("Output real : {}".format(str(np.round(training_outputs.T,1))))

################################################################################
#----------------------------------MAIN----------------------------------------#
################################################################################

""" We calculate the predicted output using randomized weights and the LOGISTIC SIGMOID function. Then we estimate the error from the real/expected training_output. Finally, using GRADIENT DESCENDENT, we correct the weights depending on the error made on predictions. The process is iterated N_iter times. The rate of learning is given by eta.
"""

eta    = 0.5   # learning rate
N_iter = 50000
start=time.time() 
for i in range(N_iter):
    input_layer = training_dataset
    outputs     = sigmoid(np.dot(input_layer,weights))
    error       = training_outputs - outputs
    adj         = error*sigmoid_derivative(outputs)*sigmoid_derivative(outputs)
    weights    += np.dot(input_layer.T,adj)
end=time.time()

################################################################################
# -------------------------------RESULTS---------------------------------------#
################################################################################

f_weights = weights
print("FINAL weights:")
for i in range(len(f_weights)):
    if i==0:
        print("Bias   : {}".format(str(np.round(f_weights[i],3))))
    else:    
        print("Omega{} : {}".format(str(i),str(np.round(f_weights[i],3))))

print("Output predicted : {}".format(str(np.round(outputs.T,1))))

print("Wall-time used  : {} sec".format(str((np.round(end-start,2)))))

x1=[0,1,0.5,0]
y1=[1,0,0.5,0]
x2=[0.75,1,1.25]
y2=[1.25,1,1.25]

x=np.array([i for i in np.arange(-0.25,1.5, 0.1)])
y=dec_bound(x,weights)

plt.plot( x,  y,       label = 'Decision boundary')
plt.plot(x1, y1, 'ko', label = 'False')
plt.plot(x2, y2, 'ro', label = 'True')
plt.grid(True)
plt.title("")
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.legend()
plt.show()

################################################################################
# -------------------------Testing a new data----------------------------------#
################################################################################

if 1:
    print('TESTING THE ALGORITHM')
    x1_test = float(input('Enter x1 coordinate:'))
    x2_test = float(input('Enter x2 coordinate:'))
    xn  = np.array([[1,x1_test,x2_test]])
    out = sigmoid(np.dot(xn,weights))
    print("Testing a new vector : ({},{})".format(str(xn[0][1]),str(xn[0][2])))
    print("Output  : {}".format(str(np.round(out[0],2))))
    if out>0.5:
        T   = 'r*'
        lbl = 'New data: True'
    if out<0.5:
        T   = 'k*'
        lbl = 'New data: False'
    plt.plot( x,  y,       label = 'Decision boundary')
    plt.plot(xn[0][1],xn[0][2],T,label=lbl)

plt.grid(True)
plt.title("")
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.legend()
plt.show()



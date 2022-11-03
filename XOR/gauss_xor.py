from random import random, randrange
from math import exp, pi, log
from pylab import plot, ylabel, show, fill
from matplotlib import pyplot as plt
from numpy import ones, zeros
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from numba import njit

################################################################################
#--------------------------SET OF FUNCTIONS------------------------------------#
################################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def dec_bound(x,weight):
    return -((weight[0]+weight[1]*x)/weight[2])

def get_rand_number(min_value, max_value):
    range  = max_value - min_value
    choice = random.uniform(min_value,max_value)
    return min_value + range*choice

# Gaussian distribution
def G(x,mu,sigma): 
    return np.exp(-np.power((x-mu),2)/(2*np.power(sigma,2)))*1/np.sqrt(2*pi*np.power(sigma,2))


# Random sampling over a probability distribution function
def get_rand_on_G(min_v, max_v,mu,sigma):
    range = max_v - min_v
    x_G   = np.random.uniform(min_v,max_v,1000) 
    N     = sum (G(x_G,mu,sigma))
    prob  = G(x_G,mu,sigma)/N                  
    rndm_pesata = (np.random.choice(x_G,1,p=prob))
    return rndm_pesata  

def Gaussian2D_Dataset(num_samples, mu_x , mu_y, sigma_x, sigma_y):
    x = []
    y = []
    for i in range(num_samples):
            r = get_rand_on_G(lower_bound, upper_bound,mu_x,sigma_x)
            x.append(r)
    x = np.asarray(x)
    for i in range(num_samples):
            r = get_rand_on_G(lower_bound, upper_bound,mu_y,sigma_y)
            y.append(r)
    y = np.asarray(y)
    dataset = np.append(x, y, axis=1)
    return dataset
lower_bound  = -8
upper_bound  =  8

def feed(Input, Target_outs, weights1, weights2, bias):
    z    = sigmoid(np.dot(Input,weights1)) 
    z    = np.append(bias, z, axis=1)
    y    = sigmoid(np.dot(z,weights2))

    loss_val   = 1/len(Input)*np.sum((y-Target_outs)**2) 

    return loss_val, y, z

################################################################################
#  INITIALIZATION DATA AND NEURAL NETWORK -------------------------------------#

num_samples_train = 150
num_samples_test  = 50
num_cluster = 4
num_neurons = 3   

eta    = 0.008      # Learning rate
N_iter = 5000       # Number of iterations

################################################################################
# FIRST DataSet ---------------------------------------------------------------#

mu11    = 1.5
sigma11 = 1.0
mu12    = 5.0
sigma12 = 0.8
num_samples = num_samples_train
x1 = Gaussian2D_Dataset(num_samples, mu11 , mu12, sigma11, sigma12)
t1 = np.array([ 1 for i in range(len(x1[:,0])) ]).reshape(-1,1)

x1_test = Gaussian2D_Dataset(num_samples_test, mu11 , mu12, sigma11, sigma12)
t1_test = np.array([ 1 for i in range(len(x1_test[:,0])) ]).reshape(-1,1)
# SECOND DataSet --------------------------------------------------------------#

mu21    = 3.0
sigma21 = 0.6
mu22    = 0.5
sigma22 = 0.8

x2 = Gaussian2D_Dataset(num_samples, mu21 , mu22, sigma21, sigma22)
t2 = np.array([ 0 for i in range(len(x2[:,0])) ]).reshape(-1,1)

x2_test = Gaussian2D_Dataset(num_samples_test, mu21 , mu22, sigma21, sigma22)
t2_test = np.array([ 0 for i in range(len(x2_test[:,0])) ]).reshape(-1,1)

################################################################################
#------------------------------- DATASET --------------------------------------#
################################################################################

x      = np.append(x1, x2, axis=0)
t_outs = np.append(t1, t2, axis=0)    # TARGET OUTPUT


x_test      = np.append(x1_test, x2_test, axis=0)    # DATASET test
t_outs_test = np.append(t1_test, t2_test, axis=0)    # TARGET OUTPUT test


c = num_cluster
if c>2 :
    mu_x    =  1.0
    sigma_x =  -2.55
    mu_y    = -4.0
    sigma_y =  1.5
    x3 = Gaussian2D_Dataset(num_samples, mu_x, mu_y, sigma_x, sigma_y)
    x      = np.append(x , x3, axis=0)
    t_outs = np.append(t_outs, t1, axis=0)
    
    x3_test = Gaussian2D_Dataset(num_samples_test, mu_x, mu_y, sigma_x, sigma_y)
    x_test      = np.append(x_test , x3_test, axis=0)
    t_outs_test = np.append(t_outs_test, t1_test, axis=0)
    if c ==4:
        mu_x    =  7.0
        sigma_x =  0.5
        mu_y    = -1.0
        sigma_y =  4.2
        x4 = Gaussian2D_Dataset(num_samples, mu_x, mu_y, sigma_x, sigma_y)
        x      = np.append(x , x4, axis=0)
        t_outs = np.append(t_outs, t1, axis=0)

        x4_test = Gaussian2D_Dataset(num_samples_test, mu_x, mu_y, sigma_x, sigma_y)
        x_test      = np.append(x_test , x4_test, axis=0)
        t_outs_test = np.append(t_outs_test, t1_test, axis=0)

bias  = np.array([ 1 for i in range(len(x[:,0])) ]).reshape(-1,1)
x     = np.append(bias, x, axis=1)                 # INPUT + BIAS

bias_test = np.array([ 1 for i in range(len(x_test[:,0])) ]).reshape(-1,1)
x_test    = np.append(bias_test, x_test, axis=1)  

n_input   = len(x[0])                              # 2 INPUT & 1 BIAS
n_neurons = num_neurons

#-------- Weights Initialization-----------------------------------------------#

np.random.seed(2)
weights1  = np.random.random((n_input, n_neurons)) -0.5# W1 x -> z
np.random.seed(1)
weights2  = np.random.random((n_neurons+1, 1))     -0.5# W2 z -> y

Loss      = []                                     # Initializing loss function
Accuracy  = []

Accuracy_test = []
Loss_test     = []

Loss_sum = []       
W1_opt   = np.array([[0,0],[0,0],[0,0]])
acc   = []
acc_t = []



################################################################################
#----------------------------------MAIN----------------------------------------#
################################################################################
from tqdm import tqdm
start=time.time() 
for i in tqdm(range(N_iter)):

    # TRAINING
    loss, y, z = feed(x,t_outs,weights1,weights2, bias)

    # TEST
    loss_t, y_t,_ = feed(x_test,t_outs_test,weights1,weights2, bias_test)

    Loss_test.append(loss_t)
    Loss.append(loss)

    # Back propagation 
    gradient_weights2 = 2*np.dot(z.T,(y-t_outs)*y*(1-y))   
    gradient_weights1 = 2 * np.dot(x.T, np.dot((y-t_outs)*y*(1-y),weights2.T) *z*(1-z) )

    weights2 += - eta*gradient_weights2                    # Weights corrections
    weights1 += - eta*gradient_weights1[:,1:n_neurons+1] 
   
    # ACCURACY ------------------------------------------------------

    acc   = (num_samples-sum(np.round(abs(y-t_outs),1)))/num_samples
    acc_t = (num_samples_test-sum(np.round(abs(y_t-t_outs_test),1)))/num_samples_test

    Accuracy.append(acc)
    Accuracy_test.append(acc_t)


if W1_opt[0,0]!=0:
    W1_opt = np.loadtxt(gE+'W1_opt.csv', delimiter=',')       
    

end=time.time()
print("Wall-time used  : {} sec".format(str((np.round(end-start,2)))))
print("Epoch : {}".format(str(N_iter)))
print("Accuracy Train : {}".format(str(acc)))
print("Accuracy Test  : {}".format(str(acc_t)))

################################################################################
# -------------------------------RESULTS---------------------------------------#
################################################################################


if 1: # EVOLUTION OF ACCURACY DURING LEARNING PROCESS
    plt.subplot(211)
    plt.plot(Accuracy     ,   'k', label = 'Train set')
    plt.plot(Accuracy_test, '--r', label = 'Test set')
    plt.grid(True)
    plt.ylabel('Accuracy %')
    plt.legend()


if 1: # EVOLUTION OF ERROR FUNCTION DURING LEARNING PROCESS
    plt.subplot(212)
    plt.plot(Loss ,   'k', label = 'Train set')
    plt.plot(Loss_test, '--r', label = 'Test set')
    plt.grid(True)
    plt.ylabel('Error Function')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if 1: # PLOT OF THE DATA and DECISION BOUNDARY
    plt.plot(x1[:,0], x1[:,1], 'ko', label = 'False')
    plt.plot(x2[:,0], x2[:,1], 'ro', label = 'True')
    if c > 2:    
        plt.plot(x3[:,0], x3[:,1], 'ko')
    if c == 4:
        plt.plot(x4[:,0], x4[:,1], 'ko')
    x1 = np.array([i for i in np.arange(-0.5,8, 0.1)])

    for i in range(n_neurons):
        globals()['weights_'+str(i)] = []
        for j in range(len(weights1)):
            globals()['weights_'+str(i)].append(weights1[j,i])
        globals()['x2_'+str(i)] = dec_bound(x1, globals()['weights_'+str(i)])
        plt.plot( x1,globals()['x2_'+str(i)],label = 'Decision boundary {}'.format(str(i)))
    if W1_opt[0,0]!=0:
        for i in range(n_neurons):
            globals()['weights_'+str(i)] = []
            for j in range(len(W1_opt)):
                globals()['weights_'+str(i)].append(W1_opt[j,i])
            globals()['x2_'+str(i)] = dec_bound(x1, globals()['weights_'+str(i)])
            plt.plot( x1,globals()['x2_'+str(i)],label = 'Test correction')

    plt.grid(True)
    plt.title("")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")


################################################################################
# -------------------------------Testing---------------------------------------#
################################################################################
if 1:

    dx = 0.2
    x  = np.array([i for i in np.arange( 0.5, 7.5, dx)],float)
    y  = np.array([i for i in np.arange(-6, 6.0, dx)],float)
    
    x,y = np.meshgrid(x,y)
    
    Data = []
    for j in range(len(y[:,0])):
        for i in range(len(x[0])):
            x12 = [x[j,i],y[j,i]]
            Data.append(x12)


    Data  = (np.asarray(Data)).reshape(-1,2)
    bias  = np.array([ 1 for i in range(len(Data[:,0])) ]).reshape(-1,1)
    x     = np.append(bias, Data, axis=1) # INPUT + BIAS

    z    = sigmoid(np.dot(x,weights1))
    z    = np.append(bias, z, axis=1)
    out  = sigmoid(np.dot(z,weights2))
    
    for i in range(len(out)-1):
    
        if out[i]>0.5:
            T   = 'k*'

        if out[i]<0.5:
            T   = 'r*'

        fill([Data[i,0]-dx/2, Data[i,0]+dx/2, Data[i,0]+dx/2, Data[i,0]-dx/2],
             [Data[i,1]-dx/2, Data[i,1]-dx/2, Data[i,1]+dx/2, Data[i,1]+dx/2],
             T ,alpha=0.2)

    plt.xlim([0, 7.5])
    plt.ylim([-6.0, 6.0])
    plt.show()






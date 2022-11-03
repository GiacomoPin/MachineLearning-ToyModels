from random import random, randrange
from math import exp, pi, log
from numpy import ones, zeros
from pylab import plot, ylabel, show
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import random
import kmeans1d

""" EXPECTATION-MAXIMIZATION ALGORITHM.
Here we provide an example for the EM-algorithm, developed in order to estimate the probability density from a given data set.
This technique is defined as semi-parametric and it combines both the advantages of parametric and non-parametric methods.
The estimated density function is represented as a linear combination of basis function (Gaussian distribution),
which are described by some parameters that can be optimized."""

################################################################################
#--------------------------SET OF FUNCTIONS------------------------------------#
################################################################################

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

#------------------------------------------------------------------------------#
#                             GENERATING DATA SET                              #
#------------------------------------------------------------------------------#
"""We create our fictional data set on which we are going to apply the EM-algorithm. 
The data set is created using 2 Gaussian distribution and we want to recover the values of the parameters of these density function. 
The algorithm can be easily generalized to higher dimensional data set and using different probability distribution."""
num_samples1 = 500
num_samples2 = 100
lower_bound  = 0
upper_bound  = 5

# FIRST Gaussian DISTRIBUTION

p      =  []
mu1    = 1.8
sigma1 = 0.4
for i in range(num_samples1):
        r = get_rand_on_G(lower_bound, upper_bound,mu1,sigma1)
        p.append(r)
p = np.asarray(p)

# SECOND Gaussian DISTRIBUTION

q      =  []
mu2    = 2.0
sigma2 = 1.2
for i in range(num_samples2):
        r = get_rand_on_G(lower_bound, upper_bound,mu2,sigma2)
        q.append(r)
q = np.asarray(q)


# Generating bar-plot using our Data set
if 0:
    h      = 0.02
    N_bars = int((upper_bound-lower_bound)/h)
    x      = np.array([i for i in np.arange(lower_bound,upper_bound,h)])
    isto   = np.zeros([N_bars],int)  
    for i in range(len(p)):
        for j in range(len(x)):
            check=abs(x[j]-p[i])/h
            if check<0.5:
                isto[j]+=1
            else:
                isto[j]+=0
if 0:
    R=np.random.random(1000)    
    plt.bar(x,isto, width=h, color='k',align='center', label='bar_j')
    plt.xlim([1.5,2.5])
    plt.legend()
    plt.show()

#-----------------------------------------------------------------------------#
# Merging of the two set of data
x = np.append(p,q)

if 0:
    plt.hist(x,bins=40,label='hist')
    plt.hist(p,bins=20,label='hist')
    plt.hist(q,bins=20,label='hist')
plt.show()

###                          Initialization                                  ###
#                          of the EM-algorithm                                 #
################################################################################
#----------------------------SET OF PARAMETERS---------------------------------#
################################################################################

N_steps  = 1500   # NUMBER OF STEPS OF THE ALGORITHM
M        = 2     # NUMBER OF J

cluster, centroids = kmeans1d.cluster(x,M)

c1 = []
c2 = []
for i in range(len(cluster)):
    if cluster[i]==0:
        c1.append(x[i])
    if cluster[i]==1:
        c2.append(x[i])
c1=np.asarray(c1)
c2=np.asarray(c2)

# INITIAL GUESS FOR prior probabilities
P2       = sum(cluster)/float(len(x))
P1       = 1 - P2

# INITIAL GUESS FOR mu
mu_init1 = centroids[0]   
mu_init2 = centroids[1]

# INITIAL GUESS FOR sigma
sigmaJ1 = np.sqrt(np.mean(np.power((c1-mu_init1),2)))
sigmaJ2 = np.sqrt(np.mean(np.power((c2-mu_init2),2)))



#------------------------------------------------------------------------------#
px_cond_J  = [0*i for i in range(M)]
nrm        = [0*i for i in range(M)]
pt_x       = [0*i for i in range(M)]
PJ_cond_x  = [0*i for i in range(M)]

mu_init = [mu_init1,    mu_init2]
sigma   = [ sigmaJ1,     sigmaJ2]
P       = [      P1,          P2]
print("Initial Guess mu: {}".format(str(mu_init)))
print("Initial Guess sigma: {}".format(str(sigma)))
print("Initial Guess PJ: {}".format(str(P)))
sigma_squared  = [0,0]

################################################################################
#----------------------------------MAIN----------------------------------------#
################################################################################

for i in range(N_steps):
    mu_init = mu_init
    sigma   = sigma
    P       = P
    for j in range(M):
        px_cond_J[j] = G(x,mu_init[j],sigma[j])
        pt_x[j]      = np.asarray(px_cond_J[j])*P[j]
    px = sum(pt_x)
    for j in range(M):                  
        PJ_cond_x[j]      = (px_cond_J[j]*P[j])/px
#   --- Updating parameters ---           
        mu_init[j]        = sum(PJ_cond_x[j]*x)/sum(PJ_cond_x[j])
        sigma_squared[j]  = (sum(PJ_cond_x[j]*np.power((x-mu_init[j]),2)))/sum(PJ_cond_x[j])
        sigma[j]          = np.sqrt(sigma_squared[j])
        P[j]              = sum(PJ_cond_x[j])/(len(x))


################################################################################
# -------------------------------RESULTS---------------------------------------#
################################################################################

if 1:
    plt.plot(x,PJ_cond_x[0],".", label="P(1|x)")
    plt.plot(x,PJ_cond_x[1],".", label="P(2|x)")
    plt.plot(x,PJ_cond_x[0]+PJ_cond_x[1], label="P(1|x)+P(2|x)")
    plt.title("Posterior probabilities", fontsize=15)
    plt.ylabel("P(J|x)")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    plt.show()

print("mu1 real = {} and mu1 estimated = {}".format(str(mu1), str(round(mu_init[0],2)) ))
print("mu2 real = {} and mu2 estimated = {}".format(str(mu2), str(round(mu_init[1],2)) ))

print("sigma1 real = {} and sigma1 estimated = {}".format(str(sigma1), str(round(sigma[0],2)) ))
print("sigma2 real = {} and sigma2 estimated = {}".format(str(sigma2), str(round(sigma[1],2)) ))

Ntot= float(num_samples1+num_samples2)
P1real= num_samples1/Ntot
P2real= num_samples2/Ntot

print("Prior P(j=1) = {} and estimated = {}".format(str(P1real), str(round(P[0],2) ) ))
print("Prior P(j=2) = {} and estimated = {}".format(str(P2real), str(round(P[1],2) ) ))


if 1:
    dx=0.01
    coord_x = np.array([i for i in np.arange(lower_bound,upper_bound,dx)],float)

    real_y    = G(coord_x,mu1,sigma1)*P1real+G(coord_x,mu2,sigma2)*P2real
    plt.plot(coord_x,real_y,'k',label="Real distribution")
    estim = G(coord_x,mu_init[0],sigma[0])*P[0]+G(coord_x,mu_init[1],sigma[1])*P[1]
    plt.plot(coord_x,estim,'r--',label="Estimated after {} steps".format(str(N_steps)))
    plt.title("Probability Density Estimation", fontsize=15)
    plt.ylabel("p(x)", fontsize=15);
    plt.xlabel("x"   , fontsize=15);
    plt.grid(True)
    plt.legend()
    plt.show()








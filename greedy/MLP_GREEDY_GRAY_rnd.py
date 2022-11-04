from random import random, randrange
from math import exp, pi, log
from pylab import plot, ylabel, show, fill
from matplotlib import pyplot as plt
from numpy import ones, zeros
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import random
import math
import threading
import queue
import copy

####################################################################################
####################################################################################
#                            Set  of functions
####################################################################################
####################################################################################


#-------------------------------------------------------------------------------
# ACTIVATION FUNCTION MLP -----------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def RELU(x):
    return np.maximum(0, x)


def Tanh_tab(n_bit_in, n_bit_weights, N_input, n_bit_out):
    minmax_matrix = np.array([-np.power(2, n_bit_in - 1), np.power(2, n_bit_in - 1) - 1]) * np.array(
        [[-np.power(2, n_bit_weights - 1)], [np.power(2, n_bit_weights - 1) - 1]])
    minimum_value = np.min(minmax_matrix) * (N_input + 1)
    maximum_value = np.max(minmax_matrix) * (N_input + 1)
    delta_value = abs(maximum_value - minimum_value) / 2

    x = np.array([i for i in np.arange(minimum_value, maximum_value)])
    normalization7_7 = delta_value / 50.0

    x = x / normalization7_7
    y = tanh(x) * 0.5 + 0.5
    sigm_tab = np.round(y * (np.power(2, n_bit_out) - 1)) - np.power(2, n_bit_out - 1)
    return sigm_tab, minimum_value

x = np.array([i for i in range(100)])
y = Tanh_tab(4,4,16,6)
plt.plot(y)
plt.show()
def Sigmoid_tab(n_bit_in, n_bit_weights, N_input, n_bit_out):
    minmax_matrix = np.array([-np.power(2, n_bit_in - 1), np.power(2, n_bit_in - 1) - 1]) * np.array(
        [[-np.power(2, n_bit_weights - 1)], [np.power(2, n_bit_weights - 1) - 1]])
    minimum_value = np.min(minmax_matrix) * (N_input + 1)
    maximum_value = np.max(minmax_matrix) * (N_input + 1)
    delta_value = abs(maximum_value - minimum_value) / 2

    x = np.array([i for i in np.arange(minimum_value, maximum_value)])
    normalization7_7 = delta_value / 10.0

    x = x / normalization7_7
    y = sigmoid(x)
    sigm_tab = np.round(y * (np.power(2, n_bit_out) - 1)) - np.power(2, n_bit_out - 1)
    return sigm_tab, minimum_value

#-------------------------------------------------------------------------------
# Other functions found in MLP_GREEDY() --------------------------------------
def activation_discrete(x, sigmoid_tab, minimum_value):
    x = x + abs(int(minimum_value))
    out = np.asarray(list(sigmoid_tab[x])).reshape(len(x[:, 0]), len(x[0, :]))
    out = out.astype(int)
    return out


def Loss_MSE(t_outs, y):
    loss1 = np.sum(np.power((y[0:N_letter] - t_outs[0:N_letter]) / np.power(2, N_bit_output - 1), 2)) / ndata
    loss2 = np.sum(
        np.power((y[N_letter:N_other + N_letter] - t_outs[N_letter:N_other + N_letter]) / np.power(2, N_bit_output - 1),2)) / ndata

    loss = (loss1 + loss2*0.5)
    return loss

def Loss_xentropy(t_outs,y):
    loss1 = np.sum(-t_outs[0:N_letter]*np.log(y[0:N_letter]))
    loss2 = np.sum(-t_outs[N_letter:N_other + N_letter]*np.log(y[N_letter:N_other + N_letter]))
    loss  = (loss1*3000+loss2*300)/3300    
    return loss

def Net_SA2(x, weights, tabulated_sigmoid1, minimum_value1, tabulated_sigmoid2, minimum_value2):
    # Hidden Layer
    z0 = np.dot(x, weights[0:N_input, :]) + weights[N_input, :] * (np.power(2, N_bit_input - 1) - 1)
    
    z = activation_discrete(z0, tabulated_sigmoid1, minimum_value1)
    # Output Layer
    y0 = np.dot(z, (weights[N_input + 1, :]).reshape(-1, 1)) + weights[N_input + 2, 0] * (
                np.power(2, N_bit_input - 1) - 1)
    y = activation_discrete(y0, tabulated_sigmoid2, minimum_value2)

    return z0, z, y0, y

#-------------------------------------------------------------------------------
# BIT DISCRETIZATION GRAY CODING -----------------------------------------------

def dec_to_4bit(number):
    binary = bin(number).replace('0b','')
    x = binary[::-1]
    while len(x)<4:
        x+='0'
    binary = x[::-1]
    return binary
    
def bit_flip(binary,i):
    binflip = copy.deepcopy(binary)
    if binflip[i]=='0' : binflip[i]='1'
    else : binflip[i]='0'
    return binflip

def dec_to_gray(int_number):
    gray = int_number ^ (int_number>>1)
    return gray

def gray_to_dec(int_number):
    inv = 0
    while (int_number):
        inv = inv^int_number
        int_number   = int_number>>1
    return inv

# DISCRETE MULTI LAYER PERCEPTRON w GREEDY optimization algorithm------------

def MLP_GREEDY(N_iter, N_input, n_neurons, N_output, N_bit_input, N_bit_weigh, N_bit_hidden, N_bit_output, dataset, X_test, t_outs_test, letter):

    # Weights initialization-------------------------------------------------
    if True:
        WEIGHTS = np.round(np.random.random((N_input + 1 + 1 + 1, n_neurons))*\
                                                (np.power(2, N_bit_weigh)-1))
        WEIGHTS = WEIGHTS.astype(int)
        np.savetxt(gOUT + 'WEIGHTS_I', WEIGHTS, delimiter=',')
    if False:
        WEIGHTS = np.loadtxt(gOUT+ 'WEIGHTS_F', delimiter=',')   
        WEIGHTS = WEIGHTS.astype(int)

    Loss      = []
    LOSS      = []
    Accuracy_test = []


    x = dataset[:, 0:N_input]
    t_outs = dataset[:, N_input: N_output + N_input] * (np.power(2,\
                         N_bit_output) - 1) - np.power(2, N_bit_output - 1)

    tabulated_sigmoid1, minimum_value1 = Tanh_tab(N_bit_input, N_bit_weigh,
                                                  N_input, N_bit_hidden)
    tabulated_sigmoid2, minimum_value2 = Tanh_tab(N_bit_hidden, N_bit_weigh,
                                                  n_neurons, N_bit_output)
    if 0:
        plt.plot(tabulated_sigmoid1)
        plt.grid(True)
        plt.show()

    random_walk=0
#------------------------------------------------------------------------------------ 
    for i in range(N_iter):  

# ----   ----   ----   ----   ----  ----  ----  ----  ----  ----  ----  ----  ---
        #1) First Feedforward
        if i==0:    
            z0,z,y0,y = Net_SA2(x,(WEIGHTS-8),
                                tabulated_sigmoid1,minimum_value1,
                                tabulated_sigmoid2,minimum_value2)
            loss_old  = Loss_MSE(t_outs,y)
            LOSS.append(loss_old)
            print(loss_old)


# ----   ----   ----   ----   ----  ----  ----  ----  ----  ----  ----  ----  ---
        #2) GREEDY Algorithm
        if False:  print('Iteration:',i) 

        RANDOM_WALK_row = []
        RANDOM_WALK_col = []
        NEW_WEIGHT      = []
        for row in range(len(WEIGHTS[:,0])):
            for column in range(len(WEIGHTS[0,:])):
            # Converting int to a 4bit number
            # Make a copy of weights on which perform a trial move.
                WEIGHTS_TRIAL = copy.deepcopy(WEIGHTS)
                WEIGHTS_TRIAL_bit = dec_to_4bit(\
                                    dec_to_gray(WEIGHTS_TRIAL[row,column]))
            # Converting the n-th weight into a 4bit string in gray code .  
            # Converting 4bit to int([0,15]), must be rescaled between [-8,7]
                
                if False: 
                    ''' TEST SULLA CODIFICA GRAY: 
                    Print the weight in and grey(decimal map)''' 
                    print('FLipping:')
                    print(WEIGHTS_TRIAL_bit, gray_to_dec(int(WEIGHTS_TRIAL_bit\
                                                                         ,2)))  
                    print('--------') 

                binary_list = list(WEIGHTS_TRIAL_bit)
                for bit in range(4):
                    flipped  = "".join(bit_flip(binary_list,bit))
                    w1elenew = gray_to_dec(int(flipped,2))
                    WEIGHTS_TRIAL[row,column]=w1elenew
 
#------------------------------------------------------------------------------
          ########## HERE I CALCULATE COST FUNCTION, IF IT'S BETTER: UPDATE!
                    z0n,zn,y0n,yn = Net_SA2(x,WEIGHTS_TRIAL-8,
                                tabulated_sigmoid1,minimum_value1,
                                tabulated_sigmoid2,minimum_value2)
     
                    # Cost function with the new configuration.
                    cost_new  = Loss_MSE(t_outs,yn)
                    Loss.append(cost_new)

                    # Saving the best value of the around.
                    if cost_new < loss_old:
                        WEIGHTS_best = copy.deepcopy(WEIGHTS_TRIAL) 
                        WEIGHTS_best[row,column] = w1elenew
                        loss_old = cost_new
                        random_walk = 0
                    if random_walk == 1 and cost_new == LOSS[i-1]:
                        RANDOM_WALK_row.append(row)
                        RANDOM_WALK_col.append(column)
                        NEW_WEIGHT.append(w1elenew)

        # IF THERE ARE MORE THAN ONE MINIMA: RANDOM PICK ONE
        if random_walk==1 and loss_old== LOSS[i-1]:
            WEIGHTS_best = copy.deepcopy(WEIGHTS_TRIAL) 
            rndchoice    = np.random.choice(len(RANDOM_WALK_row))
            WEIGHTS_best[RANDOM_WALK_row[rndchoice],RANDOM_WALK_col[rndchoice]] = NEW_WEIGHT[rndchoice]
        
        # Redefining the next configuration.
        WEIGHTS   = WEIGHTS_best
        z0,z,y0,y = z0n,zn,y0n,yn
        if random_walk==1 : print('rnd_wlk')
        if loss_old == LOSS[i-1] : random_walk=1
        if loss_old  < LOSS[i-1] : random_walk=0
        LOSS.append(loss_old)


# -----------TESTING--------------------------------------------------------#
        # TEST EVERY 10 ITERATIONS
        if i%10==0 or i == N_iter or i==0:
            z0_test, z_test, y0_test, y_test = Net_SA2(X_test, WEIGHTS-8,
                                      tabulated_sigmoid1, minimum_value1,
                                      tabulated_sigmoid2, minimum_value2)

            np.savetxt(gOUT + 'y_test{}_{}.csv'.format(str(letter), str(i)),\
                                                     y_test, delimiter=',')
            # orribile
            acc_TEST = (num_samples_test - np.sum(abs(np.round((y_test -\
                               t_outs_test) / np.power(2, N_bit_output),\
                                               0)))) / num_samples_test
            Accuracy_test.append(acc_TEST)

        if False:
            if i % 20 == 0 or i==0:
                plt.plot(y[150:450], 'o', label='Predicted')
                plt.plot(t_outs[150:450], '*', )
                plt.grid(True)
                plt.title('Loss={}'.format(str(loss_old)))
                plt.legend(bbox_to_anchor=(0.95, 0.4), loc=1, borderaxespad=0)
                plt.pause(0.005)
            plt.clf()

    np.savetxt(gOUT + 'WEIGHTS_F', WEIGHTS, delimiter=',')
    return y, y_test, Accuracy_test, Loss, LOSS


################################################################################
################################################################################
# --------------------------SET OF PARAMETERS-----------------------------------#
################################################################################
################################################################################

N_iter     = 200

N_letter   = 300   #  300
N_other    = 3000  # 3000
ndata_test = 5000  # 4000

ndata            = N_letter + N_other
num_samples      = ndata
num_samples_test = ndata_test

N_of_characters  = 26


################################################################################
################################################################################
#                        DISCRETE MLP PARAMETERS
################################################################################
################################################################################

c_out = 3 # change the number of neuron HERE
# Number of neurons will be 2 to the power of c_out

n_neurons = np.power(2, c_out) - 1  # (n_neurons + bias) be a power of 2

N_bit_weigh = 4

N_bit_input = 4

N_bit_hidden = 6

N_bit_output = 8

################################################################################
################################################################################
#                         Importing DATA 
################################################################################
################################################################################

# Defining all the paths
gDATA  = '/home/giacomo/Machine_Learning/Script/LETTER_RECOGNITION/30_11/data/'
gA     = '/home/giacomo/Machine_Learning/Script/LETTER_RECOGNITION/23_11/'
gOUT   = '/home/giacomo/Machine_Learning/Script/DISCRETE_MLP/LETTER_REC/OUTPUT/'
gIN    = '/home/giacomo/Machine_Learning/Script/DISCRETE_MLP/LETTER_REC/INPUT/'
gINPUT = '/home/giacomo/Machine_Learning/Script/DISCRETE_MLP/LETTER_REC/'
gbest  = '/home/giacomo/Machine_Learning/Script/DISCRETE_MLP/LETTER_REC/bestsofar/'

#                        Importing the INPUTS                                  #
#  Whole dataset is composed by 20000 samples.

cols = np.array([i for i in range(1, 17)])
x = np.loadtxt(gINPUT + 'letter-recognition.csv', delimiter=',', usecols=cols)
x = x - 8

N_input = len(x[0, :])
N_output = 1

# ------------------------------------------------------------------------------#
AlfaB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
         'J', 'K', 'L', 'M', 'N', 'O', 'P',
         'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# ------------------------------------------------------------------------------#

if 1:
    for letter in range(len(AlfaB[0:N_of_characters])):
        # IMPORTO I DATASET PER OGNI SINGOLA LETTERA

        globals()['dataset' + str(AlfaB[letter])] = np.loadtxt(gIN + 'dataset{}.csv'.format(str(AlfaB[letter])),
                                                               delimiter=',')
        globals()['dataset' + str(AlfaB[letter])] = globals()['dataset' + str(AlfaB[letter])].astype(int)

        # IMPORTO I TARGET (test) PER OGNI SINGOLA LETTERA

        globals()['t_outs_test' + str(AlfaB[letter])] = np.loadtxt(gIN + 't_outs_test{}.csv'.format(str(AlfaB[letter])),
                                                                   delimiter=',')[0:num_samples_test].reshape(-1, 1)
        globals()['t_outs_test' + str(AlfaB[letter])] = (globals()['t_outs_test' + str(AlfaB[letter])]) * (
                    np.power(2, N_bit_output) - 1) - np.power(2, N_bit_output - 1)
        globals()['t_outs_test' + str(AlfaB[letter])] = globals()['t_outs_test' + str(AlfaB[letter])].astype(int)

    print('-------------------------')
    print('------DATA UPLOADED------')

#                               TEST-SET

# Here we prepare the so-called Test-Set onto which the code will test the loss
# function and the accuracy of the network. The test set is quite general and
# we will use the same test set for all the 26 NNs. The Test-Set is recommended
# to be chosen big enough in order to be well representative of the whole
# universe of samples.

X_test = x[10000:10000+num_samples_test, :]
X_test = X_test.astype(int)

# ------------------------------------------------------------------------------#
# Test the train set
if 0:
    print('Check2')
    print('Max value of features (test): {}'.format(str(np.max(X_test))))
    print('Should be 7 !')
# ------------------------------------------------------------------------------#
# THIS SET WILL BE USED WHEN MERGING THE RESULT OF THE WHOLE SET OF NN (26)
TOUT_multi1 = np.loadtxt(gA + 'target_outs_test.csv', delimiter=',')
TOUT_multi2 = np.loadtxt(gA + 'target_outs.csv', delimiter=',')
TOUT_multi  = np.append(TOUT_multi2, TOUT_multi1, axis=0)

tout_multi  = (2 * TOUT_multi[0:num_samples_test, :] - 1) * (np.power(2, N_bit_output - 1))

# ------------------------------------------------------------------------------#



################################################################################
################################################################################
#                                  MAIN
################################################################################
################################################################################


if 1:
    start = time.time()
    y, y_test, Accuracy_test, Loss , LOSS = MLP_GREEDY(N_iter, N_input, 
                                      n_neurons,N_output, N_bit_input,
                                      N_bit_weigh, N_bit_hidden, N_bit_output,
                                      datasetB, X_test,
                                      t_outs_testB,'B')
    end = time.time()
    walltime = (np.round(end - start, 2))
    print('Walltime used for single NN : {}'.format(str(walltime)))


################################################################################
################################################################################
#                            RESULTS
################################################################################
################################################################################

#   ANALYSIS---------------------------- ------------------ ------------------
    Weight_init = np.loadtxt(gOUT+'WEIGHTS_I', delimiter=',')
    Weight_fin  = np.loadtxt(gOUT+'WEIGHTS_F', delimiter=',')
    diff = Weight_fin - Weight_init
    diff = diff.astype(int)
    print(diff)


    if 0:
        np.savetxt(gOUT + 'WEIGHTS_initial', Weight_init , delimiter=',')
        np.savetxt(gOUT + 'WEIGHTS_final'  ,  Weight_fin , delimiter=',')
    np.savetxt(gOUT + 'WEIGHTS_diff'   ,        diff , delimiter=',')


    dec_bound = np.array([0 for i in range(1000)])
    plt.plot(y_test[0:1000], 'o')
    plt.plot(t_outs_testC[0:1000], '*')
    plt.plot(dec_bound)
    plt.grid(True)
    plt.show()

    max_acc = np.array([1 for i in range(5)])
    plt.plot(Accuracy_test)
    plt.title('Accuratezza Test')
    plt.grid(True)
    plt.ylim([0, 1])
    plt.show()

    XXX = np.array([i*133*4 for i in range(0,len(LOSS))])
    plt.plot(Loss)
    plt.plot(XXX,LOSS,'-o')
    plt.title('LOSS train')
    plt.grid(True)
    plt.show()

#-----------------------------------------------------------------------------------#
if 0:

    tabulated_sigmoid1, minimum_value1 = Tanh_tab(N_bit_input, N_bit_weigh,
                                                  N_input, N_bit_hidden)
    tabulated_sigmoid2, minimum_value2 = Tanh_tab(N_bit_hidden, N_bit_weigh,
                                                  n_neurons, N_bit_output)
    Weight_fin = Weight_fin - 8
    z0 = np.dot(x, Weight_fin[0:N_input, :]) + Weight_fin[N_input, :]*\
                                         (np.power(2, N_bit_input - 1) - 1)

    plt.plot(tabulated_sigmoid1)
    plt.grid(True)
    plt.show()
    z0.astype(int)
    z = activation_discrete(z0, tabulated_sigmoid1, minimum_value1)
    
    # Output Layer
    y0 = np.dot(z, (Weight_fin[N_input + 1, :]).reshape(-1, 1)) + Weight_fin[N_input + 2, 0] * (
                np.power(2, N_bit_input - 1) - 1)
    y = activation_discrete(y0, tabulated_sigmoid2, minimum_value2)









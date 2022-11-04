from random import random, randrange
from math import exp, pi, log
from pylab import plot, ylabel, show, fill
from matplotlib import pyplot as plt
from numpy import ones, zeros
from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import math,copy,random, time, tqdm


from utils import sigmoid, tanh, RELU, Tanh_tab, Sigmoid_tab, activation_discrete, Loss_MSE, Loss_xentropy, Net_SA2
from quant_utils import  dec_to_4bit, bit_flip, dec_to_gray, gray_to_dec
from greedy_utils import MLP_GREEDY
    

################################################################################
# --------------------------SET OF PARAMETERS-----------------------------------#
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
#                        DISCRETE MLP PARAMETERS
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
#                         DATA PREPROCESSING
################################################################################
################################################################################

# Defining all the paths

path_target     = '/path_to_target/'
path_save   = '/path_to_save/'
path_sigle_letter    = '/path_to_dataset-letter/'
path_fulldataset = '/path_to_full_dataset/'

#                        Importing the INPUTS                                  #

#  Whole dataset is composed by 20000 samples.
cols = np.array([i for i in range(1, 17)])
x = np.loadtxt(path_fulldataset + 'letter-recognition.csv', delimiter=',', usecols=cols)
x = x - 8

N_input = len(x[0, :])
N_output = 1

# ------------------------------------------------------------------------------#
AlfaB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
         'J', 'K', 'L', 'M', 'N', 'O', 'P',
         'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# ------------------------------------------------------------------------------#

if 1:
    print(' [Uploading Datset]')
    for letter in tqdm(range(len(AlfaB[0:N_of_characters]))):

        # IMPORTO I DATASET PER OGNI SINGOLA LETTERA
        globals()['dataset' + str(AlfaB[letter])] = np.loadtxt(path_sigle_letter + 'dataset{}.csv'.format(str(AlfaB[letter])),
                                                               delimiter=',')
        globals()['dataset' + str(AlfaB[letter])] = globals()['dataset' + str(AlfaB[letter])].astype(int)

        # IMPORTO I TARGET (test) PER OGNI SINGOLA LETTERA
        globals()['t_outs_test' + str(AlfaB[letter])] = np.loadtxt(path_sigle_letter + 't_outs_test{}.csv'.format(str(AlfaB[letter])),
                                                                   delimiter=',')[0:num_samples_test].reshape(-1, 1)
        globals()['t_outs_test' + str(AlfaB[letter])] = (globals()['t_outs_test' + str(AlfaB[letter])]) * (
                    np.power(2, N_bit_output) - 1) - np.power(2, N_bit_output - 1)
        globals()['t_outs_test' + str(AlfaB[letter])] = globals()['t_outs_test' + str(AlfaB[letter])].astype(int)


    print(' [Uploaded]')

#                               TEST-SET

# Here we prepare the so-called Test-Set onto which the code will test the loss
# function and the accuracy of the network. The test set is quite general and
# we will use the same test set for all the 26 NNs. The Test-Set is recommended
# to be chosen big enough in order to be well representative of the whole
# universe of samples.

X_test = x[10000:10000+num_samples_test, :]
X_test = X_test.astype(int)


# ------------------------------------------------------------------------------#
# THIS SET WILL BE USED WHEN MERGING THE RESULT OF THE WHOLE SET OF NN (26)
TOUT_multi1 = np.loadtxt(path_target + 'target_outs_test.csv', delimiter=',')
TOUT_multi2 = np.loadtxt(path_target + 'target_outs.csv', delimiter=',')
TOUT_multi  = np.append(TOUT_multi2, TOUT_multi1, axis=0)

tout_multi  = (2 * TOUT_multi[0:num_samples_test, :] - 1) * (np.power(2, N_bit_output - 1))

# ------------------------------------------------------------------------------#



################################################################################
################################################################################
#                                  MAIN
################################################################################
################################################################################


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


if 0:
    #   ANALYSIS---------------------------- ------------------ ------------------
    Weight_init = np.loadtxt(path_save+'WEIGHTS_I', delimiter=',')
    Weight_fin  = np.loadtxt(path_save+'WEIGHTS_F', delimiter=',')
    diff = Weight_fin - Weight_init
    diff = diff.astype(int)
    print(diff)


    if 0:
        np.savetxt(path_save + 'WEIGHTS_initial', Weight_init , delimiter=',')
        np.savetxt(path_save + 'WEIGHTS_final'  ,  Weight_fin , delimiter=',')
    np.savetxt(path_save + 'WEIGHTS_diff'   ,        diff , delimiter=',')


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











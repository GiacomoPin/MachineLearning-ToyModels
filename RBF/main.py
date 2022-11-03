# Giacomo Pinali
# 06/2021
#

from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
import copy
import random

from numba import njit, prange
import time
import pandas as pd
from KMEAN import get_distance, kmeans
from RBF   import RBF

print('MNIST DIGIT RECOGNITION PROBLEM.')
print('Uploading data..')

train  = 1500
test   = 200


N_DATA    = int(train + test)
gDATA     = ''
cols      = np.array([ i for i in range(1,785) ])
full_DATASET   = np.loadtxt(gDATA+'mnist_train.csv', delimiter=',',\
                                      skiprows=1, max_rows= N_DATA,\
                                                     usecols = cols)
lab_cols = 0
full_DATASET_LABEL = np.loadtxt(gDATA+'mnist_train.csv', delimiter=',',\
                                          skiprows=1, max_rows= N_DATA,\
                                                     usecols = lab_cols)

train_img = full_DATASET[0:0+train,:]
train_label = full_DATASET_LABEL[0:0+train].astype(int)


test_img = full_DATASET[train:train+test,:]
test_label = full_DATASET_LABEL[train:train+test].astype(int)

print(' [Uploaded]')


RBF_CLASSIFIER = RBF(train_img, train_label, test_img, test_label,\
                                   num_of_classes=10, k=100, std_from_clusters=False)

Centroids, Weights, Std_list = RBF_CLASSIFIER.fit()


############################################################################################################
############################################################################################################

def rbf(x, c, s):
    distance = get_distance(x, c)
    return 1 / np.exp(-distance / s ** 2)

def get_rbf_array(X, centroids, std_list):
    RBF_list = []
    for x in X:
        RBF_list.append([rbf(x, c, s) for (c, s) in zip(centroids, std_list)])
    return np.array(RBF_list)

if True:

    tX = test_img[100:120,:]
    ty = test_label[100:120]
    RBF_test = get_rbf_array(tX, Centroids, Std_list)

    pred_ty = np.matmul(RBF_test, Weights)

    pred_ty = np.array([np.argmax(x) for x in pred_ty])
    diff = pred_ty - ty
    
    for i in range(9):
        img = tX[i,:].reshape(28,28)
        plt.imshow(img)
        plt.title('Predicted : {}'.format(str(pred_ty[i])))
        plt.show()






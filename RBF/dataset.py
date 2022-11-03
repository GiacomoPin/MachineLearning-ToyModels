from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
import copy
import random
from sklearn.preprocessing import MinMaxScaler
from numba import njit, prange
import time
import pandas as pd
from KMEAN import get_distance, kmeans
from RBF   import RBF

print('MNIST DIGIT RECOGNITION PROBLEM.')
print('Uploading data..')

train  = 2000
test   = 1000

N_DATA = int(train + test)
gDATA  = '/home/giacomo/Machine_Learning/Script/CNN/mnist_digit/'

data   =  pd.read_csv(gDATA+'mnist_train.csv', nrows=N_DATA)

'''CREATING DATASET'''
label = data.filter(['label'])
dataset_label = label.values.reshape(-1)

dataset = data.drop(['label'],axis=1)
dataset = dataset.values

if True:
    '''SCALING DATA'''
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

'''Splitting dataset: train/test'''
train_img = dataset[0:0+train,:]
train_label = dataset_label[0:0+train].astype(int)

test_img = dataset[train:train+test,:]
test_label = dataset_label[train:train+test].astype(int)

print('Uploaded')
print(train_img)
#################################################################################
#################################################################################
'''MODEL RBF'''

RBF_CLASSIFIER = RBF(train_img, train_label, test_img, test_label,\
                                   num_of_classes=10, k=100, std_from_clusters=False)

Centroids, Weights, Std_list = RBF_CLASSIFIER.fit()




def rbf(x, c, s):
    distance = get_distance(x, c)
    return 1 / np.exp(-distance / s ** 2)

def rbf_list(X, centroids, std_list):
    RBF_list = []
    for x in X:
        RBF_list.append([rbf(x, c, s) for (c, s) in zip(centroids, std_list)])
    return np.array(RBF_list)

if True:

    tX = test_img[100:120,:]
    ty = test_label[100:120]
    RBF_list_tst = rbf_list(tX, Centroids, Std_list)

    pred_ty = RBF_list_tst @ Weights

    pred_ty = np.array([np.argmax(x) for x in pred_ty])
    diff = pred_ty - ty
    
    for i in range(9):
        img = tX[i,:].reshape(28,28)
        plt.imshow(img)
        plt.title('Predicted : {}'.format(str(pred_ty[i])))
        plt.show()



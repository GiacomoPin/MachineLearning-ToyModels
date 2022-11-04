import os
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

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



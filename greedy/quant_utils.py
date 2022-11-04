import os
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

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



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:33:08 2019

@author: ragnoletto
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#matplotlib inline


def loadData():
    train_data = sio.loadmat('data.nosync/train_32x32.mat')
    test_data = sio.loadmat('data.nosync/test_32x32.mat')
    #extra_data = sio.loadmat('data.nosync/extra_32x32.mat')

    # access to the dict
    x_train = train_data['X']
    y_train = train_data['y']
    
    x_test = test_data['X']
    y_test = test_data['y']
    
    #x_extra = extra_data['X']
    #y_extra = extra_data['y']


    #image_ind = 10
    # show sample
    #plt.imshow(x_train[:,:,:,image_ind])
    #plt.show()
    #print(y_train[image_ind])
    
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = loadData()

x=True
image_ind=0
while x:
    if y_train[image_ind] == 5:
        plt.imshow(x_train[:,:,:,image_ind])
        plt.show()
        x=False
    
    image_ind +=1
print(type(x_train))
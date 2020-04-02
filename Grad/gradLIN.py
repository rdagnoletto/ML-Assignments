#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:21:35 2019

@author: ragnoletto
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import scipy.io as sio
import pickle as pkl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def loadData(randIndx=None):
    train_data = sio.loadmat('data.nosync/train_32x32.mat')
    test_data = sio.loadmat('data.nosync/test_32x32.mat')

    
    X = np.concatenate((np.moveaxis(train_data['X'],-1,0),np.moveaxis(test_data['X'],-1,0)))/255.0
    
    Y = np.concatenate((train_data['y'],test_data['y']))
    
    if randIndx is None:
        np.random.seed(521)
        randIndx = np.arange(len(Y))
        np.random.shuffle(randIndx)
    
    X = X[randIndx]
    Y = Y[randIndx]
    
#    x_train, y_train = X[:30000], Y[:30000]
#    x_val, y_val = X[30000:40000], Y[30000:40000]
#    x_test, y_test = X[40000:50000], Y[40000:50000]

    x_train, y_train = X[:60000], Y[:60000]
    x_val, y_val = X[60000:80000], Y[60000:80000]
    x_test, y_test = X[80000:], Y[80000:]
    
#    x_train, y_train = X[:6000], Y[:6000]
#    x_val, y_val = X[6000:8000], Y[6000:8000]
#    x_test, y_test = X[8000:10000], Y[8000:10000]
    
#    x_train, y_train = X[:600], Y[:600]
#    x_val, y_val = X[600:800], Y[600:800]
#    x_test, y_test = X[800:1000], Y[800:1000]
    
    return x_train, y_train, x_val, y_val, x_test, y_test, randIndx


def convertOneHot(trainTarget, valTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newval = np.zeros((valTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    # Target data is 1-10 so we do -1
    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]-1] = 1
    for item in range(0, valTarget.shape[0]):
        newval[item][valTarget[item]-1] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]-1] = 1
    return newtrain, newval, newtest



def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target
    
def accuracy(target, prediction):
    # Target data is 1-10 so we do -1
    prediction = np.argmax(prediction, axis=1).reshape((-1,1)) + 1
    equal_by_element = np.equal(target,prediction)
    return np.mean(equal_by_element)


x_train, y_train, x_val, y_val, x_test, y_test, randIndx = loadData()


x_train_flat = x_train.flatten().reshape(len(x_train), 32*32*3)

print(x_train_flat.shape)
classifier = tf.estimator.LinearClassifier(feature_columns=x_train_flat)
classifier.train(train_inpf)

result = classifier.evaluate(test_inpf)

clear_output()
    
    

fig, ax = plt.subplots()
ax.plot(trainError)
ax.set(xlabel='Iterations', ylabel='CE Loss', title='Loss from the Training Data')
ax.grid()
plt.show() 

fig, ax = plt.subplots()
ax.plot(valError)
ax.set(xlabel='Iterations', ylabel='CE Loss',title='Loss from the Validation Data')
ax.grid()
plt.show() 

fig, ax = plt.subplots()
ax.plot(trainAcc)
ax.set(xlabel='Iterations', ylabel='Accuracy',title='Accuracy for the Training Data')
ax.grid()
plt.show() 

fig, ax = plt.subplots()
ax.plot(valAcc)
ax.set(xlabel='Iterations', ylabel='Accuracy', title='Accuracy for the Validation Data')
ax.grid()
plt.show() 


print("Run Time: %.2f seconds" % (end_time - start_time))
print("Final Training Error: %.5f:" % (errTrain)) 
print("Final Validation Error: %.5f:" % (errVal)) 
print("Final Testing Error: %.5f:" % (errTest)) 

TrainAcc = accuracy(y_train,pred_train)
ValAcc = accuracy(y_val,pred_val)
TestAcc = accuracy(y_test,pred_test)

print("Final Training Accuracy: %.5f:" % (TrainAcc))
print("Final Validation Accuracy: %.5f:" % (ValAcc))
print("Final Testing Accuracy: %.5f:" % (TestAcc))
 








   
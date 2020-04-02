#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:43:39 2019

@author: ragnoletto
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    # input and output is NxDout
    exp = np.exp(x- np.amax(x, axis=1,keepdims=True))
    expSum = np.sum(exp, axis = 1, keepdims=True)
    sigma = exp/expSum
    return sigma

def computeLayer(X, W, b):
    # X is NxDin, W is DinxDout, Wx is NxDout, b is 1xDout, Output is NxDout
    return np.matmul(X,W) + b
    
def averageCE(target, prediction):
    # Target is NxC and Prediction is NxC
    # axis = 1 sums along the columns --> over k
    #s = softmax(prediction)
    #return -np.mean(np.sum(target*np.log(s), axis = 1)) 
    return -np.sum(target*np.log(prediction))/target.shape[0]

def accuracy(target, prediction):
    prediction = np.argmax(prediction, axis=1)
    print(prediction)
    equal_by_element = np.equal(target,prediction)
    #print(target.shape, prediction.shape)
    return np.mean(equal_by_element)
    
def gradCE(target, prediction):
    # Target is NxC and Prediction is NxC
    # Divides them, then sum all N
    # Output is 1xC
    # s = softmax(prediction)
    return -np.mean(np.divide(target,prediction), axis=0,keepdims=True)
    
def gradSM(s2):
    s2 = np.mean(s2,axis=0,keepdims=True)
    sm2 = softmax(s2)
    grad = -np.matmul(np.transpose(sm2),sm2)
    for i in range(s2.shape[1]):
        grad[i][i] = sm2[0][i]*(1-sm2[0][i])
        
    return grad

def calcGradient(x0,s1,x1,s2,x2,y,b1,w1,b2,w2):
    #s2_exp = np.exp(s2)
    # 1xd2 or 1xC
    # ds2 = np.multiply(gradCE(y,x2),np.mean((s2_exp/np.sum(s2_exp)),axis=0,keepdims=True))
    
    #1x10 times 10x10 = 1x10
    ds2 = np.matmul(gradCE(y,x2),gradSM(s2))
    
    db2 = ds2
    
    # d1x1 or HiddenUnitsx1
    avgX1 = np.transpose(np.mean(x1,axis=0,keepdims=True))

    # 1000x1 times 1x10 = 1000x10
    dw2 = np.matmul(avgX1,ds2)
    
    # 1x10 times 10x1000 = 1x1000
    w2ds2 = np.matmul(ds2,np.transpose(w2))
    # 1x1000
    dReluAvg = np.mean(np.heaviside(s1,1),axis=0)
    # 1x1000 times 1x1000 = 1x1000 (pointwise)
    ds1 = np.multiply(w2ds2,dReluAvg)
    
    # 1x1000
    db1 = ds1
    
    # x0 is Nx784, then mean: 1x784, then transpose: 784x1
    avgX0 = np.transpose(np.mean(x0,axis=0,keepdims=True))
    # 784x1 times 1x1000 = 784x1000
    dw1 = np.matmul(avgX0,ds1)
    
    
    
    return dw1, db1, dw2, db2
    
    
def learn(X,y,b1,v1,w1,b2,v2,w2,epochs,gamma,alpha):    
    # Back Prop
    h = 0.001
    v1_old, v2_old = v1, v2
    for i in range(epochs):
        # Forward Prop
        s1 = computeLayer(X,w1,b1)
        x1 = relu(s1)
        s2 = computeLayer(x1,w2,b2)
        x2 = softmax(s2)
        
        # Back Prop
        dw1, db1, dw2, db2 = calcGradient(X,s1,x1,s2,x2,y,b1,w1,b2,w2)    
         
        if i % 10 == 0:
            i1=random.randint(0,999)
            j1=random.randint(0,9)
            w2_plus = w2
            w2_plus[i1][j1] = w2[i1][j1] + h
            s2_plus = computeLayer(x1,w2_plus,b2)
            x2_plus = softmax(s2_plus)
            
            w2_min = w2
            w2_min[i1][j1] = w2[i1][j1] - h
            s2_min = computeLayer(x1,w2_min,b2)
            x2_min = softmax(s2_min)
            
            
            CE_min = averageCE(y,x2_min)
            CE_plus = averageCE(y,x2_plus)
            
            dw2_test = (CE_plus - CE_min)/(2*h)
            
            diff = abs(dw2_test-dw2[i1][j1])
            print(i,diff,(dw2_test), (dw2[i1][j1]))
        
        
        #v2 = gamma*v2_old + alpha*dw2
        v2 = gamma*v2_old + (1-gamma)*dw2
        #w2 = w2 - v2
        w2 = w2 - alpha*dw2
        b2 = b2 - alpha*db2
        
        #v1 = gamma*v1_old + alpha*dw1
        v1 = gamma*v1_old + (1-gamma)*dw1
        #w1 = w1 - v1
        w2 = w2 - alpha*dw2
        b1 = b1 - alpha*db1
        
        v1_old, v2_old = v1, v2
        
    return b1,v1,w1,b2,v2,w2
 
       
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()    
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget) 
trainData = trainData.reshape(-1,28*28)
validData = validData.reshape(-1,28*28)
testData = testData.reshape(-1,28*28)



hiddenunits = 1000
epochs = 200
classes = 10
gamma = 0.99
alpha = 0.005

d0 = len(trainData[1,:])

mu = 0
sigma1 = np.sqrt(2)/np.sqrt(hiddenunits+d0)

v1 = 1e-5*np.ones((d0,hiddenunits))
b1 = 1e-5*np.ones((1,hiddenunits))
#b1 = sigma1*np.ones((1,hiddenunits))
#b1 = np.random.normal(mu,sigma1,(1,hiddenunits))
w1 = np.random.normal(mu,sigma1,(d0,hiddenunits))

sigma2 = np.sqrt(2)/np.sqrt(hiddenunits+classes)

v2 = 1e-5*np.ones((hiddenunits,classes))
b2 = 1e-5*np.ones((1,classes))
#b2 = sigma2*np.ones((1,classes))
#b2 = np.random.normal(mu,sigma2,(1,classes))
w2 = np.random.normal(mu,sigma2,(hiddenunits,classes))


tb1,tv1,tw1,tb2,tv2,tw2 = learn(trainData,newtrain,b1,v1,w1,b2,v2,w2,epochs,gamma,alpha)

s1 = computeLayer(trainData,tw1,tb1)
x1 = relu(s1)
s2 = computeLayer(x1,tw2,tb2)
x2 = softmax(s2)
print("Final Training CE:", averageCE(newtrain,x2))
print("Final Training Accuracy:", accuracy(trainTarget, x2))

s1 = computeLayer(validData,tw1,tb1)
x1 = relu(s1)
s2 = computeLayer(x1,tw2,tb2)
x2 = softmax(s2)
print("Final Validation CE:", averageCE(newvalid,x2))
print("Final Training Accuracy:", accuracy(validTarget, x2))

s1 = computeLayer(testData,tw1,tb1)
x1 = relu(s1)
s2 = computeLayer(x1,tw2,tb2)
x2 = softmax(s2)
print("Final Testing CE:", averageCE(newtest,x2))
print("Final Training Accuracy:", accuracy(testTarget, x2))















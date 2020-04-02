#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:17:58 2019

@author: ragnoletto
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import time
import matplotlib.pyplot as plt

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
    exp = np.exp(x)
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
    target = np.argmax(target, axis=1)
    equal_by_element = np.equal(target,prediction)
    return np.mean(equal_by_element)
    
def gradCE(target, prediction):
    # Target is NxC and Prediction is NxC
    # Divides them, then sum all N
    # Output is 1xC
    # s = softmax(prediction)
    #return -np.mean(np.divide(target,prediction), axis=0,keepdims=True)
    return -np.divide(target,prediction)
    
def gradSM(s2):
    sm2 = softmax(s2)

    #Nx10x10
    grad = -np.einsum('ij,ik->ijk', sm2, sm2)
    np.einsum('jii->ji', grad)[:] +=sm2
    #empty = np.einsum('ij,ij->ijj', sm2, sm2)

    
    #grad = -np.matmul(np.transpose(sm2),sm2) + np.diag(np.squeeze(sm2))
        
    return grad

def calcGradient(x0,s1,x1,s2,x2,Y,b1,w1,b2,w2):
  
    # Nx10
    ds2 = np.einsum('ij,ijk->ik', gradCE(Y,x2), gradSM(s2))
    # 1x10
    db2 = np.mean(ds2, axis=0,keepdims=True)
    #1000x10
    #dw2 = np.einsum('ij,ik->jk',x1,ds2)/(x1.shape[0])
    dw2 = np.matmul(np.transpose(x1),ds2)/(x1.shape[0])

    # Nx10 times 10x1000 = Nx1000
    w2ds2 = np.matmul(ds2,np.transpose(w2))
    
    # Nx1000
    dRelu = np.heaviside(s1,1)
    
    # Nx1000 times Nx1000 = Nx1000 (pointwise)  
    ds1 = np.multiply(w2ds2,dRelu)

    # 1x1000
    db1 = np.mean(ds1, axis=0,keepdims=True)
    
    #dw1 = np.einsum('ij,ik->jk',x0,ds1)/(x0.shape[0])
    dw1 = np.matmul(np.transpose(x0),ds1)/(x0.shape[0])

    return dw1, db1, dw2, db2

def graph(Loss,Accuracy,i,X2,Y,vX,vY,tX,tY,w1,b1,w2,b2):
    
    predTrain = X2
    
    vs1 = computeLayer(vX,w1,b1)
    vx1 = relu(vs1)
    vs2 = computeLayer(vx1,w2,b2)
    predValid = softmax(vs2)
    
    ts1 = computeLayer(tX,w1,b1)
    tx1 = relu(ts1)
    ts2 = computeLayer(tx1,w2,b2)
    predTest = softmax(ts2)
    
    Loss[0][i] = averageCE(Y,predTrain)
    Loss[1][i] = averageCE(vY,predValid)
    Loss[2][i] = averageCE(tY,predTest)
    
    Accuracy[0][i] = accuracy(Y,predTrain)
    Accuracy[1][i] = accuracy(vY,predValid)
    Accuracy[2][i] = accuracy(tY,predTest)
    
    
def learn(X,Y,b1,v1,w1,b2,v2,w2,epochs,gamma,alpha,vX,vY,tX,tY):    

    Loss,Accuracy = np.zeros((3,epochs)),np.zeros((3,epochs))
    vb1_old, vb2_old = b1, b2
    v1_old, v2_old = v1, v2
    for i in range(epochs):
        # Forward Prop
        s1 = computeLayer(X,w1,b1)
        x1 = relu(s1)
        s2 = computeLayer(x1,w2,b2)
        x2 = softmax(s2)
        
        graph(Loss,Accuracy,i,x2,Y,vX,vY,tX,tY,w1,b1,w2,b2)
        
        # Back Prop
        dw1, db1, dw2, db2 = calcGradient(X,s1,x1,s2,x2,Y,b1,w1,b2,w2)    
         
        if i % 10 == 0:
            print("Epoch: %d" % i,"CE: %.3f" % averageCE(Y,x2))
        
        
        v2 = gamma*v2_old + alpha*dw2
        w2 = w2 - v2
        
        vb2 = gamma*vb2_old + alpha*db2
        b2 = b2 - vb2
        
        v1 = gamma*v1_old + alpha*dw1
        w1 = w1 - v1

        vb1 = gamma*vb1_old + alpha*db1
        b1 = b1 - vb1
        
        vb1_old, vb2_old = vb1, vb2
        v1_old, v2_old = v1, v2

        
    return b1,v1,w1,b2,v2,w2,Loss,Accuracy
 
       
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()    
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget) 
trainData = trainData.reshape(-1,28*28)
validData = validData.reshape(-1,28*28)
testData = testData.reshape(-1,28*28)



hiddenunits = 1000
epochs = 125
classes = 10
gamma = 0.95
alpha = 1-gamma
alpha = 0.05

d0 = len(trainData[1,:])

mu = 0
sigma1 = np.sqrt(2)/np.sqrt(hiddenunits+d0)

v1 = 1e-5*np.ones((d0,hiddenunits))
b1 = 1e-5*np.ones((1,hiddenunits))
w1 = np.random.normal(mu,sigma1,(d0,hiddenunits))

sigma2 = np.sqrt(2)/np.sqrt(hiddenunits+classes)

v2 = 1e-5*np.ones((hiddenunits,classes))
b2 = 1e-5*np.ones((1,classes))
w2 = np.random.normal(mu,sigma2,(hiddenunits,classes))

start = time.time()
tb1,tv1,tw1,tb2,tv2,tw2,Loss,Accuracy = learn(trainData,newtrain,b1,v1,w1,b2,v2,w2,epochs,
                                gamma,alpha,validData,newvalid,testData,newtest)
end = time.time()
print("Run Time (seconds): ",end - start)

s1 = computeLayer(trainData,tw1,tb1)
x1 = relu(s1)
s2 = computeLayer(x1,tw2,tb2)
x2 = softmax(s2)
print("Final Training CE:", averageCE(newtrain,x2))
print("Final Training Accuracy:", accuracy(newtrain, x2))

vs1 = computeLayer(validData,tw1,tb1)
vx1 = relu(vs1)
vs2 = computeLayer(vx1,tw2,tb2)
vx2 = softmax(vs2)
print("Final Validation CE:", averageCE(newvalid,vx2))
print("Final Validation Accuracy:", accuracy(newvalid, vx2))

ts1 = computeLayer(testData,tw1,tb1)
tx1 = relu(ts1)
ts2 = computeLayer(tx1,tw2,tb2)
tx2 = softmax(ts2)
print("Final Testing CE:", averageCE(newtest,tx2))
print("Final Testing Accuracy:", accuracy(newtest, tx2))


# Training data plots
plt.plot(Loss[0])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('CE Loss')
plt.title('Training CE Loss h=%d' % hiddenunits)
plt.axis([0, len(Loss[0]), 0, 4])

plt.show()

plt.plot(Accuracy[0])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy h=%d' % hiddenunits)
plt.axis([0, len(Accuracy[0]), 0, 1])

plt.show()

# Validation data plots
plt.plot(Loss[1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('CE Loss')
plt.title('Validation CE Loss h=%d' % hiddenunits)
plt.axis([0, len(Loss[1]), 0, 4])

plt.show()

plt.plot(Accuracy[1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy h=%d' % hiddenunits)

plt.axis([0, len(Accuracy[1]), 0, 1])

plt.show()

# Testing data plots
plt.plot(Loss[2])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('CE Loss')
plt.title('Testing CE Loss h=%d' % hiddenunits)
plt.axis([0, len(Loss[2]), 0, 4])

plt.show()

plt.plot(Accuracy[2])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy h=%d' % hiddenunits)

plt.axis([0, len(Accuracy[2]), 0, 1])

plt.show()

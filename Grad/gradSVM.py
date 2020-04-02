#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:15:47 2019

@author: ragnoletto
"""

import tensorflow as tf
import tflearn
import random
#import console
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
    
    x_train, y_train = X[:30000], Y[:30000]
    x_val, y_val = X[30000:40000], Y[30000:40000]
    x_test, y_test = X[40000:50000], Y[40000:50000]

#    x_train, y_train = X[:60000], Y[:60000]
#    x_val, y_val = X[60000:80000], Y[60000:80000]
#    x_test, y_test = X[80000:], Y[80000:]
    
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


def get_input_fn(images, labels, batch_size=32, capacity=60000, min_after_dequeue=3000):

  def _input_fn():
    images_batch, labels_batch = tf.train.shuffle_batch(
        tensors=[images, labels],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True,
        num_threads=4)
    features_map = {'pixels': images_batch, 'example_id':tf.constant(np.array(['%d' % random.randint(0,1) for i in range(batch_size)]))}
    return features_map, labels_batch

  return _input_fn

#tf.constant(map(lambda x: str(x + 1), np.arange(batch_size)))}
pixels = tf.contrib.layers.real_valued_column('pixels', dimension=32*32*3)

#example_id = numpy.array(['%d' % random.randint(0,9) for i in range(len(x_train))])

estimator = tf.contrib.learn.SVM(
    example_id_column='example_id',
    feature_columns=[pixels],
    l2_regularization=10.0)

x_train, y_train, x_val, y_val, x_test, y_test, randIndx = loadData()

#y_train -= 1
x_train_flat = x_train.flatten().reshape(len(x_train), 32*32*3)
#y_val -= 1
x_val_flat = x_val.flatten().reshape(len(x_val), 32*32*3)

indices = [i for i, x in enumerate(y_train) if x == 1 or x == 10]
x_train_flat = np.take(x_train_flat,indices,axis=0)
y_train = np.take(y_train,indices,axis=0)

y_train[y_train==10] = 0

indices = [i for i, x in enumerate(y_val) if x == 1 or x == 10]
x_val_flat = np.take(x_val_flat,indices,axis=0)
y_val = np.take(y_val,indices,axis=0)

y_val[y_val==10] = 0

train_input_fn = get_input_fn(x_train_flat, y_train, batch_size=32)
val_input_fn = get_input_fn(x_val_flat, y_val, batch_size=20000)

async def test():
    return {console.log("onTrainBegin")}

estimator.fit(input_fn=train_input_fn, steps=5)

metrics = estimator.evaluate(input_fn=train_input_fn, steps=1)
print("Train Loss", metrics['loss'], "\nAccuracy", metrics['accuracy'])

metrics = estimator.evaluate(input_fn=val_input_fn, steps=1)
print("Val Loss", metrics['loss'], "\nAccuracy", metrics['accuracy'])






   
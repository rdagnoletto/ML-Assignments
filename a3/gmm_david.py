#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:57:06 2019

@author: ragnoletto
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math as ma

# Loading data
data = np.load('data100D.npy')
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)
is_valid = True
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    X2 = tf.multiply(X,X)
    X2 = tf.reduce_sum(X2,axis = 1,keepdims = True)
    MU2 = tf.multiply(tf.transpose(MU), tf.transpose(MU))
    MU2 = tf.reduce_sum(MU2,axis=0,keepdims = True)
    pair_dist = X2 + MU2 - tf.multiply(2.0,tf.matmul(X,tf.transpose(MU)))
    return pair_dist

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO
    dist_matrix = distanceFunc(X,mu)
    sigmaT = tf.reshape(sigma,[-1, K])  
    log_PDF =  -1/2*tf.divide(dist_matrix,sigmaT) - tf.math.log(tf.sqrt(2.0*ma.pi*sigmaT))
    return log_PDF

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    log_post = log_PDF + tf.transpose(log_pi)
    return log_post
    
def MoG():
    X = tf.placeholder(tf.float32,[None, D], name="X")
    MU = tf.get_variable('mean',dtype = tf.float32,shape = [K,D], initializer = tf.initializers.random_normal())
    Psi = tf.get_variable('variance',dtype = tf.float32,shape = [K,1], initializer = tf.initializers.random_normal())
    Pi = tf.get_variable('posterior',dtype = tf.float32,shape = [K,1], initializer = tf.initializers.random_normal())
    
    log_Pi = hlp.logsoftmax(Pi)
    Sigma2 = tf.exp(Psi)
    
    Gauss_PDF = log_GaussPDF(X,MU,Sigma2)
    
    Log_Post = log_posterior(Gauss_PDF,log_Pi)
    Belong = tf.arg_max(Log_Post,dimension = 1)
    lossfunc = -tf.reduce_sum(hlp.reduce_logsumexp(Log_Post))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-5)
    train = optimizer.minimize(loss=lossfunc)
    return X,MU,Psi, Pi,lossfunc,Belong,train
    
D = data.shape[1]
K = 5
tf.reset_default_graph()    
X,mu,psi, pi, error,belongs,train = MoG()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
epochs = 500


for step in range(0, epochs):
    _,current_mu,curr_psi, curr_pi ,current_error = sess.run([train,mu,psi,pi,error],feed_dict = {X:data})
    if step % 10 == 0:
        print(current_error)
        
        
final_belong = sess.run([belongs],feed_dict = {X:data})
validloss = sess.run([error], feed_dict = {X:val_data})
final_belong = final_belong[0]
classdiv = np.zeros(K)
for i in range(0,K):
    classdiv[i] = np.mean(final_belong==i)

    
#RUN TO PLOT THE CLASSES, TAKES LONG TIME DUE TO ITERATION
def plotclasses(data,final_belong,current_mu):
    colors = ['red','green','blue','purple', 'orange']
    for i in range(current_mu.shape[0]):
        idx = np.where(final_belong==i)
        current_class = np.take(data,idx[0],axis=0)
        #print(idx,data.shape,current_class.shape)
        plt.scatter(current_class[:,0],current_class[:,1], color = colors[i])

    plt.grid(True)
    #plt.scatter(mu_x,mu_y, color = 'black')
    for i in range(current_mu.shape[0]):
        plt.scatter(current_mu[i][0],current_mu[i][1], color = 'black')
        plt.annotate(i, (current_mu[i][0],current_mu[i][1]))
    plt.show()
        
 
plotclasses(data,final_belong,current_mu)

sess.close()
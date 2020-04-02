#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:39:29 2019

@author: ragnoletto
"""

def gMM():
    
    X = tf.placeholder(tf.float32,[None, dim], name="X")
    
    #mu_init = np.array([[-1.01,-4.01],[0.01,-1.01],[1.1,0.5]])
#    mu_init = np.zeros((K,2))
#    for i in range(K):
#        y = 3*m.sin((i/K)*2*m.pi)
#        x = 3*m.cos((i/K)*2*m.pi)
#        mu_init[i] = [x,y]
#        print(i,x,y)
    
    #mu_inner = tf.get_variable('mean',dtype = tf.float32,shape = [K,dim], initializer = tf.truncated_normal_initializer(stddev=0.25)) 
#    tfd = tf.contrib.distributions
#    mix = 0.5
#    bimix_gauss = tfd.Mixture(
#            cat=tfd.Categorical(probs=[mix, 1.-mix]),
#            components=[
#                    tfd.Normal(loc=-5., scale=0.5),
#                    tfd.Normal(loc=+5., scale=0.5),
#                    ])
#    mu = tf.get_variable('mean',dtype = tf.float32, initializer = bimix_gauss.sample(sample_shape=(K,dim)))
    mu = tf.get_variable('mean',dtype = tf.float32,shape = [K,dim], initializer = tf.truncated_normal_initializer())

    #mu = tf.get_variable('mean',dtype = tf.float32, initializer = tf.to_float(mu_init))
    
    #testing = tf.get_variable('test',dtype = tf.float32,shape = [K,1], initializer = tf.initializers.random_normal())
    #sigma_holder = tf.get_variable('stdDev',dtype = tf.float32, initializer = tf.to_float(np.zeros((K,1))))
    #sigma_holder = tf.get_variable('stdDev',dtype = tf.float32, initializer = tf.to_float(np.ones((K,1))))
    phi = tf.get_variable('stdDev',dtype = tf.float32,shape = [K,1],initializer = tf.truncated_normal_initializer(mean=1,stddev=0.1))
#    sigma = tf.exp(sigma_holder)
    
    sigma = tf.abs(phi)
    #sigma = tf.pow(1.3,phi)
    #sigma = tf.pow(phi,2)
    
    #psi = tf.get_variable('logPiProb',dtype = tf.float32, initializer = tf.to_float((1/K)*np.ones((K,1))))
    psi = tf.get_variable('logPiProb',dtype = tf.float32,shape = [K,1], initializer = tf.truncated_normal_initializer(mean=1,stddev=0.2))
    log_pi = hlp.logsoftmax(psi)
    
    log_PDF = log_GaussPDF(X, mu, sigma)
    log_rnj = log_posterior(log_PDF, log_pi)
    lossfunc = neg_log_likelihood(log_PDF, log_pi)
    belong = tf.arg_max(log_rnj,dimension = 1)
    #optimizer = tf.train.GradientDescentOptimizer(0.00005)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.05, beta1=0.9, beta2=0.99, epsilon=1e-5)
    #optimizer = tf.train.MomentumOptimizer(0.00001,0.2)
    train = optimizer.minimize(loss=lossfunc)
    
    return X,mu,sigma,lossfunc,log_pi,log_PDF,log_rnj,train,belong
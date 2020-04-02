import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
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


# Distance function for K-means
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
    
def Kmeans():
    X = tf.placeholder(tf.float32,[None, D], name="X")
    
    MU = tf.get_variable('mean',dtype = tf.float32,shape = [K,D], initializer = tf.initializers.random_normal())
    belong = tf.arg_min(distanceFunc(X,MU),dimension = 1)
    lossfunc = tf.reduce_sum(tf.reduce_min(distanceFunc(X,MU),axis = 1))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-5)
    train = optimizer.minimize(loss=lossfunc)
    
    return X,train, MU,lossfunc, belong

D = data.shape[1]
tf.reset_default_graph()    
X,train,mu,error, belongings = Kmeans()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
epochs = 1000
errors = np.zeros(epochs)

for step in range(0, epochs):
    _,current_mu,current_error = sess.run([train,mu,error],feed_dict = {X:data})
    if step % 10 == 0:
        print(current_error)
    errors[step] = current_error
        
final_belong = sess.run([belongings],feed_dict = {X:data})
validloss,valid_belong = sess.run([error,belongings], feed_dict = {X:val_data})
final_belong = final_belong[0]
final_mu = current_mu
K = 30
classdiv = np.zeros(K)
for i in range(0,K):
    classdiv[i] = np.mean(final_belong==i)
    

    
#RUN TO PLOT THE CLASSES, TAKES LONG TIME DUE TO ITERATION
def plotclasses(data,final_belong,current_mu):
#    colors = ['red','green','blue','purple', 'orange']
    colors = plt.cm.get_cmap('hsv', K+1)
    for i in range(current_mu.shape[0]):
        idx = np.where(final_belong==i)
        current_class = np.take(data,idx[0],axis=0)
        #print(idx,data.shape,current_class.shape)
        plt.scatter(current_class[:,0],current_class[:,1], color = colors(i))

    plt.grid(True)
    #plt.scatter(mu_x,mu_y, color = 'black')
    for i in range(current_mu.shape[0]):
        plt.scatter(current_mu[i][0],current_mu[i][1], color = 'black')
        plt.annotate(i, (current_mu[i][0],current_mu[i][1]))
    plt.title('data classes with K-means')    
    plt.show()

        

plotclasses(data,final_belong,final_mu)
print(classdiv)
print(final_mu)
print(errors[-1])
plt.figure()
plotclasses(val_data, valid_belong, final_mu)
print(validloss)
#print(final_mu)
classdiv_val = np.zeros(K)
for i in range(0,K):
    classdiv_val[i] = np.mean(valid_belong==i)
print(classdiv_val)
plt.figure()
plt.plot(errors)
plt.title('Loss for training data')
sess.close()


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math as m
import collections
import random

# Loading data
data = np.load('data100D.npy')
#data = data[:,4:7]
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)
print(np.shape(data))
is_valid = True
K = 4
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45637)
  #np.random.seed(0)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  train_data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
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
    # Outputs:
    # log Gaussian PDF N X K
    
    d_half = tf.constant(dim/2)
    pi2 = tf.constant(2*m.pi)
    half = tf.constant(0.5)
    # NxDxK difference (xn-muk)
    dist = distanceFunc(X,mu)

    exp = -tf.divide(tf.divide(dist,tf.reshape(sigma,[1,-1])),2)

    fact = tf.sqrt(tf.multiply(pi2, tf.reshape(sigma,[1,-1])))

    log_pdf = exp - tf.log(fact)

    return log_pdf

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    num = log_PDF + tf.reshape(log_pi, [1,-1])
    
    denom = hlp.reduce_logsumexp(num,1,True)
    
    return num - denom

def neg_log_likelihood(log_PDF, log_pi):
    
    return -tf.reduce_sum(hlp.reduce_logsumexp(log_PDF + tf.reshape(log_pi, [1,-1])))


def gMM():
    X = tf.placeholder(tf.float32,[None, dim], name="X")
    mu = tf.get_variable('mean',dtype = tf.float32,shape = [K,dim], initializer = tf.truncated_normal_initializer(stddev=2))
    
    phi = tf.get_variable('stdDev',dtype = tf.float32,shape = [K,1],initializer = tf.truncated_normal_initializer(mean=4,stddev=0.5))

    sigma = tf.abs(phi)

    
    psi = tf.get_variable('logPiProb',dtype = tf.float32,shape = [K,1], initializer = tf.truncated_normal_initializer(mean=1,stddev=0.25))
    log_pi = hlp.logsoftmax(psi)

    log_PDF = log_GaussPDF(X, mu, sigma)
    log_rnj = log_posterior(log_PDF, log_pi)
    lossfunc = neg_log_likelihood(log_PDF, log_pi)
    belong = tf.arg_max(log_rnj,dimension = 1)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.05, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss=lossfunc)
    return X,mu,sigma,lossfunc,log_pi,log_PDF,log_rnj,train,belong


tf.reset_default_graph()    
X,mu,sigma,lossfunc,log_pi,log_PDF,log_rnj,train,belong = gMM()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
epochs = 500
#print(mu.eval())

Loss = np.zeros(epochs)
Sigma = np.zeros((epochs,K))
PI = np.zeros((epochs,K))
for step in range(0, epochs):
    _,current_loss,current_sigma,c_pi = sess.run([train,lossfunc,sigma,log_pi],feed_dict = {X:train_data})
    Loss[step] = current_loss
    Sigma[step] = np.squeeze(current_sigma)
    PI[step] = np.squeeze(c_pi)

    if step % 10 == 0:
        print(step,current_loss)
        
        
final_belong, final_lrnj, final_loss,final_mu,final_sigma,final_log_pi,flog_PDF = sess.run([belong,log_rnj,lossfunc,mu,sigma,log_pi,log_PDF],feed_dict = {X:train_data})
valid_belong,valid_loss = sess.run([belong,lossfunc], feed_dict = {X:val_data})

    


print("Final sigma: ",final_sigma)
print("Final log pi: ",final_log_pi)

plt.plot(Loss)
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('neg_log_likelihood')
plt.title('Loss of Training Data K=%d' % K)


plt.show()
print(collections.Counter(final_belong))
print(collections.Counter(valid_belong))
print("Final Training Loss: ",final_loss)
print("Final Validation Loss: ",valid_loss)



plt.plot(Sigma)
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Sigma of each cluster')
plt.title('Sigma value over epochs K=%d' % K)
plt.show()



plt.plot(PI)
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('PI of each cluster')
plt.title('PI value over epochs K=%d' % K)
plt.show()

#RUN TO PLOT THE CLASSES, TAKES LONG TIME DUE TO ITERATION
def plotclasses(data,final_belong,current_mu):
    colors = ['red','green','blue','purple', 'orange','cyan','magenta','yellow','red','green','blue','purple', 'orange','cyan','magenta','yellow'
              'red','green','blue','purple', 'orange','cyan','magenta','yellow','red','green','blue','purple', 'orange','cyan','magenta','yellow',
              'red','green','blue','purple', 'orange','cyan','magenta','yellow','red','green','blue','purple', 'orange','cyan','magenta','yellow',
              'red','green','blue','purple', 'orange','cyan','magenta','yellow','red','green','blue','purple', 'orange','cyan','magenta','yellow',
              'red','green','blue','purple', 'orange','cyan','magenta','yellow','red','green','blue','purple', 'orange','cyan','magenta','yellow']
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
    plt.title('Gaussian Clusters of 2D data K=%d' % K)
    plt.show()
        
   
plotclasses(train_data,final_belong,final_mu)



## Plot the PDF.
#import matplotlib.pyplot as plt
#x = tf.linspace(-2., 3., int(1e4)).eval()
#plt.plot(x, bimix_gauss.prob(x).eval());

sess.close()







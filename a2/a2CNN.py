import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
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
    
def accuracy(target, prediction):
    prediction = np.argmax(prediction, axis=1)
#    print(prediction)
    equal_by_element = np.equal(target,prediction)
    #print(target.shape, prediction.shape)
    return np.mean(equal_by_element)
    
def buildGraph():
    X = tf.placeholder(tf.float32,[None, 28,28,1], name="X")
#    X = tf.reshape(X, shape=[-1, 28, 28, 1])
    y = tf.placeholder(tf.float32,[None,10], name="y")
    
    W1 = tf.get_variable(name = "W1",dtype = tf.float32, shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b1 = tf.get_variable(name = "b1",dtype = tf.float32, shape=[32], initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv,b1)
    conv_relu = tf.nn.relu(conv)

    mean, var = tf.nn.moments(conv_relu, axes = [0])
    x1 = tf.nn.batch_normalization(conv_relu, mean,var,1,0,1e-3)
    x1 = tf.nn.max_pool(x1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
#    x1 = tf.layers.max_pooling2d(x1,pool_size = [2,2], strides = 2, padding = 'SAME')
    x1_flat = tf.layers.flatten(x1)
    W2 = tf.get_variable(name = "W2",dtype = tf.float32, shape=[6272,784], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b2 = tf.get_variable(name = "b2", dtype = tf.float32,shape=[784], initializer=tf.constant_initializer(1))
    
#    dropout_rate = 0.9 # == 1 - keep_prob
#    x1_dropped = tf.layers.dropout(x1_flat, dropout_rate)
    
    s2 = tf.matmul(x1_flat,W2)  
#    s2 = tf.matmul(x1_dropped,W2)
    
    s2 = tf.nn.bias_add(s2,b2)
    x2 = tf.nn.relu(s2)
    W3 = tf.get_variable(name = "W3", dtype = tf.float32, shape=[784,10], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b3 = tf.get_variable(name = "b3",dtype = tf.float32, shape=[10], initializer=tf.constant_initializer(1))

    s3 = tf.matmul(x2,W3)
    s3 = tf.nn.bias_add(s3,b3)
    
    y_pred = tf.nn.softmax(s3)
    
    #NORMAL CE
    CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y,logits = s3), name='cross_entropy')

    #CE USING L2 REGULARIZATION
#    CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y,logits = s3) 
#                + 0.5*tf.nn.l2_loss(W1) + 0.5*tf.nn.l2_loss(W2) + 0.5*tf.nn.l2_loss(W3), name='cross_entropy')


    optimizer = tf.train.AdamOptimizer(learning_rate = 0.000001, epsilon = 1e-04)
    train = optimizer.minimize(loss=CE)
    return X,y,W1,W2,W3,b1,b2,b3,y_pred, CE,train

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()   
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget) 
#trainData = np.reshape(trainData, shape=[-1, 28, 28, 1])
tf.reset_default_graph()
epochs = 50
batch_size = 32
number_of_batches = len(newtrain)/batch_size
trainerror = np.zeros(epochs)
trainacc = np.zeros(epochs)
trainData = trainData.reshape((-1,28,28,1))


X, y,W1,W2,W3,b1,b2,b3,y_pred, CEloss, train = buildGraph()
#config=tf.ConfigProto(log_device_placement=True)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
#saver = tf.train.Saver()
for i in range(0,epochs):
    trainData, newtrain = shuffle(trainData,newtrain)

    for batch_nbr in range(0,int(number_of_batches)):
        batch_trainData = trainData[batch_nbr*batch_size+1:(batch_nbr+1)*batch_size]                            
        batch_trainTarget = newtrain[batch_nbr*batch_size+1:(batch_nbr+1)*batch_size]
        _,currentW1, currentW2, currentW3,currentb1,currentb2, currentb3, currentLoss, yhat = sess.run([train,W1,W2, W3,b1,b2,b3, CEloss, y_pred], feed_dict={X: batch_trainData, y: batch_trainTarget})
#        print(currentb1)
    if number_of_batches%int(number_of_batches) != 0:
        batch_trainData = trainData[int(number_of_batches)*batch_size+1:]
        batch_trainTarget = newtrain[int(number_of_batches)*batch_size+1:]
        _,currentW1, currentW2, currentW3,currentb1,currentb2, currentb3, currentLoss, yhat = sess.run([train,W1,W2,W3,b1,b2,b3,CEloss, y_pred], feed_dict={X: batch_trainData, y: batch_trainTarget})
    if i % 10 == 0:
#        save_path = saver.save(sess, "/tmp/nol2.ckpt") 
        print("Iteration: %d, error-training: %.5f" %(i, currentLoss))
    trainerror[i] = currentLoss

testData = testData.reshape((-1,28,28,1))
validData = validData.reshape((-1,28,28,1))

errTrain,ytrain = sess.run([CEloss,y_pred] , feed_dict = {X: trainData, y : newtrain})

errTest,ytest = sess.run([CEloss, y_pred], feed_dict = {X: testData, y: newtest})

errValid, yvalid = sess.run([CEloss,y_pred], feed_dict = {X: validData, y: newvalid})

TrainAcc = accuracy(trainTarget,ytrain)
ValidAcc = accuracy(validTarget,yvalid)
TestAcc = accuracy(testTarget,ytest)

print("Final training Error: %.5f:" % (errTrain))
print("Final Validation Error: %.5f:" % (errValid))  
print("Final Testing Error: %.5f:" % (errTest)) 
print("Final training Accuracy: %.5f:" % (TrainAcc))
print("Final Validation Accuracy: %.5f:" % (ValidAcc))  
print("Final Testing Accuracy: %.5f:" % (TestAcc)) 

fig, ax = plt.subplots()
ax.plot(trainerror)

ax.set(xlabel='iterations', ylabel='trainerror',
       title='Error from the training Data')
ax.grid()

plt.show()    
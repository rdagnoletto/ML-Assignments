import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import sys

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSEloss(W, b, x, y, reg):

    mse = (np.linalg.norm(np.matmul(x,W) + b - y)**2)/(2*len(y))
    wdl = (np.linalg.norm(W)**2)*(reg/2)
    return  mse + wdl

def grad_MSE(W, b, x, y, reg):

    gradW = np.matmul(np.transpose(x),(np.matmul(x,W) + b - y))/(len(y)) + W*reg
    gradB = np.mean((np.matmul(x,W) + b - y))

    return gradW, gradB

def crossEntropyLoss(W, b, x, y, reg):
    matrixMult = (np.matmul(x,W) + b)

    CEloss = np.mean(-(y*matrixMult) + np.log(1+np.exp(matrixMult))) + reg/2*np.linalg.norm(W)**2
    return CEloss
    
def grad_CE(W, b, x, y, reg):
    matrixMult = (np.matmul(x,W) + b)
    
    gradW = np.matmul(np.transpose(x),(-(y - 1/(1+np.exp(-matrixMult)))))/len(y) + reg*W
    gradB = np.mean(-(y - 1/(1+np.exp(-matrixMult))))

    return gradW,gradB

    
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType="None"):
    loss = np.zeros(epochs)
    accuracy = np.zeros(epochs)
    if lossType == "CE":
        for i in range(epochs): 
            gt = grad_CE(W,b,x,y,reg)
            W = W - alpha * gt[0]
            b = b - alpha * gt[1]
            loss[i] = crossEntropyLoss(W,b,x,y,reg)
            accuracy[i] = calcAccuracy(W, b, x, y, "Log")
            if npl.norm(alpha*gt[0]) < error_tol:
                return W,b,loss[:i],accuracy[:i]
        
    else:        
        
        for i in range(epochs): 
            gt = grad_MSE(W,b,x,y,reg)
            W = W - alpha * gt[0]
            b = b - alpha * gt[1]
            loss[i] = MSEloss(W,b,x,y,reg)
            accuracy[i] = calcAccuracy(W, b, x, y, "Lin")
            if npl.norm(alpha*gt[0]) < error_tol:
                return W,b,loss[:i],accuracy[:i]
            
    return W,b,loss,accuracy

    
def calcAccuracy(W, b, x, y, regType="Log"):
    matrixMult = (np.matmul(x,W) + b)
    if regType == "Lin":
        y2 = matrixMult
        correct = 0
        for i in range(len(y)):
            guess = 0
            if y2[i] > 0.5:
                guess = 1
            if guess == y[i]:
                correct += 1
        
        return correct/len(y)
    else:
        y2 = 1/(1+np.exp(-matrixMult))
        correct = 0
        for i in range(len(y)):
            guess = 0
            if y2[i] > 0.5:
                guess = 1
            if guess == y[i]:
                correct += 1
        return correct/len(y)
            
            

#def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()



epochs = 5000
reg = 0.1
alpha = 0.0001
error_tol = 1e-07

N,n,m = trainData.shape
W0 = np.zeros((n*m,1)) #inital guess att zeros
#W0 = np.random.uniform(size=(784,1))
print(N,n,m)
quit



b0 = np.zeros((1,1))
N,n,m = trainData.shape
trainData = trainData.reshape(-1,n*m)
validData = validData.reshape(-1,n*m)
testData = testData.reshape(-1,n*m)
CE = grad_descent(W0,b0,trainData,trainTarget,alpha,epochs,reg,error_tol,lossType="CE")
MSE = grad_descent(W0,b0,trainData,trainTarget,alpha,epochs,reg,error_tol,lossType="MSE")


plt.subplot(2, 1, 1)
plt.plot(CE[2])
plt.grid(True)
#plt.xlabel('Epoch')
plt.ylabel('CE Loss')
plt.title('Logistic Regression')
plt.axis([0, len(CE[2]), 0, 0.5])

plt.subplot(2, 1, 2)
plt.plot(CE[3])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.axis([0, len(CE[3]), 0.95, 1])


plt.show()


plt.subplot(2, 1, 1)
plt.plot(MSE[2])
plt.grid(True)
#plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Linear Regression')
plt.axis([0, len(MSE[2]), 0, 0.5])

plt.subplot(2, 1, 2)
plt.plot(CE[3])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.axis([0, len(MSE[3]), 0.95, 1])


plt.show()

print(crossEntropyLoss(CE[0], CE[1], trainData, trainTarget, reg))
print(crossEntropyLoss(CE[0], CE[1], validData, validTarget, reg))
print(crossEntropyLoss(CE[0], CE[1], testData, testTarget, reg))
print("0")
print(crossEntropyLoss(CE[0], CE[1], trainData, trainTarget, 0))
print(crossEntropyLoss(CE[0], CE[1], validData, validTarget, 0))
print(crossEntropyLoss(CE[0], CE[1], testData, testTarget, 0))
print("acc")
print(calcAccuracy(CE[0], CE[1], trainData, trainTarget, "Log"))
print(calcAccuracy(CE[0], CE[1], validData, validTarget, "Log"))
print(calcAccuracy(CE[0], CE[1], testData, testTarget, "Log"))


print(MSEloss(MSE[0], MSE[1], trainData, trainTarget, reg))
print(MSEloss(MSE[0], MSE[1], validData, validTarget, reg))
print(MSEloss(MSE[0], MSE[1], testData, testTarget, reg))
print("0")
print(MSEloss(MSE[0], MSE[1], trainData, trainTarget, 0))
print(MSEloss(MSE[0], MSE[1], validData, validTarget, 0))
print(MSEloss(MSE[0], MSE[1], testData, testTarget, 0))
print("acc")
print(calcAccuracy(MSE[0], MSE[1], trainData, trainTarget, "Lin"))
print(calcAccuracy(MSE[0], MSE[1], validData, validTarget, "Lin"))
print(calcAccuracy(MSE[0], MSE[1], testData, testTarget, "Lin"))

#MSE = grad_descent(W0,b0,trainData,trainTarget,alpha,epochs,reg,error_tol,lossType="MSE")


#trainData = np.reshape(trainData, (-1,28*28))
#trainTarget = np.hstack(trainTarget)

#validData = np.reshape(validData, (-1,28*28))
#validTarget = np.hstack(validTarget)

#testData = np.reshape(testData, (-1,28*28))
#testTarget = np.hstack(testTarget)
#print(trainTarget)
#grad_CE(W0,b0,trainData,trainTarget,reg)
#MSE(W,b,trainData, trainTarget,reg)

#W005, b005, loss = grad_descent(W0,b0,trainData,trainTarget,0.005,epochs,0.1,EPS,lossType="CE")


#W001, b001 = grad_descent(W0,b0,trainData,trainTarget,0.001,epochs,0,EPS)
#W0001, b0001 = grad_descent(W0,b0,trainData,trainTarget,0.0001,epochs,0,EPS)
#
#trainMSE005 = MSE(W005,b005,trainData, trainTarget,reg)
#validMSE005 = MSE(W005,b005,validData, validTarget,reg)
#testMSE005 = MSE(W005,b005,testData, testTarget,reg)
#
#trainMSE001 = MSE(W001,b001,trainData, trainTarget,reg)
#validMSE001 = MSE(W001,b001,validData, validTarget,reg)
#testMSE001 = MSE(W001,b001,testData, testTarget,reg)
#
#trainMSE0001 = MSE(W0001,b0001,trainData, trainTarget,reg)
#validMSE0001 = MSE(W0001,b0001,validData, validTarget,reg)
#testMSE0001 = MSE(W0001,b0001,testData, testTarget,reg)
#
#print(trainMSE005,validMSE005,testMSE005)
#print(trainMSE001,validMSE001,testMSE001)
#print(trainMSE0001,validMSE0001,testMSE0001)

#W001, b001 = grad_descent(W0,b0,trainData,trainTarget,0.005,epochs,0.001,EPS)
#W1, b1 = grad_descent(W0,b0,trainData,trainTarget,0.005,epochs,0.1,EPS)
#W5, b5 = grad_descent(W0,b0,trainData,trainTarget,0.005,epochs,0.5,EPS)
#
#trainMSE001 = MSE(W001,b001,trainData, trainTarget,reg)
#validMSE001 = MSE(W001,b001,validData, validTarget,reg)
#testMSE001 = MSE(W001,b001,testData, testTarget,reg)
#
#trainMSE1 = MSE(W1,b1,trainData, trainTarget,reg)
#validMSE1 = MSE(W1,b1,validData, validTarget,reg)
#testMSE1 = MSE(W1,b1,testData, testTarget,reg)
#
#trainMSE5 = MSE(W5,b5,trainData, trainTarget,reg)
#validMSE5 = MSE(W5,b5,validData, validTarget,reg)
#testMSE5 = MSE(W5,b5,testData, testTarget,reg)
#
#print(trainMSE001,validMSE001,testMSE001)
#print(trainMSE1,validMSE1,testMSE1)
#print(trainMSE5,validMSE5,testMSE5)















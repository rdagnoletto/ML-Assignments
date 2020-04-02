 import numpy as np
import numpy.linalg as npl
import tensorflow as tf
import matplotlib.pyplot as plt

with np.load('notMNIST.npz') as data : 
    Data, Target = data ['images'], data['labels']
    posClass = 2
    negClass = 9 
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600] 
    testData, testTarget = Data[3600:], Target[3600:]


def MSE(W,b,x,y,reg):
    N = np.size(y,0)
#    WT = np.transpose(W)
#    xT = np.transpose(x)
#    yT = np.transpose(y)
#    bT = np.transpose(b)
#    mse = (WT@xT@x@W - 2@WT@xT@y + yT@y + 2*WT@xT@b
#            -2*yT@b + bT@b)/(2*N) + reg/2*WT@W
    mse = (1/(2*N))*npl.norm(np.matmul(x,W) + b - y)**2+(reg/2)*npl.norm(W)**2
    return mse

def grad_MSE(W,b,x,y,reg):
    N = np.size(y,0)
#    WT = np.transpose(W)
#    xT = np.transpose(x)
#    yT = np.transpose(y)
#    nablaw = 1/(N) * np.sum((x@W + b - y)*x,0) + reg*W.T
#    nablab = 1/(N) * np.sum((x@W + b - y), 0)
    nablaw = np.matmul(np.transpose(x),(np.matmul(x,W) + b - y))/N + W*reg
    nablab = np.mean(np.matmul(x,W) + b - y)

#    nablaw = (2*xT@x@W - 2*xT@y + 2*xT@b)/(2*N) + reg/2*W
#    nablab = 2*WT@xT - 2*yT + 2*b
    return nablaw, nablab
    
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType="None"):
    loss = np.zeros(epochs)
    if lossType == "CE":
        for i in range(epochs): 
    #        print(i)
            gt = grad_CE(W,b,x,y,reg)
            Wopt = W - alpha * gt[0]
            b = b - alpha * gt[1]
            W = Wopt
            if i % 100 == 0:
                print(gt[0])
            loss[i] = crossEntropyLoss(W,b,x,y,reg)
            if npl.norm(alpha*gt[0]) < error_tol:
                print("YO")
                return Wopt,b,loss[:i]
        
    else:        
        
        for i in range(epochs): 
    #        print(i)
            gt = grad_MSE(W,b,x,y,reg)
            Wopt = W - alpha * gt[0]
            b = b - alpha * gt[1]
            W = Wopt
            loss[i] = MSE(W,b,x,y,reg)
            if npl.norm(alpha*gt[0]) < error_tol:
                return Wopt,b,loss[:i]
            
    return Wopt,b,loss

def loss_plot(W,b,x,y):
    loss = (np.matmul(x,W) + b - y)**2
    plt.plot(loss)
    return 0
    
def crossEntropyLoss(W, b, x, y, reg):
    
    CEloss = np.mean((1-y)*(np.matmul(x,W) + b) + np.log(1+np.exp(-(np.matmul(x,W) + b)))) + reg/2*npl.norm(W)**2
    CEloss2 = np.mean(-y*(np.matmul(x,W) + b) + np.log(1+np.exp((np.matmul(x,W) + b)))) + reg/2*npl.norm(W)**2
    print(CEloss==CEloss2)
    return CEloss
    
def grad_CE(W, b, x, y, reg):
    nablaw = np.mean((x - x/(1+np.exp(np.matmul(x,W) + b)) - y*x))
    nablab = np.mean((1-y) + (-1)/(1+np.exp(np.matmul(x,W) + b)))
    
    return nablaw,nablab



epochs = 5000
reg = 0.001
alpha = 0.005
error_tol = 1e-07

N,n,m = trainData.shape
W0 = np.zeros((n*m,1)) #inital guess att zeros
b = 0;
trainData = trainData.reshape(N,n*m);
Wopt = grad_descent(W0,b,trainData,trainTarget,alpha,epochs,reg,error_tol,"CE")
W = Wopt[0]
b = Wopt[1]
print(Wopt[2])
#plt.plot(Wopt[2])

#Now Calculate Validation and test error.
#validData = validData.reshape(len(validTarget), np.size(validData,1)*np.size(validData,2))
#testData = testData.reshape(len(testTarget), np.size(testData,1)*np.size(testData,2))
#plt.figure()

#         W_LS = np.matmul(npl.inv(np.inner(trainData,trainData)),np.inner(tainData,trainTarget))

#loss_plot(W,b,validData,validTarget)
#print(MSE(W,b,validData,validTarget,reg))
#print(MSE(W,b,testData,testTarget,reg))
#plt.figure()
#loss_plot(W,b,testData,testTarget)
#validError = MSE(W,b,validData,validTarget,reg)
#testError = MSE(W,b,testData,testTarget,reg)


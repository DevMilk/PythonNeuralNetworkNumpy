#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
import matplotlib as mb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import optimizers
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




#OUR GOAL IS TO TRAIN OUR NEURAL NETWORK FOR X^2+Y^2=Z EQUATION
aralik = 2 #Set range
outnorm=2*pow(aralik,2)
sampleSize=50
dataSet = np.zeros((sampleSize,3),dtype='double')
for i in range(0,sampleSize):
    dataSet[i,0]=random.random()*(aralik)
    dataSet[i,1]=random.random()*(aralik)
    dataSet[i,2]=pow(dataSet[i,0],2) + pow(dataSet[i,1],2)
X=np.copy(dataSet[:,0])
Y=np.copy(dataSet[:,1])
Z =np.copy(dataSet[:,2])




#Functions to visualize dataset and results
def Show3DGraph(X,Y,Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y,Z)
    plt.show()
def compare3D(X,Y,Z1,Z2):    
    Z2t=np.copy(Z2).reshape(Z2.shape[0])
    Z1t=np.copy(Z1).reshape(Z1.shape[0])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot_trisurf(X, Y, Z2t,cmap='Oranges')
    ax.plot_trisurf(X, Y, Z1t,cmap='Blues')
    plt.show()
#visualize results we want
Show3DGraph(X,Y,Z)




#Neural Network structure with 3 layers
class ThreeLayerNN:
    def __init__(self, HiddenLayerNeuronCount):
        self.input      = np.zeros((2,1),dtype='double')
        self.weights1   = np.random.rand(2,HiddenLayerNeuronCount) 
        self.weights2   = np.random.rand(HiddenLayerNeuronCount,1)
        self.y          = np.zeros((1,1),dtype='double')
        self.output     = np.zeros((1,1),dtype='double')
        self.d_weights1 = np.copy(self.weights1)
        self.d_weights2 = np.copy(self.weights2)
        self.layer1     =np.zeros((HiddenLayerNeuronCount,1),dtype='double')
        self.b1=0
        self.b2=0
        self.d_b1=0
        self.d_b2=0
        self.simu=[]
        self.counter=0
    def relu(self,x):
        return np.maximum(0,x)
    def relu_derivative(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x)) 
    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*self.sigmoid(1-x)
    
    def feedforward(self):
        dot=np.dot(self.input.T, self.weights1) +self.b1
        self.layer1 = self.sigmoid(dot)
        dot=np.dot(self.layer1, self.weights2) +self.b2
        self.output = self.sigmoid(dot)
        #Calculate error 
        return pow(self.y*(outnorm/2)-self.output*(outnorm/2),2)/2
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        mu = ((self.output-self.y) * self.sigmoid_derivative(self.output))
        self.d_weights2 += np.dot(self.layer1.T,mu )
        self.d_b2+=np.sum(mu)   

        mu=(np.dot(mu, self.weights2.T) * self.sigmoid_derivative(self.layer1))
        self.d_weights1 += np.dot(self.input,mu) #Negatif değerlerde sapıtmanın nedeni olabilir
        self.d_b1+=np.sum(mu)

    def train(self,batch_size,error_rate,learning_rate,data):
        dataSize= data.shape[0]
        #Use %20 of dataset for test
        testSize=(int)(dataSize/5)

        
        #Set test and train datas
        testData=np.copy(data[:testSize,:])
        trainData=np.copy(data[testSize:,:])
		#Normalize train datas
        for i in range(0,trainData.shape[0]):
            trainData[i,0]/=aralik
            trainData[i,1]/=aralik
            trainData[i,2]/=outnorm
        #Normalize test datas
        for i in range(0,testData.shape[0]):
            testData[i,0]/=aralik
            testData[i,1]/=aralik
            testData[i,2]/=outnorm
        error=100       
		
        self.simu=[] #We need that for plotting error over each iteration
        self.counter=0
        epochFinish = (int)((dataSize*4/5)/batch_size)        
        while(error>error_rate):
            #Shuffle training Data at each iteration      
            np.random.shuffle(trainData)
        
            for i in range(0,epochFinish):
 
                
                    #Fill delta weights and delta biases with zeros
                    self.d_weights1=np.zeros(self.weights1.shape,dtype='double')
                    self.d_weights2=np.zeros(self.weights2.shape,dtype='double')
                    self.d_b1=0
                    self.d_b2=0
                    for j in range(i*batch_size,(i+1)*batch_size):
                        self.input[0]=trainData[j,0]
                        self.input[1]=trainData[j,1]
                        self.y[0]=trainData[j,2]
                        self.feedforward()
                        self.backprop()
                        
                     #Update Weights and biases  
                    self.weights1 -= self.d_weights1*learning_rate
                    self.weights2 -= self.d_weights2*learning_rate
                    self.b1       -= self.d_b1*learning_rate
                    self.b2       -= self.d_b2*learning_rate
            #Calculate error using testData 
            error=0
            for i in range(0,testSize):
                    self.input[0]=testData[i,0]
                    self.input[1]=testData[i,1]
                    self.y[0]=testData[i,2]
                    error+=self.feedforward()
            error/=testSize
            if(self.counter%1000==0):
                print("Itetion: ",self.counter," Average Error: ",error)
            self.simu.append(error)
            self.counter=self.counter+1
    def simulate(self):
        x= np.arange(0, self.counter, 1)
        y= np.array(self.counter)
        y=np.asarray(self.simu)
        y= y.reshape(y.shape[0])
        return x,y
        
    def result(self,testData):
        testSize= testData.shape[0]
        Result = np.zeros(testSize,dtype='double')
        error=0
        test=np.copy(testData)
        
        for i in range(0,test.shape[0]):
            test[i,0]/=aralik
            test[i,1]/=aralik
            
        for i in range(0,testSize):
            self.input[0] =test[i,0]
            self.input[1] =test[i,1]
            self.feedforward()
            Result[i]=self.output[0]#Results of Neural network
        return Result*outnorm #Reverse Normalize and return value
        





batch_size=10 #10 datas from whole data will be used for test
HiddenNeuron=8 #Hidden Layer's neuron count going to be 8
lastTest = np.zeros((sampleSize,3),dtype='double')

for i in range(0,sampleSize):
    lastTest[i,0]=random.random()*(aralik)
    lastTest[i,1]=random.random()*(aralik)
    lastTest[i,2]=pow(lastTest[i,0],2) + pow(lastTest[i,1],2)


#Test for error_rate=0.1, learning_rate=0.1
myNN= ThreeLayerNN(HiddenNeuron)
myNN.train(batch_size,0.1,0.1,dataSet) #Set parameters and begin training
result1=myNN.result(lastTest[:,:2]) #Give test data to predict its value
x1,y1=myNN.simulate() #X and Y values to plot 


#Test for error_rate=0.05, learning_rate=0.01

myNN= ThreeLayerNN(HiddenNeuron)
myNN.train(batch_size,0.05,0.01,dataSet)
result2=myNN.result(lastTest[:,:2])
x2,y2=myNN.simulate()


#Test for error_rate=0.01, learning_rate=0.05
myNN= ThreeLayerNN(HiddenNeuron)
myNN.train(batch_size,0.01,0.05,dataSet)
result3=myNN.result(lastTest[:,:2]) 
x3,y3=myNN.simulate()




#Function for plotting line plot and will used to plot error over per iteration
def line_plot(x,y):
    plt.plot(x,y)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Over Per Iteration')
    plt.grid(True)
    plt.show()

line_plot(x1,y1)
line_plot(x2,y2)
line_plot(x3,y3)





#Gerçek değerlerle tahminlerin karşılaştırılması. Mavi tahminleri gösteriyor
X=lastTest[:,0]
Y=lastTest[:,1]
Z=lastTest[:,2]
#Compare predictions with target values
compare3D(X,Y,result1,Z)
compare3D(X,Y,result2,Z)
compare3D(X,Y,result3,Z)





XY_test=np.copy(dataSet[:(int)(sampleSize/5),:2])/aralik
Z_test=np.copy(dataSet[:(int)(sampleSize/5),2])/outnorm
XY_train=np.copy(dataSet[(int)(sampleSize/5):,:2])/aralik
Z_train=np.copy(dataSet[(int)(sampleSize/5):,2])/outnorm




#Compare our neural network against Keras model
model = Sequential()
model.add(Dense(HiddenNeuron,activation='sigmoid',input_dim=2))
model.add(Dense(1,activation='sigmoid'))
sgd = optimizers.sgd(learning_rate=0.1)
model.compile(optimizer=sgd,
              loss='mean_squared_error')
model.fit(XY_train, Z_train,
          batch_size=batch_size,
          epochs=1000,
          verbose=1,
         validation_data=(XY_test,Z_test))





#Visualize compare between Keras model and our neural network
compare3D(X,Y,result3,model.predict(lastTest[:,:2])*outnorm)


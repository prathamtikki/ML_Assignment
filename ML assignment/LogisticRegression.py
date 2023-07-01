import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class LogisticRegression:
    def __init__(self, training_data, testing_data, threshold, learning_rate,batch_size):
        self.learning_rate = learning_rate
        self.training_data = training_data
        self.testing_data = testing_data
        self.threshold = threshold
        self.no_train = training_data.shape[0]
        self.no_test = testing_data.shape[0]
        self.training_data = self.training_data.to_numpy()
        self.testing_data = self.testing_data.to_numpy()
        self.W = np.zeros((training_data.shape[1]-2, 1))
        self.B = 0
        self.batch_size = batch_size


    def calc_y(self, x):
        x = x.astype(float)
        return 1/(1+np.exp(-x))

    def train(self, epochs):
        cost_list = []
        X_all = self.training_data[:,2:].transpose()
        

        Y_all = self.training_data[:,1].transpose()
        

        for i in range(self.no_train):
            if (Y_all[i] == 'M'):
                Y_all[i] = 1
            else:
                Y_all[i] = 0

        # intitalise weights W w zeros of shape n,1

        for i in range(epochs):
            for start in range(0,self.no_train,self.batch_size):
                stop = start+self.batch_size
                if(stop>self.no_train):
                    stop = self.no_train
                X = X_all[:,start:stop]
                
                Y = Y_all[start:stop]
               
                z = np.dot(self.W.T, X) + self.B

                # probalistic prediction
                A = self.calc_y(z)

                cost = -(1/(stop-start))*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

                # gradient descent
                dW = (1/(stop-start))*np.dot(A-Y, X.T)
                dB = (1/(stop-start))*np.sum(A - Y)
                self.W = self.W - self.learning_rate * dW.T
                self.B = self.B - self.learning_rate*dB

                
                cost_list.append(cost)

        return self.W, self.B, cost_list
    


    def test(self):
        X = self.testing_data[:,2:].transpose()
        

        Y = self.testing_data[:,1].transpose()
        for i in range(self.no_test):
            if (Y[i] == 'M'):
                Y[i] = 1
            else:
                Y[i] = 0

        Z = np.dot(self.W.T, X)+self.B
        A = self.calc_y(Z)
        correct = 0
        tp = 0
        fp = 0
        fn = 0
    

        for i in range(self.no_test):
            if(A[0][i]>=self.threshold):
                A[0][i] = 1
            else:
                A[0][i] = 0

        for i in range(self.no_test):
            if A[0][i] == Y[i]:
                correct+=1
            if A[0][i] == 1 and Y[i] == 1:
                tp+=1
            elif A[0][i] == 1 and Y[i] == 0: 
                fp +=1
            elif A[0][i] == 0 and Y[i] == 1:
                fn+=1

        acc = correct/self.no_test
        if(self.batch_size == 1):
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            return acc,precision,recall
        else :return acc

     

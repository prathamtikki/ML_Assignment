import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, training_data, testing_data):

        self.training_data = training_data
        self.testing_data = testing_data
        self.no_train = training_data.shape[0]
        self.no_test = testing_data.shape[0]
        self.training_data = self.training_data.to_numpy()
        self.testing_data = self.testing_data.to_numpy()
        self.w = np.zeros(training_data.shape[1]-2)

    def train(self, epochs):
        training_acc=[]
        iteration = 0
        while iteration < epochs:
            count = 0

            for i in range(0,self.no_train):
                Y = self.training_data[i][1]

                y = 0
                if Y == "M":
                    y = 1
                else:
                    y = -1

                x = self.training_data[i][2:]

                wx = (np.dot(self.w, x))*y

                if (wx <= 0):
                    count += 1
                    add = x*y
                    self.w = self.w+(add)

            iteration += 1

            
            training_acc.append(100*(self.no_train-count)/self.no_train)
            if count == 0:
                break


        return training_acc

    def test(self):
        x = np.delete(self.testing_data,[0,1],1)

        results = np.dot(x,self.w)
     
        
        

        Y = self.testing_data[:,1]



        for i in range(self.no_test):
            prediction = 0
            if results[i] <= 0:
                prediction = -1
            else:
                prediction = 1
            results[i] = prediction
            if Y[i] == 'M':
                Y[i] = 1
            else:
                Y[i] = -1
        

        correct = 0
        fp = 0
        fn = 0
        tp = 0

        for i in range(self.no_test):
            if(results[i]==Y[i]):
                    correct+=1
                    if results[i] ==1:
                        tp+=1
            elif results[i] == 1 and Y[i]== -1:
                    fp+=1
            elif results[i] == -1 and Y[i]== 1:
                    fn+=1
        
        accuracy = (correct)/self.no_test
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        return accuracy,precision,recall

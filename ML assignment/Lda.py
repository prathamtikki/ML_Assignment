from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd


class Fischers:
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data

        self.no_train = training_data.shape[0]
        self.no_test = testing_data.shape[0]
        self.training_data = self.training_data.to_numpy()
        self.testing_data = self.testing_data.to_numpy()
        self.w = np.zeros(training_data.shape[1]-2)
        self.decision = 0

    def train(self):
        lda = LDA()
        X = self.training_data[:, 2:]
        Y = self.training_data[:, 1]
        
        X_transformed = lda.fit_transform(X, Y)
        
        X_pos = []
        X_neg = []
        for i in range(self.no_train):
            
            
            
            if Y[i] == 'M':
                X_pos.append(X_transformed[i])
            else:
                X_neg.append(X_transformed[i])

        posav = np.average(X_pos)
        negav = np.average(X_neg)
        posvar = np.std(X_pos)
        negvar = np.std(X_neg)

        self.decision = self.solve(posav,negav,posvar,negvar)[-1]
        

        self.w = lda.coef_

        return X_pos,X_neg,self.decision
       
        

    def test(self):

        X = self.testing_data[:, 2:]
        Y = self.testing_data[:, 1]

        X_transformed  = np.dot(X,self.w.transpose())
  


   
      
        
        correct = 0
        fp = 0
        fn = 0
        tp = 0
        for j in range(self.no_test):
                
                result = ""
                if(X_transformed[j]>=self.decision):
                    result = 'M'
                else:
                    result = 'B'
                if(result==Y[j]):
                    correct+=1
                    if result =="M":
                        tp+=1
                elif result == 'M' and Y[j]== "B":
                    fp+=1
                elif result == 'B' and Y[j]== "M":
                    fn+=1
            

        accuracy = (correct)/self.no_test
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        return accuracy,precision,recall



    def solve(self,m1,m2,std1,std2):
        a = 1/(2*std1**2) - 1/(2*std2**2)
        b = m2/(std2**2) - m1/(std1**2)
        c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
        return np.roots([a,b,c])

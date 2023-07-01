import pandas
import numpy as np


class FeatureEngineering:
    def __init__(self):
        pass

    def normalise(dataset):
        
        for column in dataset:
            if column == 'id' or column == 'diagnosis':
                continue
            dataset[column] = (
                dataset[column]-dataset[column].mean())/dataset[column].std()
        return dataset
    
    def removenans(dataset):
        
            
        for column in dataset:
            if column == 'id' or column == 'diagnosis':
                continue
            dataset[column].fillna(dataset[column].mean(),inplace = True) 
        return dataset
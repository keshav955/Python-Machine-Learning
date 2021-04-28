# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:19:13 2020

@author: keshav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('heart _uncleaned.csv')

dataset.isna().sum()
dataset.dtypes


def check(x):
    try :  
        
        float(x) 
        res = True
        return x
    except : 
        
        print("Not a float") 
        res = False
        return np.NaN
    

# taking Care of nan Values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')

for x in dataset.columns :
 #   print(x)
    dataset[x]=dataset[x].apply(check)   
    impute.fit(dataset.loc[:,[x]])
    #impute.fit(X.iloc[:,2])
    dataset.loc[:,[x]] =impute.transform(dataset.loc[:,[x]])
    #X.iloc[:, 2]=impute.transform(X.iloc[:,2])
    
    
    
    
    

       
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:06:44 2020

@author: keshav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3]


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

def abcd(x):
    if x<40 :
        return 0 
    else :
        return 1
X.iloc[:,1]= X.iloc[:,1].apply(abcd)